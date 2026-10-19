import os
import csv
import time
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score

import active_sim_opamp as mlp
import active_sim_opamp_svm as svm
from OPAMP import (
    export_netlist,
    simulate_point,
    FIELDNAMES,
    DESIGN_KEYS,
    SCH_PATH,
    NETLIST_DIR,
    STUDY_DIR,
)


# ============================================================
# 2段オペアンプ (opamp.sch) の all_sat の能動学習を、MLP 版と SVM 版で比べる学習曲線
#
#   active_sim_opamp.py (MLP) と active_sim_opamp_svm.py (SVM) の結果を使う。
#
#   1. テスト用の点を設計空間から一様ランダムに N_TEST 点作ってシミュレーションする
#      (どちらの学習にも使っていない共通の正解。点の作り方は能動学習のランダムな点と同じ)
#   2. 各ラウンドの点数で、次の組み合わせのモデルを学習し直してテスト点での誤り率・F1 を求める
#        MLP / MLP の点 : MLP 版の能動学習そのもの (実行時と同じ乱数なので同じモデルになる)
#        SVM / SVM の点 : SVM 版の能動学習そのもの
#        SVM / MLP の点 : (CMP_CROSS=1 のとき) MLP が選んだ点で SVM を学習
#        MLP / SVM の点 : (CMP_CROSS=1 のとき) SVM が選んだ点で MLP を学習
#      交差させた組み合わせで、差が「モデルの違い」と「選んだ点の違い」のどちらから来るかを見る
#   3. 学習曲線 (点数 vs 誤り率・F1、経過時間 vs 誤り率) を描く
#
#   途中で止めても、もう一度実行すれば続きから再開する。
#   (シミュレーション済みのテスト点と、計算済みの学習曲線は飛ばす)
#
#   出力 (study/opamp_model_compare/):
#     test.csv     : テスト点のシミュレーション結果
#     failed.csv   : シミュレーションに失敗したテスト点
#     curve.csv    : 組み合わせ・ラウンドごとの点数、誤り率、F1、学習時間
#     curve.png    : 点数 vs 誤り率 / 点数 vs F1 / 経過時間 vs 誤り率
#
#   環境変数:
#     CMP_TEST    : テスト点の数 (既定 20000)
#     CMP_CROSS   : 0 にすると交差させた組み合わせを計算しない (既定 1)
#     NUM_WORKERS : ngspice の並列数 (OPAMP.py と同じ)
#     WORK_DIR    : 作業用ネットリストを置く親フォルダ (OPAMP.py と同じ)
#     SVM_JOBS    : SVM の予測の並列数 (active_sim_opamp_svm.py と同じ)
# ============================================================


# ============================================================
# パス
# ============================================================

OUT_DIR = os.path.join(
    STUDY_DIR,
    "opamp_model_compare"
)

TEST_PATH = os.path.join(OUT_DIR, "test.csv")
FAILED_PATH = os.path.join(OUT_DIR, "failed.csv")
CURVE_PATH = os.path.join(OUT_DIR, "curve.csv")
PLOT_PATH = os.path.join(OUT_DIR, "curve.png")


# ============================================================
# 設定
# ============================================================

N_TEST = int(os.environ.get("CMP_TEST", 20000))
USE_CROSS = os.environ.get("CMP_CROSS", "1") != "0"

# テスト点を作る乱数 (能動学習の点と重ならないよう、別の系列にする)
TEST_SEED = [mlp.SEED, 777]

PARAM_KEYS = mlp.PARAM_KEYS
LABEL = mlp.LABEL

# 能動学習の結果 (点を選んだモデル → samples.csv, rounds.csv)
RUNS = {
    "mlp": (mlp.SAMPLES_PATH, mlp.ROUNDS_PATH),
    "svm": (svm.SAMPLES_PATH, svm.ROUNDS_PATH),
}

# 学習に使うモデル (学習する関数, 予測する関数)
MODELS = {
    "mlp": (mlp.fit_ensemble, mlp.predict_ensemble),
    "svm": (svm.fit_svm, svm.predict_svm),
}

# 図の色はモデルで、線の種類は「どちらが選んだ点か」で分ける
COLORS = {"mlp": "#d62728", "svm": "#1f77b4"}
NAMES = {"mlp": "MLP", "svm": "SVM"}


# ============================================================
# テスト点のシミュレーション
# ============================================================

def done_indices():
    idx = set()
    for path in [TEST_PATH, FAILED_PATH]:
        if os.path.exists(path) and os.path.getsize(path):
            idx |= set(pd.read_csv(path, usecols=["idx"])["idx"].astype(int))
    return idx


def simulate_test(P):
    """
    テスト点をシミュレーションし、終わった順に test.csv に追記する (記録済みの点は飛ばす)
    """

    for path in [TEST_PATH, FAILED_PATH]:
        mlp.repair_tail(path)

    mlp.init_csv(TEST_PATH, ["idx"] + FIELDNAMES)
    mlp.init_csv(FAILED_PATH, ["idx"] + PARAM_KEYS + ["error"])

    skip = done_indices()
    todo = [i for i in range(len(P)) if i not in skip]

    print(f"テスト点: {len(P)} 点 (済 {len(P) - len(todo)}, 残り {len(todo)})")

    if not todo:
        return

    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    if not spice_template or not os.path.exists(spice_template):
        raise SystemExit("Failed to generate base netlist. Please check xschem execution.")

    with open(spice_template, "r") as src:
        template = src.read()

    num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))
    work_dir = tempfile.mkdtemp(prefix="opamp_cmp_", dir=os.environ.get("WORK_DIR"))

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor, \
                open(TEST_PATH, "a", newline="") as f, \
                open(FAILED_PATH, "a", newline="") as ff:

            writer = csv.DictWriter(f, fieldnames=["idx"] + FIELDNAMES)
            failed = csv.DictWriter(ff, fieldnames=["idx"] + PARAM_KEYS + ["error"])

            futures = {}
            for i in todo:
                v = dict(zip(PARAM_KEYS, (float(x) for x in P[i])))
                sizes = {k: v[k] for k in DESIGN_KEYS}
                fut = executor.submit(
                    simulate_point, template, work_dir, f"test_{i}",
                    sizes, v["Ibias_uA"], v["Vin_V"]
                )
                futures[fut] = (i, v)

            t0 = time.perf_counter()

            for k, fut in enumerate(as_completed(futures), 1):
                i, v = futures[fut]

                try:
                    row, err = fut.result()
                except Exception as e:
                    row, err = None, f"ERROR: {e}"

                if row is None:
                    failed.writerow({"idx": i, **v, "error": err})
                    print(err)
                else:
                    writer.writerow({"idx": i, **row})

                if k % 1000 == 0 or k == len(futures):
                    f.flush()
                    ff.flush()
                    el = time.perf_counter() - t0
                    print(
                        f"    sim {k}/{len(futures)}  ({el:.0f}s, "
                        f"残り約 {el / k * (len(futures) - k):.0f}s)",
                        flush=True
                    )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================
# 学習曲線
# ============================================================

def load_run(name):
    """
    能動学習の結果 (samples.csv と rounds.csv)。
    samples.csv より後ろのラウンドの記録 (再開途中など) は使わない
    """

    samples_path, rounds_path = RUNS[name]

    if not os.path.exists(samples_path) or not os.path.exists(rounds_path):
        raise SystemExit(f"{NAMES[name]} 版の能動学習の結果がありません: {samples_path}")

    samples = pd.read_csv(samples_path).dropna(subset=PARAM_KEYS + [LABEL])
    rounds = pd.read_csv(rounds_path).sort_values("round")
    rounds = rounds[rounds["round"] <= samples["round"].max()]

    return samples, rounds


def check_space(name, samples):
    """
    能動学習の点が今の PARAM_SPACE の範囲に入っているか (設定を変えた後の古い結果でないか)
    """

    for k, (lo, hi, _, _) in mlp.PARAM_SPACE.items():
        out = ((samples[k] < lo - 1e-6) | (samples[k] > hi + 1e-6)).sum()
        if out:
            print(f"  WARNING: {NAMES[name]} 版の点のうち {out} 点が今の {k} の範囲 "
                  f"[{lo}, {hi}] の外です。設定を変える前の結果かもしれません")


def load_curve():
    if not os.path.exists(CURVE_PATH):
        return []
    return pd.read_csv(CURVE_PATH).to_dict("records")


def learning_curve(runs, X_test, y_test):

    curve = load_curve()
    done = {(r["model"], r["data"], int(r["round"])) for r in curve}

    # (学習するモデル, 点を選んだモデル)
    combos = [("mlp", "mlp"), ("svm", "svm")]
    if USE_CROSS:
        combos += [("svm", "mlp"), ("mlp", "svm")]

    for model, data in combos:

        samples, rounds = runs[data]
        fit, predict = MODELS[model]

        X_all = mlp.to_features(samples[PARAM_KEYS].to_numpy())
        y_all = samples[LABEL].to_numpy().astype(int)
        round_all = samples["round"].to_numpy()

        print(f"\n{NAMES[model]} を {NAMES[data]} が選んだ点で学習 ({len(rounds)} ラウンド)")

        for _, r in rounds.iterrows():

            rnd = int(r["round"])

            if (model, data, rnd) in done:
                continue

            mask = round_all <= rnd
            X, y = X_all[mask], y_all[mask]

            if y.min() == y.max():
                continue

            t0 = time.perf_counter()
            # 能動学習を実行したときと同じ乱数 (MLP の初期値) で学習する
            models = fit(X, y, mlp.SEED + rnd, mlp.ENSEMBLE_SIZE)
            fit_s = time.perf_counter() - t0

            pred = (predict(models, X_test) >= 0.5).astype(int)

            row = {
                "model": model,
                "data": data,
                "round": rnd,
                "n": len(y),
                "error": float((pred != y_test).mean()),
                "f1": float(f1_score(y_test, pred)),
                "fit_s": fit_s,
                # 能動学習を実行したとき、このラウンドの学習までにかかった時間
                "elapsed_s": float(r["elapsed_s"]),
            }

            curve.append(row)
            pd.DataFrame(curve).to_csv(CURVE_PATH, index=False)

            print(
                f"  round {rnd:2d}  n={len(y):6d}  error={row['error']:.4f}  "
                f"F1={row['f1']:.4f}  ({fit_s:.0f}s)",
                flush=True
            )

    return pd.DataFrame(curve)


# ============================================================
# 図とまとめ
# ============================================================

def label_of(model, data):
    if model == data:
        return f"{NAMES[model]} (own active-learning samples)"
    return f"{NAMES[model]} trained on {NAMES[data]}'s samples"


def plot_curve(curve, path):

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # 凡例の順: 自分の点で学習した2本 → 交差させた2本
    order = [("mlp", "mlp"), ("svm", "svm"), ("svm", "mlp"), ("mlp", "svm")]

    for model, data in order:

        g = curve[(curve["model"] == model) & (curve["data"] == data)]

        if g.empty:
            continue

        g = g.sort_values("n")
        own = model == data
        style = dict(
            color=COLORS[model],
            linestyle="-" if own else "--",
            linewidth=2 if own else 1.5,
            marker="o" if own else None,
            markersize=3,
            label=label_of(model, data),
        )

        axes[0].plot(g["n"], g["error"], **style)
        axes[1].plot(g["n"], g["f1"], **style)

        # 経過時間は、その方法で能動学習を実行したときの時間なので、自分の点で学習した線だけ描く
        if own:
            axes[2].plot(g["elapsed_s"] / 60, g["error"], **style)

    axes[0].set_ylabel("test error rate (1 - accuracy)")
    axes[1].set_ylabel("test F1 (all saturated)")
    axes[2].set_ylabel("test error rate (1 - accuracy)")

    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("number of simulations")

    axes[2].set_xlabel("elapsed time of the active-learning run [min]\n(simulation + training + point selection)")

    for ax in [axes[0], axes[2]]:
        ax.set_yscale("log")

    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7)

    fig.suptitle(
        "2-stage op-amp all_sat active learning: MLP vs SVM\n"
        f"(test: {N_TEST} uniform random points not used for training)",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nLearning curve saved: {path}")


def points_to_reach(g, target):
    """
    誤り率が初めて target 以下になったときの点数 (届かなければ None)
    """

    g = g.sort_values("n")
    hit = g[g["error"] <= target]

    return None if hit.empty else int(hit["n"].iloc[0])


def summarize(curve):

    print("\n最終ラウンドのテスト誤り率:")

    for (model, data), g in curve.groupby(["model", "data"]):
        last = g.sort_values("n").iloc[-1]
        print(
            f"  {label_of(model, data):40s}  n={int(last['n']):6d}  "
            f"error={last['error']:.4f}  F1={last['f1']:.4f}"
        )

    own = {
        m: curve[(curve["model"] == m) & (curve["data"] == m)].sort_values("n")
        for m in ["mlp", "svm"]
    }

    if any(g.empty for g in own.values()):
        return

    # 相手の最終の誤り率に、何点で届いたか
    print()
    for a, b in [("mlp", "svm"), ("svm", "mlp")]:
        target = own[b]["error"].iloc[-1]
        n = points_to_reach(own[a], target)
        if n is None:
            print(f"  {NAMES[a]} は {NAMES[b]} の最終誤り率 {target:.4f} に届かない")
        else:
            print(f"  {NAMES[a]} は {n} 点で {NAMES[b]} の最終誤り率 {target:.4f} に届いた "
                  f"({NAMES[b]} は {int(own[b]['n'].iloc[-1])} 点)")


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    runs = {name: load_run(name) for name in RUNS}

    for name, (samples, rounds) in runs.items():
        print(f"{NAMES[name]} 版: {len(samples)} 点, {len(rounds)} ラウンド")
        check_space(name, samples)

    # 1. テスト点
    P_test = mlp.sample_space(np.random.default_rng(TEST_SEED), N_TEST)
    simulate_test(P_test)

    test = pd.read_csv(TEST_PATH).dropna(subset=PARAM_KEYS + [LABEL])
    print(f"テスト点: {len(test)} 点 (飽和率 {test[LABEL].mean():.3f})")

    X_test = mlp.to_features(test[PARAM_KEYS].to_numpy())
    y_test = test[LABEL].to_numpy().astype(int)

    # 2. 学習曲線
    curve = learning_curve(runs, X_test, y_test)

    # 3. 図とまとめ
    plot_curve(curve, PLOT_PATH)
    summarize(curve)


if __name__ == "__main__":
    main()
