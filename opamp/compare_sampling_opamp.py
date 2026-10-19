import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import active_sim_opamp as mlp
import compare_models_opamp as cm
from OPAMP import STUDY_DIR


# ============================================================
# 2段オペアンプ (opamp.sch) の all_sat の境界を MLP で学習するとき、
# 能動学習 (active_sim_opamp.py) とランダムサンプリングで、
# 同じシミュレーション回数でどちらが正確なモデルになるかを比べる
#
#   1. テスト点: compare_models_opamp.py と同じもの (study/opamp_model_compare/test.csv) を使う。
#      無ければここでシミュレーションする (同じ乱数・同じ点数なので、同じ点になる)。
#      MLP と SVM の比較と、同じテスト点で比べられる
#   2. ランダムサンプリング用の点を、能動学習と同じ点数だけ設計空間から一様ランダムに作って
#      シミュレーションする (能動学習の「ランダムな点」と同じ作り方。FACE_POINTS も入る)
#   3. 能動学習の各ラウンドの点数 n で、
#        active : 能動学習でそのラウンドまでに集めた n 点 (実行時と同じ乱数なので同じモデル)
#        random : ランダムサンプリングの点から n 点 (並べ方を変えて N_REPEATS 回)
#      で同じ MLP アンサンブル (active_sim_opamp.py と同じ構成) を学習し、テスト点での誤り率・F1 を比べる
#
#   途中で止めても、もう一度実行すれば続きから再開する。
#   (シミュレーション済みの点と、計算済みの学習曲線は飛ばす)
#
#   出力 (study/opamp_sampling_compare_w2p5/):
#     random_pool.csv : ランダムサンプリングの点のシミュレーション結果
#     failed.csv      : シミュレーションに失敗した点
#     curve.csv       : 点数ごとのテスト誤り率・F1 (active / random)
#     curve.png       : 点数 vs テスト誤り率・F1
#
#   環境変数:
#     CMP_RANDOM  : ランダムサンプリングの点数 (既定は能動学習と同じ点数)
#     CMP_REPEATS : random の並べ方を変えて繰り返す回数 (既定 3)
#     CMP_TEST    : テスト点の数 (compare_models_opamp.py と同じ。既定 20000)
#     NUM_WORKERS : ngspice の並列数 (OPAMP.py と同じ)
#     WORK_DIR    : 作業用ネットリストを置く親フォルダ (OPAMP.py と同じ)
# ============================================================


# ============================================================
# パス
# ============================================================

# (study/opamp_sampling_compare は、設計空間を広げる前の比較の結果なので別にする)
OUT_DIR = os.path.join(
    STUDY_DIR,
    "opamp_sampling_compare_w2p5"
)

POOL_PATH = os.path.join(OUT_DIR, "random_pool.csv")
FAILED_PATH = os.path.join(OUT_DIR, "failed.csv")
CURVE_PATH = os.path.join(OUT_DIR, "curve.csv")
PLOT_PATH = os.path.join(OUT_DIR, "curve.png")


# ============================================================
# 設定
# ============================================================

N_REPEATS = int(os.environ.get("CMP_REPEATS", 3))

# ランダムサンプリングの点を作る乱数 (能動学習の点やテスト点と重ならないよう、別の系列にする)
POOL_SEED = [mlp.SEED, 555]

PARAM_KEYS = mlp.PARAM_KEYS
LABEL = mlp.LABEL

COLORS = {"active": "#d62728", "random": "#7f7f7f"}


# ============================================================
# データ
# ============================================================

def load_sim(path):
    """
    シミュレーション結果を点の番号順に読む
    """

    df = pd.read_csv(path).dropna(subset=PARAM_KEYS + [LABEL])

    return df.sort_values("idx").reset_index(drop=True)


def load_active():
    """
    能動学習の結果。samples.csv より後ろのラウンドの記録 (再開途中など) は使わない
    """

    if not os.path.exists(mlp.SAMPLES_PATH) or not os.path.exists(mlp.ROUNDS_PATH):
        raise SystemExit(f"先に active_sim_opamp.py を実行してください: {mlp.SAMPLES_PATH}")

    active = pd.read_csv(mlp.SAMPLES_PATH).dropna(subset=PARAM_KEYS + [LABEL])
    rounds = pd.read_csv(mlp.ROUNDS_PATH).sort_values("round")
    rounds = rounds[rounds["round"] <= active["round"].max()]

    return active, rounds


# ============================================================
# 学習曲線
# ============================================================

def load_curve():
    if not os.path.exists(CURVE_PATH):
        return []
    return pd.read_csv(CURVE_PATH).to_dict("records")


def learning_curve(active, pool, rounds, X_test, y_test):
    """
    能動学習の各ラウンドの点数で、active / random のテスト誤り率を求める
    """

    curve = load_curve()
    done = {(r["strategy"], int(r["repeat"]), int(r["round"])) for r in curve}

    X_act = mlp.to_features(active[PARAM_KEYS].to_numpy())
    y_act = active[LABEL].to_numpy().astype(int)
    round_act = active["round"].to_numpy()

    X_pool = mlp.to_features(pool[PARAM_KEYS].to_numpy())
    y_pool = pool[LABEL].to_numpy().astype(int)

    # random は並べ方を N_REPEATS 通り作り、先頭から n 点を使う
    orders = [
        np.random.default_rng([mlp.SEED, 4242, k]).permutation(len(pool))
        for k in range(N_REPEATS)
    ]

    jobs = [("active", 0)] + [("random", k) for k in range(N_REPEATS)]

    for _, r in rounds.iterrows():

        rnd = int(r["round"])
        n = int(r["n_labeled"])

        for strategy, k in jobs:

            if (strategy, k, rnd) in done:
                continue

            if strategy == "active":
                mask = round_act <= rnd
                X, y = X_act[mask], y_act[mask]
                # active_sim_opamp.py の実行時と同じ乱数
                seed = mlp.SEED + rnd
            else:
                if n > len(pool):
                    continue
                idx = orders[k][:n]
                X, y = X_pool[idx], y_pool[idx]
                seed = mlp.SEED + 1000 * (k + 1) + rnd

            if y.min() == y.max():
                continue

            t0 = time.perf_counter()
            models = mlp.fit_ensemble(X, y, seed, mlp.ENSEMBLE_SIZE)
            pred = (mlp.predict_ensemble(models, X_test) >= 0.5).astype(int)

            row = {
                "strategy": strategy,
                "repeat": k,
                "round": rnd,
                "n": len(y),
                "error": float((pred != y_test).mean()),
                "f1": float(f1_score(y_test, pred)),
                "fit_s": time.perf_counter() - t0,
            }

            curve.append(row)
            pd.DataFrame(curve).to_csv(CURVE_PATH, index=False)

            print(
                f"  [{strategy} #{k}] round {rnd:2d}  n={len(y):6d}  "
                f"error={row['error']:.4f}  F1={row['f1']:.4f}  "
                f"({row['fit_s']:.0f}s)",
                flush=True
            )

    return pd.DataFrame(curve)


def random_mean(curve):
    """
    random の点数ごとの平均・最小・最大
    """

    g = curve[curve["strategy"] == "random"].groupby("n")

    return pd.DataFrame({
        "error": g["error"].mean(),
        "error_min": g["error"].min(),
        "error_max": g["error"].max(),
        "f1": g["f1"].mean(),
        "f1_min": g["f1"].min(),
        "f1_max": g["f1"].max(),
    }).reset_index()


def points_to_reach(rand, target):
    """
    random の平均誤り率が target 以下になる点数 (log-log で線形補間)。
    届かなければ None
    """

    n = rand["n"].to_numpy(dtype=float)
    e = rand["error"].to_numpy()

    for i in range(len(n)):
        if e[i] <= target:
            if i == 0:
                return n[0]
            # 1つ前の点との間で補間
            t = (np.log(target) - np.log(e[i - 1])) / (np.log(e[i]) - np.log(e[i - 1]))
            return float(np.exp(np.log(n[i - 1]) + t * (np.log(n[i]) - np.log(n[i - 1]))))

    return None


# ============================================================
# 図とまとめ
# ============================================================

def plot_curve(curve, n_test, path):

    act = curve[curve["strategy"] == "active"].sort_values("n")
    rand = random_mean(curve)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    for ax, metric, ylabel in [
        (axes[0], "error", "test error rate (1 - accuracy)"),
        (axes[1], "f1", "test F1 (all saturated)"),
    ]:

        ax.fill_between(rand["n"], rand[f"{metric}_min"], rand[f"{metric}_max"],
                        color=COLORS["random"], alpha=0.2, linewidth=0)
        ax.plot(rand["n"], rand[metric], color=COLORS["random"], linewidth=2,
                marker="o", markersize=3,
                label=f"random sampling (mean of {N_REPEATS}, band: min-max)")
        ax.plot(act["n"], act[metric], color=COLORS["active"], linewidth=2,
                marker="o", markersize=3, label="active learning")

        ax.set_xscale("log")
        ax.set_xlabel("number of simulations")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    axes[0].set_yscale("log")

    fig.suptitle(
        "2-stage op-amp all_sat classifier (MLP ensemble): active learning vs random sampling\n"
        f"(test: {n_test} uniform random points not used for training)",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nLearning curve saved: {path}")


def summarize(curve):

    act = curve[curve["strategy"] == "active"].sort_values("n")
    rand = random_mean(curve)

    table = act[["n", "error", "f1"]].merge(
        rand[["n", "error", "f1"]], on="n", how="left",
        suffixes=("_active", "_random")
    )

    print("\n点数ごとのテスト誤り率・F1 (random は平均):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    final = act.iloc[-1]
    n_rand = points_to_reach(rand, final["error"])

    print(f"\n能動学習 {int(final['n'])} 回 の誤り率: {final['error']:.4f}")

    if n_rand is None:
        print(
            f"ランダムサンプリングは {int(rand['n'].max())} 回でもこの誤り率に届かない "
            f"(最終 {rand['error'].iloc[-1]:.4f}, 能動学習の "
            f"{rand['error'].iloc[-1] / final['error']:.1f} 倍)"
        )
    else:
        print(
            f"ランダムサンプリングで同じ誤り率に届く回数: 約 {n_rand:.0f} 回 "
            f"(能動学習の {n_rand / final['n']:.2f} 倍)"
        )


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(cm.OUT_DIR, exist_ok=True)

    active, rounds = load_active()
    cm.check_space("mlp", active)

    n_pool = int(os.environ.get("CMP_RANDOM", len(active)))

    print(f"能動学習: {len(active)} 点, {len(rounds)} ラウンド")
    print(f"ランダムサンプリング {n_pool} 点, random の繰り返し {N_REPEATS} 回")

    # 1. テスト点 (compare_models_opamp.py と共通)
    P_test = mlp.sample_space(np.random.default_rng(cm.TEST_SEED), cm.N_TEST)
    cm.simulate_test(P_test)

    # 2. ランダムサンプリングの点
    P_pool = mlp.sample_space(np.random.default_rng(POOL_SEED), n_pool)
    cm.simulate_points(P_pool, POOL_PATH, FAILED_PATH, "random")

    test = load_sim(cm.TEST_PATH)
    pool = load_sim(POOL_PATH)

    print(f"\nテスト点: {len(test)} 点 (飽和率 {test[LABEL].mean():.3f})")
    print(f"ランダムサンプリング: {len(pool)} 点 (飽和率 {pool[LABEL].mean():.3f})")
    print(f"能動学習: {len(active)} 点 (飽和率 {active[LABEL].mean():.3f})")

    X_test = mlp.to_features(test[PARAM_KEYS].to_numpy())
    y_test = test[LABEL].to_numpy().astype(int)

    # 3. 学習曲線
    print("\n学習曲線を計算中 ...")
    curve = learning_curve(active, pool, rounds, X_test, y_test)

    plot_curve(curve, len(test), PLOT_PATH)
    summarize(curve)


if __name__ == "__main__":
    main()
