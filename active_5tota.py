import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
)


# ============================================================
# 5TOTA の all_sat 分類で「能動学習」が全探索より少ない点数で済むかを検証する
#
#   既存の全グリッドデータ 5tota_wsplit_results.csv を「シミュレータの代わり」に使う。
#   (学習用の設計の行 = まだラベルを知らない候補点。1行を問い合わせる = 1回シミュレーションする)
#
#   比較する2つのサンプリング方法:
#     random : 候補点から一様ランダムに追加していく (基準)
#     active : MLP を ENSEMBLE_SIZE 個学習し、平均確率のエントロピーが大きい
#              (= 境界付近で、モデル同士の意見も割れている) 点を優先して追加する
#              - 毎回 RANDOM_FRAC の割合はランダムに選ぶ (未知の飽和領域の見落とし対策)
#              - 上位候補を k-means で BATCH 個に分け、各クラスタから1点ずつ選ぶ
#                (並列シミュレーションを想定した1回分の点が、1か所に固まらないようにする)
#
#   入力・テスト設計の分け方は classify_5tota_wsplit.py と同じ。
#   (同じ SEED なのでテスト設計も同じになり、全データで学習した MLP と比べられる)
#
#   停止条件の候補として、正解 (テスト設計) を使わずに計算できる指標も毎ラウンド記録する:
#     disagree_rate : 停止判定用に固定した候補点 N_STOP 点のうち、
#                     ENSEMBLE_SIZE 個の MLP の判定が全会一致しない点の割合
#     change_rate   : 同じ固定点で、前ラウンドのアンサンブルと予測が変わった点の割合
#     est_err       : 次に追加する点のうち「ランダムに選んだ点」について、
#                     シミュレーション前の予測が外れていた割合 (直近 STOP_WINDOW ラウンド分を合算)
#                     = 正解データなしで推定した誤り率
#   これらを本当のテスト誤り率と並べ、どのしきい値で止めれば十分な精度になるかを調べる。
#
#   出力 (study/5tota_active/):
#     learning_curve.csv  : ラウンドごとの学習点数、テスト精度、停止判定用の指標
#     learning_curve.png  : 学習点数 vs テスト精度 (random / active)
#     stop_metrics.png    : 学習点数 vs 停止判定用の指標
#     stop_rules.csv      : 各停止ルール (指標としきい値) で止めた場合の点数とテスト精度
#     queried_*.png       : active で問い合わせた点を Vin-Vb 平面に重ねた図
#
#   環境変数で設定を変えられる (動作確認用):
#     AL_BUDGET  : 問い合わせる点数の上限
#     AL_REPEATS : 乱数を変えて繰り返す回数
# ============================================================


# ============================================================
# パス
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

STUDY_DIR = os.path.join(
    PROJECT_ROOT,
    "study"
)

CSV_PATH = os.path.join(
    STUDY_DIR,
    "5tota_wsplit_results.csv"
)

# 全データで学習した分類器の結果 (あれば図に基準線として描く)
FULL_METRICS_PATH = os.path.join(
    STUDY_DIR,
    "5tota_wsplit_classifier",
    "metrics.csv"
)

# 結果の保存先
OUT_DIR = os.path.join(
    STUDY_DIR,
    "5tota_active"
)


# ============================================================
# 設定
# ============================================================

# 設計 (W, L の組) を区別する列
DESIGN_KEYS = ["Wn_um", "Wp_um", "Wtail_um", "L_um"]

# 入力にする列
FEATURES = ["log_Wn", "log_Wp", "log_Wtail", "log_L", "Vb_V", "Vin_V"]

LABEL = "all_sat"

# テストに回す設計の割合 (classify_5tota_wsplit.py と同じ)
TEST_DESIGN_RATIO = 0.2

# 最初にランダムに取る点数
N_INIT = 500

# 1ラウンドで追加する点数 (並列シミュレーション1回分のイメージ)
BATCH = 500

# 問い合わせる点数の上限 (学習用候補 約152万点のうち)
BUDGET = int(os.environ.get("AL_BUDGET", 20000))

# 乱数を変えて繰り返す回数 (結果のばらつきを見る)
N_REPEATS = int(os.environ.get("AL_REPEATS", 1))

# active で使う MLP の個数
ENSEMBLE_SIZE = 5

# 1ラウンドの追加点のうち、ランダムに選ぶ割合
RANDOM_FRAC = 0.2

# 候補点全部を毎回スコアするのは重いので、ラウンドごとにこの数だけ抜き出してスコアする
N_CANDIDATES = 200000

# k-means にかける上位候補の数 (= 境界寄りの点の数) は 追加点数 × この倍率
TOP_FACTOR = 10

# 停止判定用に固定する候補点の数
N_STOP = 100000

# est_err を何ラウンド分合算するか (1ラウンドのランダム点だけでは少なくてぶれるため)
STOP_WINDOW = 3

# 停止ルール: 指標がしきい値を STOP_PATIENCE ラウンド続けて下回ったら止める
STOP_PATIENCE = 3
STOP_RULES = {
    "disagree_rate": [0.02, 0.01, 0.005, 0.002],
    "change_rate": [0.01, 0.005, 0.002, 0.001],
    "est_err": [0.02, 0.01, 0.005],
}

SEED = 0


# ============================================================
# データ
# ============================================================

def load_data():
    """
    CSV から必要な列だけを読み、W, L を log に変換した列を足す
    """

    df = pd.read_csv(
        CSV_PATH,
        usecols=DESIGN_KEYS + ["Vb_V", "Vin_V", LABEL]
    )

    for col in ["Wn", "Wp", "Wtail", "L"]:
        df[f"log_{col}"] = np.log(df[f"{col}_um"])

    return df


def split_by_design(df):
    """
    (Wn, Wp, Wtail, L) の組単位で学習用とテスト用に分ける
    (classify_5tota_wsplit.py と同じ分け方)
    """

    designs = df[DESIGN_KEYS].drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    n_test = int(round(len(designs) * TEST_DESIGN_RATIO))
    test_idx = rng.choice(len(designs), size=n_test, replace=False)

    test_designs = designs.iloc[test_idx]

    key = df[DESIGN_KEYS].apply(tuple, axis=1)
    test_keys = set(test_designs.apply(tuple, axis=1))
    is_test = key.isin(test_keys)

    print(
        f"設計数: 全 {len(designs)} / "
        f"学習 {len(designs) - n_test} / テスト {n_test}"
    )

    return (
        df[~is_test].reset_index(drop=True),
        df[is_test].reset_index(drop=True),
    )


# ============================================================
# モデル
# ============================================================

def build_mlp(seed):
    """
    classify_5tota_wsplit.py の MLP と同じ構成。
    データが少ない序盤でも学習できるようにバッチサイズを小さめにする。
    """

    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 64, 32),
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        ),
    )


def fit_one(X, y, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = build_mlp(seed)
        model.fit(X, y)
    return model


def fit_ensemble(X, y, seed, n_models):
    """
    初期値だけ変えた MLP を n_models 個、並列に学習する
    """

    return Parallel(n_jobs=n_models)(
        delayed(fit_one)(X, y, seed * 100 + k)
        for k in range(n_models)
    )


def predict_ensemble(models, X):
    """
    各モデルの確率 (n_models, n_samples)
    """

    return np.stack([m.predict_proba(X)[:, 1] for m in models])


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def stop_metrics(models, X_stop, prev_pred):
    """
    停止判定用の固定点での disagree_rate と change_rate。
    次のラウンドで使うため、今回のアンサンブルの予測も返す。
    """

    probs = predict_ensemble(models, X_stop)
    votes = probs >= 0.5
    pred = probs.mean(axis=0) >= 0.5

    s = {
        "disagree_rate": float((votes.any(axis=0) & ~votes.all(axis=0)).mean()),
        "change_rate": (
            float((pred != prev_pred).mean())
            if prev_pred is not None else np.nan
        ),
    }

    return s, pred


def evaluate(y_true, prob):
    pred = (prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "n_err": int((pred != y_true).sum()),
    }


# ============================================================
# 追加点の選び方
# ============================================================

def select_random(rng, unlabeled, n):
    return rng.choice(unlabeled, size=n, replace=False)


def select_active(rng, models, X_pool, unlabeled, n):
    """
    境界付近 (k-means で散らす) 分と、ランダム分を分けて返す
    """

    n_rand = int(round(n * RANDOM_FRAC))
    n_act = n - n_rand

    # スコアする候補を抜き出す
    cand = unlabeled
    if len(cand) > N_CANDIDATES:
        cand = rng.choice(cand, size=N_CANDIDATES, replace=False)

    prob = predict_ensemble(models, X_pool[cand]).mean(axis=0)
    score = entropy(prob)

    # 境界寄りの上位を k-means で n_act 個に分け、各クラスタで最もスコアが高い点を取る
    n_top = min(len(cand), n_act * TOP_FACTOR)
    top = np.argpartition(-score, n_top - 1)[:n_top]

    X_top = StandardScaler().fit_transform(X_pool[cand[top]])
    km = MiniBatchKMeans(
        n_clusters=n_act,
        batch_size=4096,
        n_init=1,
        random_state=int(rng.integers(1 << 31)),
    ).fit(X_top)

    order = np.lexsort((-score[top], km.labels_))
    first = np.r_[True, km.labels_[order][1:] != km.labels_[order][:-1]]
    picked = cand[top[order[first]]]

    # クラスタが空になって足りない分は、スコア順で埋める
    if len(picked) < n_act:
        rest = np.setdiff1d(cand[top[np.argsort(-score[top])]], picked,
                            assume_unique=True)
        picked = np.r_[picked, rest[:n_act - len(picked)]]

    remaining = np.setdiff1d(unlabeled, picked, assume_unique=True)
    rand = rng.choice(remaining, size=n_rand, replace=False)

    return picked, rand


# ============================================================
# 1回分の実験
# ============================================================

def run(strategy, repeat, X_pool, y_pool, X_test, y_test):
    """
    strategy の方法で BUDGET 点まで問い合わせ、ラウンドごとのテスト精度を返す
    """

    seed = SEED + 1000 * repeat
    rng = np.random.default_rng(seed)

    # 停止判定用の固定点 (追加点の選び方の乱数とは分けておく)
    stop_set = np.random.default_rng(seed + 1).choice(
        len(X_pool), size=min(N_STOP, len(X_pool)), replace=False
    )
    X_stop = X_pool[stop_set]

    labeled = select_random(rng, np.arange(len(X_pool)), N_INIT)
    is_labeled = np.zeros(len(X_pool), dtype=bool)
    is_labeled[labeled] = True

    rows = []
    rnd = 0
    prev_pred = None

    while True:
        t0 = time.perf_counter()

        y_lab = y_pool[labeled]

        # 序盤に片方のクラスしか無いと学習できないので、ランダムに足して待つ
        if y_lab.min() == y_lab.max():
            models = None
        else:
            models = fit_ensemble(X_pool[labeled], y_lab, seed + rnd,
                                  ENSEMBLE_SIZE)

        # 次に追加する点 (rand はそのうちランダムに選んだ点)
        new = rand = None
        if len(labeled) < BUDGET:
            n_add = min(BATCH, BUDGET - len(labeled))
            unlabeled = np.flatnonzero(~is_labeled)

            if strategy == "active" and models is not None:
                picked, rand = select_active(rng, models, X_pool,
                                             unlabeled, n_add)
                new = np.r_[picked, rand]
            else:
                new = rand = select_random(rng, unlabeled, n_add)

        if models is not None:
            prob = predict_ensemble(models, X_test).mean(axis=0)
            m = evaluate(y_test, prob)

            s, prev_pred = stop_metrics(models, X_stop, prev_pred)

            # ランダムに選んだ追加点について、シミュレーション前の予測が外れていた数
            s["n_rand"] = 0
            s["n_rand_err"] = 0
            if rand is not None:
                pred_rand = predict_ensemble(models, X_pool[rand]).mean(axis=0)
                s["n_rand"] = len(rand)
                s["n_rand_err"] = int(
                    ((pred_rand >= 0.5).astype(int) != y_pool[rand]).sum()
                )

            # 直近 STOP_WINDOW ラウンド分を合算して誤り率を推定する
            recent = rows[-(STOP_WINDOW - 1):] if STOP_WINDOW > 1 else []
            n_r = s["n_rand"] + sum(r["n_rand"] for r in recent)
            e_r = s["n_rand_err"] + sum(r["n_rand_err"] for r in recent)
            s["est_err"] = e_r / n_r if n_r > 0 else np.nan

            t = time.perf_counter() - t0

            print(
                f"  [{strategy} #{repeat}] n={len(labeled):6d}  "
                f"acc={m['accuracy']:.4f}  err={m['n_err']:6d}  "
                f"disagree={s['disagree_rate']:.4f}  "
                f"change={s['change_rate']:.4f}  "
                f"est_err={s['est_err']:.4f}  "
                f"sat率={y_lab.mean():.3f}  ({t:.1f}s)",
                flush=True
            )

            rows.append({
                "strategy": strategy,
                "repeat": repeat,
                "round": rnd,
                "n_labeled": len(labeled),
                "sat_ratio_labeled": y_lab.mean(),
                **m,
                **s,
            })

        if new is None:
            break

        labeled = np.r_[labeled, new]
        is_labeled[new] = True
        rnd += 1

    return pd.DataFrame(rows), labeled


def summarize_stop_rules(curve):
    """
    各停止ルール (指標がしきい値を STOP_PATIENCE ラウンド続けて下回ったら止める) で
    止めた場合の点数とテスト精度。止まらなかったルールは NaN。
    """

    out = []

    for (strategy, repeat), g in curve.groupby(["strategy", "repeat"]):
        g = g.sort_values("round").reset_index(drop=True)

        for col, thresholds in STOP_RULES.items():
            for thr in thresholds:
                hit = None
                streak = 0
                for i, ok in enumerate((g[col] < thr).to_numpy()):
                    streak = streak + 1 if ok else 0
                    if streak >= STOP_PATIENCE:
                        hit = i
                        break

                row = {
                    "strategy": strategy,
                    "repeat": repeat,
                    "rule": col,
                    "threshold": thr,
                    "n_labeled": np.nan,
                    "accuracy": np.nan,
                    "n_err": np.nan,
                }
                if hit is not None:
                    r = g.iloc[hit]
                    row.update(
                        n_labeled=int(r["n_labeled"]),
                        accuracy=r["accuracy"],
                        n_err=int(r["n_err"]),
                    )
                out.append(row)

    return pd.DataFrame(out)


# ============================================================
# 図
# ============================================================

def plot_learning_curve(curve, path):
    """
    学習点数 vs テスト精度 / 誤り数。繰り返した場合は平均と最小～最大の幅を描く
    """

    full = None
    if os.path.exists(FULL_METRICS_PATH):
        fm = pd.read_csv(FULL_METRICS_PATH)
        fm = fm[fm["model"] == "MLP"]
        if len(fm):
            full = fm.iloc[0]

    colors = {"random": "#7f7f7f", "active": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for metric, ax in zip(["accuracy", "n_err"], axes):
        for strategy, g in curve.groupby("strategy"):
            s = g.groupby("n_labeled")[metric].agg(["mean", "min", "max"])
            ax.plot(s.index, s["mean"], marker="o", ms=3,
                    color=colors.get(strategy), label=strategy)
            ax.fill_between(s.index, s["min"], s["max"],
                            color=colors.get(strategy), alpha=0.2)

        ax.set_xscale("log")
        ax.set_xlabel("問い合わせた点数 (= シミュレーション回数)")
        ax.grid(alpha=0.3, which="both")

    if full is not None:
        axes[0].axhline(
            full["accuracy"], color="k", ls="--",
            label=f"全データ MLP ({int(full['n_train'])} 点)"
        )

    axes[0].set_ylabel("テスト accuracy")
    axes[0].set_title("テスト設計での正解率")
    axes[0].legend(loc="lower right")

    axes[1].set_yscale("log")
    axes[1].set_ylabel("テストの誤り数")
    axes[1].set_title("テスト設計での誤り数")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stop_metrics(curve, path):
    """
    学習点数 vs 停止判定用の指標。
    est_err の図には本当のテスト誤り率 (1 - accuracy) を破線で重ねる。
    """

    colors = {"random": "#7f7f7f", "active": "#d62728"}
    panels = [
        ("disagree_rate", "全会一致しない点の割合"),
        ("change_rate", "前ラウンドから予測が変わった点の割合"),
        ("est_err", "推定誤り率 (実線) / テスト誤り率 (破線)"),
    ]

    curve = curve.assign(test_err=1 - curve["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for (col, title), ax in zip(panels, axes):
        for strategy, g in curve.groupby("strategy"):
            s = g.groupby("n_labeled")[col].mean()
            ax.plot(s.index, s.values, marker="o", ms=3,
                    color=colors.get(strategy), label=strategy)

            if col == "est_err":
                t = g.groupby("n_labeled")["test_err"].mean()
                ax.plot(t.index, t.values, ls="--",
                        color=colors.get(strategy),
                        label=f"{strategy} (テスト)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("問い合わせた点数 (= シミュレーション回数)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_queried(train, labeled, path_fmt, n_maps=4):
    """
    問い合わせた点が多い設計について、正解の Vin-Vb マップに問い合わせ点を重ねる
    """

    q = train.iloc[labeled]
    counts = q.groupby(DESIGN_KEYS).size().sort_values(ascending=False)

    for design, cnt in counts.head(n_maps).items():
        d = dict(zip(DESIGN_KEYS, design))

        mask = np.ones(len(train), dtype=bool)
        qmask = np.ones(len(q), dtype=bool)
        for k in DESIGN_KEYS:
            mask &= np.isclose(train[k].to_numpy(), d[k])
            qmask &= np.isclose(q[k].to_numpy(), d[k])

        sub = train[mask]
        vin = np.sort(sub["Vin_V"].unique())
        vb = np.sort(sub["Vb_V"].unique())
        grid = np.full((len(vb), len(vin)), np.nan)
        grid[np.searchsorted(vb, sub["Vb_V"].to_numpy()),
             np.searchsorted(vin, sub["Vin_V"].to_numpy())] = sub[LABEL]

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.pcolormesh(vin, vb, grid, cmap="Blues", vmin=-0.3, vmax=1.3,
                      shading="nearest")
        ax.contour(vin, vb, grid, levels=[0.5], colors="k", linewidths=0.8)
        ax.scatter(q.loc[qmask, "Vin_V"], q.loc[qmask, "Vb_V"],
                   s=8, c="#d62728", label=f"問い合わせ点 ({cnt})")

        ax.set_xlabel("Vin [V]")
        ax.set_ylabel("Vb [V]")
        ax.set_title(
            f"Wn={d['Wn_um']} Wp={d['Wp_um']} "
            f"Wtail={d['Wtail_um']} L={d['L_um']} [um]",
            fontsize=10
        )
        ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(
            path_fmt.format(
                f"wn_{d['Wn_um']}_wp_{d['Wp_um']}"
                f"_wt_{d['Wtail_um']}_l_{d['L_um']}"
            ),
            dpi=130
        )
        plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    df = load_data()
    train, test = split_by_design(df)

    X_pool = train[FEATURES].to_numpy(dtype=np.float32)
    y_pool = train[LABEL].to_numpy()
    X_test = test[FEATURES].to_numpy(dtype=np.float32)
    y_test = test[LABEL].to_numpy()

    print(f"候補点 {len(train)} 行 / テスト {len(test)} 行")
    print(f"テストの飽和率: {y_test.mean():.3f}")
    print(f"初期 {N_INIT} 点 + {BATCH} 点ずつ追加、上限 {BUDGET} 点, "
          f"繰り返し {N_REPEATS} 回")

    curves = []
    last_active = None

    for repeat in range(N_REPEATS):
        for strategy in ["random", "active"]:
            print(f"\n=== {strategy} (繰り返し {repeat}) ===")
            curve, labeled = run(strategy, repeat, X_pool, y_pool,
                                 X_test, y_test)
            curves.append(curve)
            if strategy == "active":
                last_active = labeled

            # 途中で止めても結果が残るように毎回保存する
            pd.concat(curves).to_csv(
                os.path.join(OUT_DIR, "learning_curve.csv"),
                index=False
            )

    curve = pd.concat(curves)

    plot_learning_curve(
        curve,
        os.path.join(OUT_DIR, "learning_curve.png")
    )
    plot_stop_metrics(
        curve,
        os.path.join(OUT_DIR, "stop_metrics.png")
    )
    plot_queried(
        train, last_active,
        os.path.join(OUT_DIR, "queried_{}.png")
    )

    rules = summarize_stop_rules(curve)
    rules.to_csv(os.path.join(OUT_DIR, "stop_rules.csv"), index=False)

    # 最終ラウンドの比較
    final = curve[curve["n_labeled"] == curve["n_labeled"].max()]
    print("\n" + final.groupby("strategy")[
        ["accuracy", "f1", "pr_auc", "n_err"]
    ].mean().to_string())

    # 停止ルールごとの、止めた点数とそのときのテスト精度
    # (stopped = 上限までに止まった回数。点数と精度は止まった回の平均)
    print("\n停止ルール (しきい値を "
          f"{STOP_PATIENCE} ラウンド続けて下回ったら停止):")
    print(rules.groupby(["strategy", "rule", "threshold"]).agg(
        stopped=("n_labeled", "count"),
        n_labeled=("n_labeled", "mean"),
        accuracy=("accuracy", "mean"),
        n_err=("n_err", "mean"),
    ).to_string())
    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
