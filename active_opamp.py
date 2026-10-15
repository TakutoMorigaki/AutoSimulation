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
# 2段オペアンプ (opamp.sch) の all_sat 分類で「能動学習」を検証する
# (active_5tota.py のオペアンプ版)
#
#   既存の全グリッドデータ opamp_results.csv を「シミュレータの代わり」に使う。
#   (学習用の設計の行 = まだラベルを知らない候補点。1行を問い合わせる = 1回シミュレーションする)
#
#   入力 : 7種類の W, L, Ibias, Vin (10次元。W, L, Ibias は log を取る)
#   出力 : all_sat (1 = M1～M11 がすべて飽和)
#
#   比較する2つのサンプリング方法:
#     random : 候補点から一様ランダムに追加していく (基準)
#     active : MLP を ENSEMBLE_SIZE 個学習し、平均確率のエントロピーが大きい
#              (= 境界付近で、モデル同士の意見も割れている) 点を優先して追加する
#              - 毎回 RANDOM_FRAC の割合はランダムに選ぶ (未知の飽和領域の見落とし対策)
#              - 上位候補を k-means で分け、各クラスタから1点ずつ選ぶ
#                (並列シミュレーションを想定した1回分の点が、1か所に固まらないようにする)
#
#   5TOTA より次元が多く、必要な点数も増えると考えられるので、
#   1ラウンドの追加点数を「それまでの点数の BATCH_GROWTH 倍」(最低 BATCH) に増やしていく。
#   (ラウンド数 = MLP の学習回数を抑えるため)
#
#   テスト設計の分け方は classify_opamp.py と同じ。
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
#   出力 (study/opamp_active/):
#     learning_curve.csv  : ラウンドごとの学習点数、テスト精度、停止判定用の指標 (1ラウンドごとに保存)
#     learning_curve.png  : 学習点数 vs テスト精度 (random / active)
#     stop_metrics.png    : 学習点数 vs 停止判定用の指標
#     stop_rules.csv      : 各停止ルール (指標としきい値) で止めた場合の点数とテスト精度
#     queried_points.csv  : 最後の active で問い合わせた点と、追加されたラウンド
#     queried_*.png       : active で問い合わせた点を Vin-Ibias 平面に重ねた図
#
#   環境変数で設定を変えられる:
#     AL_BUDGET     : 問い合わせる点数の上限
#     AL_REPEATS    : 乱数を変えて繰り返す回数
#     AL_STRATEGIES : 実行する方法 (例: "active" だけにする。既定は "random,active")
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
    "opamp_results.csv"
)

# 全データで学習した分類器の結果 (あれば図に基準線として描く)
FULL_METRICS_PATH = os.path.join(
    STUDY_DIR,
    "opamp_classifier",
    "metrics.csv"
)

# 結果の保存先
OUT_DIR = os.path.join(
    STUDY_DIR,
    "opamp_active"
)


# ============================================================
# 設定
# ============================================================

# 設計 (W, L の組) を区別する列
SIZE_COLS = ["Win", "Wload", "Wtail", "W6", "W7", "Wbp", "Wbn", "L"]
DESIGN_KEYS = [f"{c}_um" for c in SIZE_COLS]

# 入力にする列
FEATURES = [f"log_{c}" for c in SIZE_COLS] + ["log_Ibias", "Vin_V"]

LABEL = "all_sat"

# テストに回す設計の割合 (classify_opamp.py と同じ)
TEST_DESIGN_RATIO = 0.2

# 最初にランダムに取る点数
N_INIT = 1000

# 1ラウンドで追加する点数の最低値
BATCH = 1000

# 1ラウンドで追加する点数 = max(BATCH, それまでの点数 × BATCH_GROWTH)
BATCH_GROWTH = 0.1

# 問い合わせる点数の上限 (学習用候補 約487万点のうち)
BUDGET = int(os.environ.get("AL_BUDGET", 60000))

# 乱数を変えて繰り返す回数 (結果のばらつきを見る)
N_REPEATS = int(os.environ.get("AL_REPEATS", 1))

# 実行する方法
STRATEGIES = os.environ.get("AL_STRATEGIES", "random,active").split(",")

# active で使う MLP の個数
ENSEMBLE_SIZE = 5

# 1ラウンドの追加点のうち、ランダムに選ぶ割合
RANDOM_FRAC = 0.2

# 候補点全部を毎回スコアするのは重いので、ラウンドごとにこの数だけ抜き出してスコアする
N_CANDIDATES = 500_000

# k-means にかける上位候補の数 (= 境界寄りの点の数) は 追加点数 × この倍率
TOP_FACTOR = 10

# 予測を何行ずつに分けるか
PREDICT_CHUNK = 500_000

# 停止判定用に固定する候補点の数
N_STOP = 100_000

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
    CSV から必要な列だけを読み、W, L, Ibias を log に変換した列を足す
    """

    df = pd.read_csv(
        CSV_PATH,
        usecols=DESIGN_KEYS + ["Ibias_uA", "Vin_V", LABEL],
        dtype={
            **{k: np.float32 for k in DESIGN_KEYS + ["Ibias_uA", "Vin_V"]},
            LABEL: np.int8,
        },
    )

    for col in SIZE_COLS:
        df[f"log_{col}"] = np.log(df[f"{col}_um"])

    df["log_Ibias"] = np.log(df["Ibias_uA"])

    return df


def split_by_design(df):
    """
    設計 (W, L の組) 単位で学習用とテスト用に分ける
    (classify_opamp.py と同じ分け方。行数が多いので判定は merge で行う)
    """

    designs = df[DESIGN_KEYS].drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    n_test = int(round(len(designs) * TEST_DESIGN_RATIO))
    test_idx = rng.choice(len(designs), size=n_test, replace=False)

    test_designs = designs.iloc[test_idx]

    is_test = (
        df[DESIGN_KEYS]
        .merge(test_designs.assign(_test=True), on=DESIGN_KEYS, how="left")
        ["_test"].notna().to_numpy()
    )

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
    classify_opamp.py の MLP と同じ構成。
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
    各モデルの確率の平均を PREDICT_CHUNK 行ずつ計算してつなげる
    """

    return np.concatenate([
        np.mean(
            [m.predict_proba(X[i:i + PREDICT_CHUNK])[:, 1] for m in models],
            axis=0
        )
        for i in range(0, len(X), PREDICT_CHUNK)
    ])


def predict_members(models, X):
    """
    各モデルの確率 (n_models, n_samples)
    """

    return np.stack([
        np.concatenate([
            m.predict_proba(X[i:i + PREDICT_CHUNK])[:, 1]
            for i in range(0, len(X), PREDICT_CHUNK)
        ])
        for m in models
    ])


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def stop_metrics(models, X_stop, prev_pred):
    """
    停止判定用の固定点での disagree_rate と change_rate。
    次のラウンドで使うため、今回のアンサンブルの予測も返す。
    """

    probs = predict_members(models, X_stop)
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

    score = entropy(predict_ensemble(models, X_pool[cand]))

    # 境界寄りの上位を k-means で n_act 個に分け、各クラスタで最もスコアが高い点を取る
    # (クラスタ数が多いので、初期値は k-means++ ではなくランダムにして時間を抑える)
    n_top = min(len(cand), n_act * TOP_FACTOR)
    top = np.argpartition(-score, n_top - 1)[:n_top]

    X_top = StandardScaler().fit_transform(X_pool[cand[top]])
    km = MiniBatchKMeans(
        n_clusters=n_act,
        init="random",
        batch_size=min(n_top, max(4096, 3 * n_act)),
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

def run(strategy, repeat, X_pool, y_pool, X_test, y_test, on_round):
    """
    strategy の方法で BUDGET 点まで問い合わせ、ラウンドごとのテスト精度を返す。
    ラウンドごとに on_round(row) を呼ぶ (途中経過の保存用)。
    """

    seed = SEED + 1000 * repeat
    rng = np.random.default_rng(seed)

    # 停止判定用の固定点 (追加点の選び方の乱数とは分けておく)
    stop_set = np.random.default_rng(seed + 1).choice(
        len(X_pool), size=min(N_STOP, len(X_pool)), replace=False
    )
    X_stop = X_pool[stop_set]

    labeled = select_random(rng, np.arange(len(X_pool)), N_INIT)
    added_round = np.zeros(len(labeled), dtype=int)
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
        t_fit = time.perf_counter() - t0

        # 次に追加する点 (rand はそのうちランダムに選んだ点)
        new = rand = None
        if len(labeled) < BUDGET:
            n_add = min(
                max(BATCH, int(len(labeled) * BATCH_GROWTH)),
                BUDGET - len(labeled)
            )
            unlabeled = np.flatnonzero(~is_labeled)

            if strategy == "active" and models is not None:
                picked, rand = select_active(rng, models, X_pool,
                                             unlabeled, n_add)
                new = np.r_[picked, rand]
            else:
                new = rand = select_random(rng, unlabeled, n_add)

        if models is not None:
            m = evaluate(y_test, predict_ensemble(models, X_test))

            s, prev_pred = stop_metrics(models, X_stop, prev_pred)

            # ランダムに選んだ追加点について、シミュレーション前の予測が外れていた数
            s["n_rand"] = 0
            s["n_rand_err"] = 0
            if rand is not None:
                pred_rand = predict_ensemble(models, X_pool[rand])
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
                f"acc={m['accuracy']:.4f}  err={m['n_err']:7d}  "
                f"disagree={s['disagree_rate']:.4f}  "
                f"change={s['change_rate']:.4f}  "
                f"est_err={s['est_err']:.4f}  "
                f"sat率={y_lab.mean():.3f}  "
                f"(学習 {t_fit:.0f}s / 計 {t:.0f}s)",
                flush=True
            )

            row = {
                "strategy": strategy,
                "repeat": repeat,
                "round": rnd,
                "n_labeled": len(labeled),
                "sat_ratio_labeled": y_lab.mean(),
                **m,
                **s,
                "fit_s": t_fit,
            }
            rows.append(row)
            on_round(row)

        if new is None:
            break

        rnd += 1
        labeled = np.r_[labeled, new]
        added_round = np.r_[added_round, np.full(len(new), rnd)]
        is_labeled[new] = True

    return pd.DataFrame(rows), labeled, added_round


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
    問い合わせた点が多い設計について、正解の Vin-Ibias マップに問い合わせ点を重ねる
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
        ib = np.sort(sub["Ibias_uA"].unique())
        grid = np.full((len(ib), len(vin)), np.nan)
        grid[np.searchsorted(ib, sub["Ibias_uA"].to_numpy()),
             np.searchsorted(vin, sub["Vin_V"].to_numpy())] = sub[LABEL]

        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        ax.pcolormesh(vin, ib, grid, cmap="Blues", vmin=-0.3, vmax=1.3,
                      shading="nearest")
        ax.contour(vin, ib, grid, levels=[0.5], colors="k", linewidths=0.8)
        ax.scatter(q.loc[qmask, "Vin_V"], q.loc[qmask, "Ibias_uA"],
                   s=8, c="#d62728", label=f"問い合わせ点 ({cnt})")

        ax.set_xlabel("Vin [V]")
        ax.set_ylabel("Ibias [uA]")
        ax.set_title(
            f"Win={d['Win_um']:g} Wload={d['Wload_um']:g} "
            f"Wtail={d['Wtail_um']:g} W6={d['W6_um']:g}\n"
            f"W7={d['W7_um']:g} Wbp={d['Wbp_um']:g} "
            f"Wbn={d['Wbn_um']:g} L={d['L_um']:g} [um]",
            fontsize=9
        )
        ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(
            path_fmt.format(
                "_".join(f"{c.lower()}_{d[f'{c}_um']:g}" for c in SIZE_COLS)
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

    t0 = time.perf_counter()
    df = load_data()
    train, test = split_by_design(df)
    del df
    print(f"読み込み {time.perf_counter() - t0:.0f}s")

    X_pool = train[FEATURES].to_numpy(dtype=np.float32)
    y_pool = train[LABEL].to_numpy()
    X_test = test[FEATURES].to_numpy(dtype=np.float32)
    y_test = test[LABEL].to_numpy()

    print(f"候補点 {len(train)} 行 / テスト {len(test)} 行")
    print(f"テストの飽和率: {y_test.mean():.3f}")
    print(f"初期 {N_INIT} 点 + max({BATCH}, 点数×{BATCH_GROWTH}) 点ずつ追加、"
          f"上限 {BUDGET} 点, 方法 {STRATEGIES}, 繰り返し {N_REPEATS} 回")

    curve_path = os.path.join(OUT_DIR, "learning_curve.csv")
    all_rows = []

    # 途中で止めても結果が残るように、1ラウンドごとに保存する
    def on_round(row):
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(curve_path, index=False)

    last_active = None

    for repeat in range(N_REPEATS):
        for strategy in STRATEGIES:
            print(f"\n=== {strategy} (繰り返し {repeat}) ===")
            _, labeled, added_round = run(
                strategy, repeat, X_pool, y_pool, X_test, y_test, on_round
            )
            if strategy == "active":
                last_active = labeled
                q = train.iloc[labeled][
                    DESIGN_KEYS + ["Ibias_uA", "Vin_V", LABEL]
                ].copy()
                q["round"] = added_round
                q.to_csv(os.path.join(OUT_DIR, "queried_points.csv"),
                         index=False)

    curve = pd.DataFrame(all_rows)

    plot_learning_curve(
        curve,
        os.path.join(OUT_DIR, "learning_curve.png")
    )
    plot_stop_metrics(
        curve,
        os.path.join(OUT_DIR, "stop_metrics.png")
    )
    if last_active is not None:
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
