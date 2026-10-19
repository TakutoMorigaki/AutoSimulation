import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


# ============================================================
# 2段オペアンプ (opamp.sch) で「全トランジスタ (M1～M11) が飽和するか (all_sat)」を
# 設計パラメータだけから予測する二値分類器を4種類比較する
#
#   LightGBM / MLP / ロジスティック回帰 / SVM (RBF)
#
#   入力 : 7種類の W, L, Ibias, Vin   (W, L, Ibias は log を取る)
#          Win: PMOS 差動対 M1, M2 / Wload: NMOS 負荷ミラー M3, M4 / Wtail: PMOS テール M5
#          W6: 2段目 PMOS 電流源 M6 / W7: 2段目 NMOS ソース接地 M7
#          Wbp: バイアス PMOS M8, M9 / Wbn: バイアス NMOS M10, M11
#          (OPAMP.py で作った opamp_results.csv を使う。M12 は all_sat に含まれない)
#   出力 : all_sat (1 = M1～M11 がすべて飽和)
#
#   gain_dc_dB, vds, vdsat などシミュレーションの出力は入力に入れない。
#   評価は「学習に使っていない W, L の組」で行う。
#
#   データが約3000万行あるので、
#     - CSV から必要な列だけを float32 で読む
#     - MLP, ロジスティック回帰, SVM は学習データを間引く (LightGBM は全部使う)
#     - 予測は PREDICT_CHUNK 行ずつに分ける
#     - 比較図は誤りの多い MAX_MAPS 個の設計だけ描く
# ============================================================


# ============================================================
# パス
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..")
)

STUDY_DIR = os.path.join(
    PROJECT_ROOT,
    "study"
)

CSV_PATH = os.path.join(
    STUDY_DIR,
    "opamp_results.csv"
)

# 結果の保存先
OUT_DIR = os.path.join(
    STUDY_DIR,
    "opamp_classifier"
)

MODEL_DIR = os.path.join(
    OUT_DIR,
    "models"
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

# テストに回す設計の割合
TEST_DESIGN_RATIO = 0.2

# SVM は計算量が O(n^2) 以上なので学習データを間引く
SVM_TRAIN_SAMPLES = 30000

# MLP の学習データ数 (None なら全部)
MLP_TRAIN_SAMPLES = 3_000_000

# ロジスティック回帰の学習データ数
# (3次の多項式特徴量が 286 個になり、メモリを多く使うので間引く)
LOGREG_TRAIN_SAMPLES = 1_000_000

# 予測を何行ずつに分けるか (多項式特徴量などでメモリを使いすぎないため)
PREDICT_CHUNK = 500_000

# 比較図を描く設計の数 (誤りの多い順)
MAX_MAPS = 30

SEED = 0


# ============================================================
# データ
# ============================================================

def load_data():
    """
    CSV を読み、W, L を log に変換した列を足す
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
    (Win, Wload, Wtail, L) の組単位で学習用とテスト用に分ける
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

    return df[~is_test], df[is_test], test_designs


def subsample(df, n):
    """
    ラベルの比率を保ったまま n 行に間引く
    """

    if n is None or len(df) <= n:
        return df

    parts = [
        g.sample(n=int(round(n * len(g) / len(df))), random_state=SEED)
        for _, g in df.groupby(LABEL)
    ]

    return pd.concat(parts)


# ============================================================
# モデル
# ============================================================

def build_models():
    """
    比較するモデルと、それぞれの学習データ数
    """

    return {
        "LightGBM": (
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                random_state=SEED,
                verbose=-1,
            ),
            None,
        ),
        "MLP": (
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 64, 32),
                    batch_size=1024,
                    learning_rate_init=1e-3,
                    max_iter=100,
                    early_stopping=True,
                    n_iter_no_change=8,
                    random_state=SEED,
                ),
            ),
            MLP_TRAIN_SAMPLES,
        ),
        # 境界は非線形なので、3次の多項式特徴量を足して線形分離しやすくする
        "LogReg": (
            make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=3),
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=10.0),
            ),
            LOGREG_TRAIN_SAMPLES,
        ),
        "SVM": (
            make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=SEED,
                ),
            ),
            SVM_TRAIN_SAMPLES,
        ),
    }


def predict_proba(model, X):
    """
    飽和する確率を PREDICT_CHUNK 行ずつ予測してつなげる
    """

    return np.concatenate([
        model.predict_proba(X[i:i + PREDICT_CHUNK])[:, 1]
        for i in range(0, len(X), PREDICT_CHUNK)
    ])


def errors_per_design(df, y_true, probs):
    """
    設計ごとの各モデルの誤り数 (df と probs の行は対応している)
    """

    err = df[DESIGN_KEYS].copy()
    for name, prob in probs.items():
        err[f"{name}_err"] = ((prob >= 0.5).astype(int) != y_true).astype(int)

    return err.groupby(DESIGN_KEYS, as_index=False).sum()


def evaluate(y_true, prob):
    """
    しきい値 0.5 での指標と、しきい値によらない指標
    """

    pred = (prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
    }


# ============================================================
# 図
# ============================================================

def plot_pr_curves(y_true, probs, path):
    """
    全モデルの PR 曲線を1枚に重ねる
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    for name, prob in probs.items():
        p, r, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(r, p, label=f"{name} (AP={ap:.4f})")

    ax.axhline(y_true.mean(), color="gray", ls=":", label="ランダム")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR 曲線 (テスト設計)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_design_maps(test, probs, design, path):
    """
    1つの設計について、正解と各モデルの予測を Vin-Ibias 平面に並べる
    (TP / TN / FP / FN で色分け)
    """

    mask = np.ones(len(test), dtype=bool)
    for k in DESIGN_KEYS:
        mask &= np.isclose(test[k].to_numpy(), design[k])

    sub = test[mask]
    y = sub[LABEL].to_numpy()

    vin = np.sort(sub["Vin_V"].unique())
    ib = np.sort(sub["Ibias_uA"].unique())
    xi = np.searchsorted(vin, sub["Vin_V"].to_numpy())
    yi = np.searchsorted(ib, sub["Ibias_uA"].to_numpy())

    def to_grid(values):
        g = np.full((len(ib), len(vin)), np.nan)
        g[yi, xi] = values
        return g

    # 0: TN, 1: TP, 2: FP, 3: FN
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#dddddd", "#4c9be8", "#f28e2b", "#d62728"]
    )

    names = list(probs.keys())
    fig, axes = plt.subplots(
        1, len(names) + 1,
        figsize=(3.2 * (len(names) + 1), 3.6),
        sharey=True
    )

    truth = to_grid(y)
    axes[0].pcolormesh(vin, ib, truth, cmap=cmap, vmin=-0.5, vmax=3.5,
                       shading="nearest")
    axes[0].set_title("正解 (all_sat)")

    for ax, name in zip(axes[1:], names):
        pred = (probs[name][mask] >= 0.5).astype(int)
        cls = np.where(
            pred == y, y,
            np.where(pred == 1, 2, 3)
        )
        n_err = int((pred != y).sum())

        ax.pcolormesh(vin, ib, to_grid(cls), cmap=cmap, vmin=-0.5,
                      vmax=3.5, shading="nearest")
        ax.contour(vin, ib, truth, levels=[0.5], colors="k",
                   linewidths=0.8)
        ax.set_title(f"{name} (誤り {n_err})")

    for ax in axes:
        ax.set_xlabel("Vin [V]")
    axes[0].set_ylabel("Ibias [uA]")

    handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=lab)
        for i, lab in enumerate(["非飽和", "飽和", "FP (誤って飽和)",
                                 "FN (見逃し)"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        " ".join(f"{c}={design[f'{c}_um']:g}" for c in SIZE_COLS)
        + " [um]  (テスト設計)"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    df = load_data()
    train, test, test_designs = split_by_design(df)

    print(f"学習 {len(train)} 行 / テスト {len(test)} 行")
    print(f"テストの飽和率: {test[LABEL].mean():.3f}")

    X_test = test[FEATURES].to_numpy()
    y_test = test[LABEL].to_numpy()

    rows = []
    probs = {}

    for name, (model, n_train) in build_models().items():

        tr = subsample(train, n_train)

        print(f"\n[{name}] 学習 {len(tr)} 行 ...", flush=True)

        t0 = time.perf_counter()
        model.fit(tr[FEATURES].to_numpy(), tr[LABEL].to_numpy())
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        prob = predict_proba(model, X_test)
        t_pred = time.perf_counter() - t0

        m = evaluate(y_test, prob)
        print(
            "  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
            + f"  fit={t_fit:.1f}s  predict={t_pred:.1f}s"
        )

        rows.append({
            "model": name,
            "n_train": len(tr),
            **m,
            "fit_s": t_fit,
            "predict_s": t_pred,
        })
        probs[name] = prob

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        os.path.join(OUT_DIR, "metrics.csv"),
        index=False
    )

    # テスト設計ごとの誤り数
    per_design = errors_per_design(test, y_test, probs)
    per_design.to_csv(
        os.path.join(OUT_DIR, "errors_per_design.csv"),
        index=False
    )

    plot_pr_curves(
        y_test, probs,
        os.path.join(OUT_DIR, "pr_curves.png")
    )

    # 全モデルの誤りの合計が多い設計から MAX_MAPS 個だけ描く
    err_cols = [c for c in per_design.columns if c.endswith("_err")]
    worst = (
        per_design.assign(total=per_design[err_cols].sum(axis=1))
        .sort_values("total", ascending=False)
        .head(MAX_MAPS)
    )

    for rank, (_, d) in enumerate(worst.iterrows(), 1):
        fname = f"map_{rank:02d}_" + "_".join(
            f"{c.lower()}_{d[f'{c}_um']:g}" for c in SIZE_COLS
        ) + ".png"
        plot_design_maps(test, probs, d, os.path.join(OUT_DIR, fname))

    print("\n" + metrics.to_string(index=False))
    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
