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
# 5TOTA 回路で「全トランジスタ (M1～M5) が飽和するか (all_sat)」を
# 設計パラメータだけから予測する二値分類器を4種類比較する
#
#   LightGBM / MLP / ロジスティック回帰 / SVM (RBF)
#
#   入力 : W, L, Vb, Vin   (W, L は log を取る)
#          (M3～M5 の W は差動対の W から決まるので、設計の自由度は W, L だけ)
#   出力 : all_sat (1 = M1, M2 とも飽和)
#
#   gain_dc_dB, vds, vdsat などシミュレーションの出力は入力に入れない。
#   評価は「学習に使っていない (W, L) の組」で行う。
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
    "5tota_saturation_results.csv"
)

# 結果の保存先
OUT_DIR = os.path.join(
    STUDY_DIR,
    "5tota_classifier"
)

MODEL_DIR = os.path.join(
    OUT_DIR,
    "models"
)


# ============================================================
# 設定
# ============================================================

# 設計 (W, L の組) を区別する列
DESIGN_KEYS = ["W_um", "L_um"]

# 入力にする列
FEATURES = ["log_W", "log_L", "Vb_V", "Vin_V"]

LABEL = "all_sat"

# テストに回す設計の割合
TEST_DESIGN_RATIO = 0.2

# SVM は計算量が O(n^2) 以上なので学習データを間引く
SVM_TRAIN_SAMPLES = 30000

# MLP の学習データ数 (None なら全部)
MLP_TRAIN_SAMPLES = None

SEED = 0


# ============================================================
# データ
# ============================================================

def load_data():
    """
    CSV を読み、W, L を log に変換した列を足す
    """

    df = pd.read_csv(CSV_PATH)

    for col in ["W", "L"]:
        df[f"log_{col}"] = np.log(df[f"{col}_um"])

    return df


def split_by_design(df):
    """
    (W, L) の組単位で学習用とテスト用に分ける
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
            None,
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
    1つの設計について、正解と各モデルの予測を Vin-Vb 平面に並べる
    (TP / TN / FP / FN で色分け)
    """

    mask = np.ones(len(test), dtype=bool)
    for k in DESIGN_KEYS:
        mask &= np.isclose(test[k].to_numpy(), design[k])

    sub = test[mask]
    y = sub[LABEL].to_numpy()

    vin = np.sort(sub["Vin_V"].unique())
    vb = np.sort(sub["Vb_V"].unique())
    xi = np.searchsorted(vin, sub["Vin_V"].to_numpy())
    yi = np.searchsorted(vb, sub["Vb_V"].to_numpy())

    def to_grid(values):
        g = np.full((len(vb), len(vin)), np.nan)
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
    axes[0].pcolormesh(vin, vb, truth, cmap=cmap, vmin=-0.5, vmax=3.5,
                       shading="nearest")
    axes[0].set_title("正解 (all_sat)")

    for ax, name in zip(axes[1:], names):
        pred = (probs[name][mask] >= 0.5).astype(int)
        cls = np.where(
            pred == y, y,
            np.where(pred == 1, 2, 3)
        )
        n_err = int((pred != y).sum())

        ax.pcolormesh(vin, vb, to_grid(cls), cmap=cmap, vmin=-0.5,
                      vmax=3.5, shading="nearest")
        ax.contour(vin, vb, truth, levels=[0.5], colors="k",
                   linewidths=0.8)
        ax.set_title(f"{name} (誤り {n_err})")

    for ax in axes:
        ax.set_xlabel("Vin [V]")
    axes[0].set_ylabel("Vb [V]")

    handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=lab)
        for i, lab in enumerate(["非飽和", "飽和", "FP (誤って飽和)",
                                 "FN (見逃し)"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"W={design['W_um']} L={design['L_um']} [um]  (テスト設計)"
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
        prob = model.predict_proba(X_test)[:, 1]
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
    per_design = []
    for _, d in test_designs.iterrows():
        mask = np.ones(len(test), dtype=bool)
        for k in DESIGN_KEYS:
            mask &= np.isclose(test[k].to_numpy(), d[k])
        row = {k: d[k] for k in DESIGN_KEYS}
        for name, prob in probs.items():
            row[f"{name}_err"] = int(
                ((prob[mask] >= 0.5).astype(int) != y_test[mask]).sum()
            )
        per_design.append(row)

    pd.DataFrame(per_design).to_csv(
        os.path.join(OUT_DIR, "errors_per_design.csv"),
        index=False
    )

    plot_pr_curves(
        y_test, probs,
        os.path.join(OUT_DIR, "pr_curves.png")
    )

    for _, d in test_designs.iterrows():
        fname = (
            f"map_w_{d['W_um']}_l_{d['L_um']}.png"
        )
        plot_design_maps(test, probs, d, os.path.join(OUT_DIR, fname))

    print("\n" + metrics.to_string(index=False))
    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
