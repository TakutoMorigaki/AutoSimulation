import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classify_cascade import (
    STUDY_DIR,
    DESIGN_KEYS,
    FEATURES,
    LABEL,
    load_data,
    subsample,
    build_models,
    evaluate,
)


# ============================================================
# cascade 回路の飽和分類器を、設計単位の5分割交差検証で比較する
#
#   144 個の (W1, W2, L1, L2) の組を無作為に5グループに分け、
#   各グループを1回ずつテスト、残り4グループを学習に使う。
#   どの設計も必ず1回テストされる。
#
#   モデル側のシード (SVM の間引き, MLP の初期値) は固定し、
#   ばらつきの原因を「設計の分け方」だけにする。
# ============================================================


# ============================================================
# パス・設定
# ============================================================

OUT_DIR = os.path.join(
    STUDY_DIR,
    "cascade_classifier",
    "cv"
)

N_FOLDS = 5

# 設計の並べ替えに使うシード
CV_SEED = 0

# 図に出す指標
PLOT_METRICS = ["accuracy", "f1", "pr_auc", "roc_auc"]


# ============================================================
# 分割
# ============================================================

def assign_folds(df):
    """
    各設計に 0 ～ N_FOLDS-1 の fold 番号を振り、行ごとの fold 番号を返す
    """

    designs = df[DESIGN_KEYS].drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(CV_SEED)
    order = rng.permutation(len(designs))

    designs["fold"] = -1
    for f, idx in enumerate(np.array_split(order, N_FOLDS)):
        designs.loc[idx, "fold"] = f

    print(
        "fold ごとの設計数: "
        + ", ".join(
            str(n) for n in designs["fold"].value_counts().sort_index()
        )
    )

    return df.merge(designs, on=DESIGN_KEYS, how="left")["fold"].to_numpy()


# ============================================================
# 図
# ============================================================

def plot_cv(metrics, path):
    """
    指標ごとに、fold の値 (点) と平均 ± 標準偏差 (棒) を並べる
    """

    models = list(metrics["model"].unique())
    x = np.arange(len(models))

    fig, axes = plt.subplots(
        1, len(PLOT_METRICS),
        figsize=(3.6 * len(PLOT_METRICS), 3.8)
    )

    for ax, m in zip(axes, PLOT_METRICS):
        g = metrics.groupby("model")[m]
        mean = g.mean().reindex(models)
        std = g.std().reindex(models)

        ax.bar(x, mean, yerr=std, capsize=4, color="#9ecae1",
               edgecolor="#3182bd")

        for i, name in enumerate(models):
            v = metrics.loc[metrics["model"] == name, m]
            ax.scatter(np.full(len(v), i), v, color="k", s=12, zorder=3)

        lo = metrics[m].min()
        ax.set_ylim(max(0, lo - (1 - lo) * 0.3), 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20)
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"設計単位 {N_FOLDS} 分割交差検証 (棒: 平均 ± 標準偏差, 点: 各 fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    df = load_data()
    folds = assign_folds(df)

    X = df[FEATURES].to_numpy()
    y = df[LABEL].to_numpy()

    rows = []

    # 全行の予測確率 (各行は自分がテストになった fold の予測)
    oof = {}

    for f in range(N_FOLDS):

        is_test = folds == f
        train = df[~is_test]

        print(
            f"\n===== fold {f} : 学習 {(~is_test).sum()} 行 / "
            f"テスト {is_test.sum()} 行 (飽和率 {y[is_test].mean():.3f}) ====="
        )

        for name, (model, n_train) in build_models().items():

            tr = subsample(train, n_train)

            t0 = time.perf_counter()
            model.fit(tr[FEATURES].to_numpy(), tr[LABEL].to_numpy())
            t_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            prob = model.predict_proba(X[is_test])[:, 1]
            t_pred = time.perf_counter() - t0

            m = evaluate(y[is_test], prob)
            print(
                f"  {name:9s} "
                + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
                + f"  fit={t_fit:.1f}s  predict={t_pred:.1f}s",
                flush=True
            )

            rows.append({
                "fold": f,
                "model": name,
                "n_train": len(tr),
                **m,
                "fit_s": t_fit,
                "predict_s": t_pred,
            })

            oof.setdefault(name, np.full(len(df), np.nan))[is_test] = prob

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        os.path.join(OUT_DIR, "cv_metrics.csv"),
        index=False
    )

    # 平均と標準偏差
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "fit_s", "predict_s"]
    summary = metrics.groupby("model", sort=False)[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"))

    # 全 144 設計の誤り数 (各設計がテストになった fold での値)
    err = df[DESIGN_KEYS].copy()
    err["fold"] = folds
    for name, prob in oof.items():
        err[f"{name}_err"] = ((prob >= 0.5).astype(int) != y).astype(int)
    err_per_design = (
        err.groupby(DESIGN_KEYS + ["fold"], as_index=False).sum()
    )
    err_per_design.to_csv(
        os.path.join(OUT_DIR, "cv_errors_per_design.csv"),
        index=False
    )

    # fold ごとの順位
    rank = (
        metrics.pivot(index="fold", columns="model", values="f1")
        .rank(axis=1, ascending=False)
    )

    plot_cv(metrics, os.path.join(OUT_DIR, "cv_metrics.png"))

    print("\n===== 平均 ± 標準偏差 =====")
    for name, r in summary.iterrows():
        print(
            f"  {name:9s} "
            + "  ".join(
                f"{c}={r[f'{c}_mean']:.4f}±{r[f'{c}_std']:.4f}"
                for c in ["accuracy", "f1", "pr_auc"]
            )
        )

    print("\n===== F1 の fold ごとの順位 =====")
    print(rank.to_string())

    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
