import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classify_5tota import (
    STUDY_DIR,
    DESIGN_KEYS,
    FEATURES,
    LABEL,
    load_data,
    subsample,
    build_models,
    evaluate,
    plot_design_maps,
)


# ============================================================
# 5TOTA 回路の飽和分類器を、L を1つずつテストに回す検証で比較する
#
#   L (0.28 / 0.56 / 1.12 / 2.24 um) のうち1つを、その L の4設計
#   (W 4通り) まるごとテストにし、残り3つの L で学習する。
#
#   5TOTA の飽和境界はほぼ L だけで決まるので、設計単位の分割では
#   テストと同じ L が必ず学習に残り、汎化を測れない。
#   この検証では「学習にない L」でどれだけ当たるかを見る。
#
#     L = 0.56, 1.12 : 学習した L の間 (補間)
#     L = 0.28, 2.24 : 学習した L の外 (外挿)
# ============================================================


# ============================================================
# パス
# ============================================================

OUT_DIR = os.path.join(
    STUDY_DIR,
    "5tota_classifier",
    "cv_l"
)

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# ============================================================
# 図
# ============================================================

def plot_by_l(metrics, path):
    """
    テストにした L ごとに、各モデルの F1 と PR-AUC を並べる
    """

    models = list(metrics["model"].unique())
    ls = sorted(metrics["test_L_um"].unique())
    x = np.arange(len(ls))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, m in zip(axes, ["f1", "pr_auc"]):
        for i, name in enumerate(models):
            v = (
                metrics[metrics["model"] == name]
                .set_index("test_L_um")[m]
                .reindex(ls)
            )
            ax.bar(x + (i - (len(models) - 1) / 2) * width, v, width,
                   label=name)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"L={l}\n({'外挿' if l in (min(ls), max(ls)) else '補間'})"
             for l in ls]
        )
        lo = metrics[m].min()
        ax.set_ylim(max(0, lo - (1 - lo) * 0.2), 1.0)
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="lower left")
    fig.suptitle("5TOTA: L を1つずつテストにした検証 (横軸 = テストにした L)")
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
    l_values = sorted(df["L_um"].unique())

    rows = []
    err_rows = []

    for l in l_values:

        is_test = np.isclose(df["L_um"].to_numpy(), l)
        train = df[~is_test]
        test = df[is_test].reset_index(drop=True)

        kind = "外挿" if l in (l_values[0], l_values[-1]) else "補間"

        print(
            f"\n===== テスト L={l} ({kind}) : 学習 {len(train)} 行 / "
            f"テスト {len(test)} 行 (飽和率 {test[LABEL].mean():.3f}) ====="
        )

        X_test = test[FEATURES].to_numpy()
        y_test = test[LABEL].to_numpy()

        probs = {}

        for name, (model, n_train) in build_models().items():

            tr = subsample(train, n_train)

            t0 = time.perf_counter()
            model.fit(tr[FEATURES].to_numpy(), tr[LABEL].to_numpy())
            t_fit = time.perf_counter() - t0

            prob = model.predict_proba(X_test)[:, 1]

            m = evaluate(y_test, prob)
            print(
                f"  {name:9s} "
                + "  ".join(f"{k}={v:.4f}" for k, v in m.items())
                + f"  fit={t_fit:.1f}s",
                flush=True
            )

            rows.append({
                "test_L_um": l,
                "kind": kind,
                "model": name,
                "n_train": len(tr),
                **m,
                "fit_s": t_fit,
            })
            probs[name] = prob

        # 設計ごとの誤り数と比較図
        designs = test[DESIGN_KEYS].drop_duplicates()
        for _, d in designs.iterrows():
            mask = np.isclose(test["W_um"].to_numpy(), d["W_um"])
            row = {k: d[k] for k in DESIGN_KEYS}
            for name, prob in probs.items():
                row[f"{name}_err"] = int(
                    ((prob[mask] >= 0.5).astype(int) != y_test[mask]).sum()
                )
            err_rows.append(row)

            plot_design_maps(
                test, probs, d,
                os.path.join(
                    OUT_DIR,
                    f"map_w_{d['W_um']}_l_{d['L_um']}.png"
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        os.path.join(OUT_DIR, "cv_l_metrics.csv"),
        index=False
    )

    pd.DataFrame(err_rows).to_csv(
        os.path.join(OUT_DIR, "cv_l_errors_per_design.csv"),
        index=False
    )

    plot_by_l(metrics, os.path.join(OUT_DIR, "cv_l_metrics.png"))

    # 補間 / 外挿 / 全体の平均
    summary = pd.concat([
        metrics.groupby(["kind", "model"], sort=False)[METRIC_COLS].mean(),
        metrics.groupby("model", sort=False)[METRIC_COLS].mean()
        .assign(kind="全体").set_index("kind", append=True)
        .swaplevel(),
    ])
    summary.to_csv(os.path.join(OUT_DIR, "cv_l_summary.csv"))

    print("\n===== F1 (行 = テストにした L) =====")
    print(metrics.pivot(index="test_L_um", columns="model", values="f1")
          .round(4).to_string())

    print("\n===== 平均 =====")
    print(summary[["accuracy", "f1", "pr_auc"]].round(4).to_string())

    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
