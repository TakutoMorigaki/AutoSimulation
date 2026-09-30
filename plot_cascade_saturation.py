import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "cascade_results.csv"
)

# 図の保存先
DETAIL_DIR = os.path.join(
    STUDY_DIR,
    "cascade_saturation"
)

# W1 × W2 の全組み合わせを並べた飽和マップ
OVERVIEW_PATH = os.path.join(
    DETAIL_DIR,
    "cascade_saturation_map_all.png"
)

# W の組ごとに描き分けるための列
W_KEYS = ["W1_um", "W2_um"]


# ============================================================
# 補助関数
# ============================================================

def w_label(w1, w2):
    return f"W1 = {w1:g} um, W2 = {w2:g} um"


def add_gain(df):
    """
    W の組と Vb ごとに小信号ゲイン dVout/dVin を計算し、gain 列として追加する
    """

    df = df.sort_values(W_KEYS + ["Vb_V", "Vin_V"]).copy()

    df["gain"] = 0.0

    for _, s in df.groupby(W_KEYS + ["Vb_V"]):

        if len(s) > 1:
            df.loc[s.index, "gain"] = np.gradient(
                s["Vout_V"].to_numpy(),
                s["Vin_V"].to_numpy()
            )

    return df


def find_max_gain(df):
    """
    全トランジスタが飽和している点のうち、|gain| が最大の行を返す
    (飽和点がなければ None)
    """

    sat = df[df["all_sat"] == 1]

    if sat.empty:
        return None

    return sat.loc[sat["gain"].abs().idxmax()]


def mark_max_gain(ax, best):

    ax.plot(
        best["Vin_V"],
        best["Vb_V"],
        marker="*",
        markersize=14,
        color="tab:red",
        markeredgecolor="white",
        linestyle="none",
        label=(
            f"Max |gain| {abs(best['gain']):.3g} V/V\n"
            f"(Vin = {best['Vin_V']:.2f} V, Vb = {best['Vb_V']:.2f} V)"
        )
    )


# ============================================================
# Vb × Vin の飽和マップ
# ============================================================

def draw_saturation_map(ax, sub):

    grid = sub.pivot_table(
        index="Vb_V",
        columns="Vin_V",
        values="all_sat",
        aggfunc="max"
    )

    ax.pcolormesh(
        grid.columns.to_numpy(),
        grid.index.to_numpy(),
        grid.to_numpy(),
        cmap="Greens",
        vmin=0,
        vmax=1.5,
        shading="nearest"
    )


def plot_summary(sub, w1, w2):
    """
    W の組1つ分の飽和マップ
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    draw_saturation_map(ax, sub)

    # 飽和領域全体で |gain| が最大となる点
    best = find_max_gain(sub)

    if best is not None:
        mark_max_gain(ax, best)
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("Vin [V]")
    ax.set_ylabel("Vb [V]")
    ax.set_title(f"All saturated ({w_label(w1, w2)})")

    fig.tight_layout()

    path = os.path.join(
        DETAIL_DIR,
        f"cascade_saturation_map_w1_{w1:g}_w2_{w2:g}.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saturation map saved: {path}")


def plot_overview(df):
    """
    W1 (行) × W2 (列) の全組み合わせの飽和マップを1枚に並べる
    """

    w1_list = sorted(df["W1_um"].unique())
    w2_list = sorted(df["W2_um"].unique())

    fig, axes = plt.subplots(
        len(w1_list),
        len(w2_list),
        figsize=(3.2 * len(w2_list), 2.8 * len(w1_list)),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for i, w1 in enumerate(w1_list):

        for j, w2 in enumerate(w2_list):

            ax = axes[i, j]
            sub = df[(df["W1_um"] == w1) & (df["W2_um"] == w2)]

            if sub.empty:
                ax.set_visible(False)
                continue

            draw_saturation_map(ax, sub)

            best = find_max_gain(sub)

            if best is not None:
                mark_max_gain(ax, best)
                gain_text = f"max |gain| {abs(best['gain']):.3g}"
            else:
                gain_text = "no saturated point"

            ax.set_title(
                f"W1 = {w1:g}, W2 = {w2:g}\n{gain_text}",
                fontsize=9
            )

    for ax in axes[-1]:
        ax.set_xlabel("Vin [V]")

    for ax in axes[:, 0]:
        ax.set_ylabel("Vb [V]")

    fig.suptitle("Cascade all-saturated region (rows: W1, columns: W2 [um])")
    fig.tight_layout()
    fig.savefig(OVERVIEW_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Overview map saved: {OVERVIEW_PATH}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    if "all_sat" not in df.columns:
        raise SystemExit(
            "CSVに飽和判定の列がありません。"
            "最新の cascade.py でシミュレーションし直してください。"
        )

    if not all(key in df.columns for key in W_KEYS):
        raise SystemExit(
            "CSVに W1_um, W2_um の列がありません。"
            "最新の cascade.py でシミュレーションし直してください。"
        )

    # 浮動小数点の誤差を丸める
    for col in W_KEYS + ["Vb_V", "Vin_V"]:
        df[col] = df[col].round(3)

    df = add_gain(df)

    print(f"Conditions: {len(df)}, all saturated: {int(df['all_sat'].sum())}")

    os.makedirs(DETAIL_DIR, exist_ok=True)

    # W の組ごとの飽和マップ
    for (w1, w2), w_sub in df.groupby(W_KEYS):
        plot_summary(w_sub, w1, w2)

    # 全組み合わせを並べた飽和マップ
    plot_overview(df)

    # W の組ごとに、利得が最大となる点を表示
    print()
    print("Max |gain| (all saturated):")

    for (w1, w2), w_sub in df.groupby(W_KEYS):

        best = find_max_gain(w_sub)

        if best is not None:
            print(
                f"  {w_label(w1, w2)}: {abs(best['gain']):.4g} V/V "
                f"at Vb = {best['Vb_V']:.2f} V, Vin = {best['Vin_V']:.2f} V"
            )
        else:
            print(f"  {w_label(w1, w2)}: no all-saturated point")
