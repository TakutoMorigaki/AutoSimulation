import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

# 図の保存先
DETAIL_DIR = os.path.join(
    STUDY_DIR,
    "5tota_saturation"
)

MAP_PATH = os.path.join(
    DETAIL_DIR,
    "5tota_saturation_map.png"
)

RANGE_PATH = os.path.join(
    DETAIL_DIR,
    "5tota_saturation_vin_range.png"
)

# 飽和していない領域の色
UNSAT_COLOR = "#e6e6e6"


# ============================================================
# 補助関数
# ============================================================

def get_transistors(df):
    """
    M1_sat, M2_sat ... の列からトランジスタ名を取得する
    """

    names = []

    for col in df.columns:

        match = re.fullmatch(r'(M\d+)_sat', col)

        if match:
            names.append(match.group(1))

    return sorted(names, key=lambda n: int(n[1:]))


def to_grid(sub, value):
    """
    Vb (縦) × Vin (横) の2次元データに変換する
    """

    return sub.pivot_table(
        index="Vb_V",
        columns="Vin_V",
        values=value,
        aggfunc="first",
        dropna=False    # 全て非飽和 (NaN) の列も残して軸をそろえる
    )


def find_max_gain(df):
    """
    全トランジスタが飽和している点のうち、DC Gain が最大の行を返す
    (飽和点がなければ None)
    """

    sat = df[df["all_sat"] == 1]

    if sat.empty:
        return None

    return sat.loc[sat["gain_dc_dB"].idxmax()]


def draw_grid(ax, grid, cmap, vmin, vmax):

    return ax.pcolormesh(
        grid.columns.to_numpy(),
        grid.index.to_numpy(),
        grid.to_numpy(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest"
    )


# ============================================================
# 1. 飽和マップ (飽和領域を DC Gain で色付け)
# ============================================================

def plot_saturation_map(df):

    fig, ax = plt.subplots(figsize=(6, 4.5))

    sat = df[df["all_sat"] == 1]
    vmin = sat["gain_dc_dB"].min()
    vmax = sat["gain_dc_dB"].max()

    # 飽和していない点は NaN にして灰色で表示
    # (Bluesの白に近い端は背景と区別しにくいので使わない)
    cmap = ListedColormap(
        plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256))
    )
    cmap.set_bad(UNSAT_COLOR)

    sub = df.copy()
    sub["gain_sat"] = sub["gain_dc_dB"].where(sub["all_sat"] == 1)

    image = draw_grid(ax, to_grid(sub, "gain_sat"), cmap, vmin, vmax)

    # 飽和領域内で利得が最大となる点
    best = find_max_gain(df)

    if best is not None:

        ax.plot(
            best["Vin_V"],
            best["Vb_V"],
            marker="*",
            markersize=14,
            color="tab:red",
            markeredgecolor="white",
            linestyle="none",
            label=(
                f"Max gain {best['gain_dc_dB']:.2f} dB\n"
                f"(Vin = {best['Vin_V']:.2f} V, Vb = {best['Vb_V']:.2f} V)"
            )
        )

        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("Vin [V]")
    ax.set_ylabel("Vb [V]")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("DC Gain [dB] (all transistors saturated)")

    ax.set_title(
        "5T OTA saturation region\n"
        "(gray: at least one transistor not saturated, white: not simulated)"
    )

    fig.savefig(MAP_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saturation map saved: {MAP_PATH}")


# ============================================================
# 2. トランジスタ別の飽和マップ
# ============================================================

def plot_by_transistor(df, transistors):

    os.makedirs(DETAIL_DIR, exist_ok=True)

    # 0: 非飽和 (灰), 1: 飽和 (青)
    cmap = ListedColormap([UNSAT_COLOR, "#2a6fb0"])

    panels = transistors + ["all"]

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.2 * len(panels), 3.4),
        sharex=True,
        sharey=True
    )

    for ax, name in zip(axes, panels):

        col = "all_sat" if name == "all" else f"{name}_sat"

        draw_grid(ax, to_grid(df, col), cmap, 0, 1)

        ax.set_title("All" if name == "all" else name)
        ax.set_xlabel("Vin [V]")

    axes[0].set_ylabel("Vb [V]")

    fig.suptitle("Saturated region per transistor (blue: saturated)")
    fig.tight_layout()

    path = os.path.join(
        DETAIL_DIR,
        "5tota_sat_by_transistor.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Per-transistor map saved: {path}")


# ============================================================
# 3. 飽和する Vin の下限・上限 vs Vb
# ============================================================

def plot_vin_range(df):

    sat = df[df["all_sat"] == 1]

    vin_range = (
        sat.groupby("Vb_V")["Vin_V"]
        .agg(["min", "max"])
        .reset_index()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharex=True,
        sharey=True
    )

    axes[0].plot(vin_range["Vb_V"], vin_range["min"], color="#2a6fb0", linewidth=2, marker="o", markersize=4)
    axes[1].plot(vin_range["Vb_V"], vin_range["max"], color="#2a6fb0", linewidth=2, marker="o", markersize=4)

    axes[0].set_title("Lower limit of saturated Vin")
    axes[1].set_title("Upper limit of saturated Vin")

    for ax in axes:
        ax.set_xlabel("Vb [V]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Vin [V]")

    fig.tight_layout()
    fig.savefig(RANGE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Vin range plot saved: {RANGE_PATH}")

    return vin_range


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    # 浮動小数点の誤差を丸める
    for col in ["Vb_V", "Vin_V"]:
        df[col] = df[col].round(3)

    transistors = get_transistors(df)

    print(f"Transistors: {transistors}")
    print(f"Conditions: {len(df)}, all saturated: {int(df['all_sat'].sum())}")

    os.makedirs(DETAIL_DIR, exist_ok=True)

    plot_saturation_map(df)
    plot_by_transistor(df, transistors)
    vin_range = plot_vin_range(df)

    # 飽和する Vin の範囲を表示
    print()
    print("Saturated Vin range [V]:")

    for _, r in vin_range.iterrows():
        print(
            f"  Vb = {r['Vb_V']:.2f} V: "
            f"{r['min']:.2f} ~ {r['max']:.2f}"
        )

    # 利得が最大となる点を表示
    best = find_max_gain(df)

    print()

    if best is not None:
        print(
            f"Max gain (all saturated): {best['gain_dc_dB']:.4f} dB "
            f"at Vb = {best['Vb_V']:.2f} V, Vin = {best['Vin_V']:.2f} V"
        )
    else:
        print("No all-saturated point")
