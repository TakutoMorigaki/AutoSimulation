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

# W × L の全組み合わせを並べた飽和マップ
OVERVIEW_PATH = os.path.join(
    DETAIL_DIR,
    "5tota_saturation_map_all.png"
)

# W, L の組ごとに描き分けるための列
WL_KEYS = ["W_um", "L_um"]

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


def wl_label(w, l):
    return f"W = {w:g} um, L = {l:g} um"


def wl_path(name, w, l):
    """
    W, L の組ごとの図の保存先 (例: 5tota_saturation_map_w_11.2_l_0.28.png)
    """

    return os.path.join(
        DETAIL_DIR,
        f"{name}_w_{w:g}_l_{l:g}.png"
    )


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


def gain_cmap():
    """
    飽和領域を DC Gain で塗るカラーマップ
    (飽和していない点は NaN にして灰色で表示)
    """

    # Bluesの白に近い端は背景と区別しにくいので使わない
    cmap = ListedColormap(
        plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256))
    )
    cmap.set_bad(UNSAT_COLOR)

    return cmap


def gain_range(df):
    """
    飽和している点の DC Gain の最小値・最大値 (カラーバーの範囲)
    """

    sat = df[df["all_sat"] == 1]

    if sat.empty:
        return None, None

    return sat["gain_dc_dB"].min(), sat["gain_dc_dB"].max()


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


def draw_gain_map(ax, sub, cmap, vmin, vmax):
    """
    飽和領域を DC Gain で塗ったマップを描く
    """

    sub = sub.copy()
    sub["gain_sat"] = sub["gain_dc_dB"].where(sub["all_sat"] == 1)

    return draw_grid(ax, to_grid(sub, "gain_sat"), cmap, vmin, vmax)


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
            f"Max gain {best['gain_dc_dB']:.2f} dB\n"
            f"(Vin = {best['Vin_V']:.2f} V, Vb = {best['Vb_V']:.2f} V)"
        )
    )


# ============================================================
# 1. 飽和マップ (飽和領域を DC Gain で色付け)
# ============================================================

def plot_saturation_map(sub, w, l):

    fig, ax = plt.subplots(figsize=(6, 4.5))

    vmin, vmax = gain_range(sub)

    image = draw_gain_map(ax, sub, gain_cmap(), vmin, vmax)

    # 飽和領域内で利得が最大となる点
    best = find_max_gain(sub)

    if best is not None:
        mark_max_gain(ax, best)
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("Vin [V]")
    ax.set_ylabel("Vb [V]")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("DC Gain [dB] (all transistors saturated)")

    ax.set_title(
        f"5T OTA saturation region ({wl_label(w, l)})\n"
        "(gray: at least one transistor not saturated, white: not simulated)"
    )

    path = wl_path("5tota_saturation_map", w, l)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saturation map saved: {path}")


# ============================================================
# 2. トランジスタ別の飽和マップ
# ============================================================

def plot_by_transistor(sub, w, l, transistors):

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

        draw_grid(ax, to_grid(sub, col), cmap, 0, 1)

        ax.set_title("All" if name == "all" else name)
        ax.set_xlabel("Vin [V]")

    axes[0].set_ylabel("Vb [V]")

    fig.suptitle(
        f"Saturated region per transistor ({wl_label(w, l)}, blue: saturated)"
    )
    fig.tight_layout()

    path = wl_path("5tota_sat_by_transistor", w, l)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Per-transistor map saved: {path}")


# ============================================================
# 3. 飽和する Vin の下限・上限 vs Vb
# ============================================================

def plot_vin_range(sub, w, l):

    sat = sub[sub["all_sat"] == 1]

    if sat.empty:
        print(f"No all-saturated point ({wl_label(w, l)}): Vin range plot skipped")
        return

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

    fig.suptitle(wl_label(w, l))
    fig.tight_layout()

    path = wl_path("5tota_saturation_vin_range", w, l)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Vin range plot saved: {path}")


# ============================================================
# 4. W (行) × L (列) の全組み合わせを並べた飽和マップ
# ============================================================

def plot_overview(df):

    w_list = sorted(df["W_um"].unique())
    l_list = sorted(df["L_um"].unique())

    fig, axes = plt.subplots(
        len(w_list),
        len(l_list),
        figsize=(3.6 * len(l_list), 2.8 * len(w_list)),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    # 組どうしを比べられるよう、カラーバーの範囲は全組で共通にする
    vmin, vmax = gain_range(df)
    cmap = gain_cmap()

    image = None

    for i, w in enumerate(w_list):

        for j, l in enumerate(l_list):

            ax = axes[i, j]
            sub = df[(df["W_um"] == w) & (df["L_um"] == l)]

            if sub.empty:
                ax.set_visible(False)
                continue

            image = draw_gain_map(ax, sub, cmap, vmin, vmax)

            best = find_max_gain(sub)

            if best is not None:
                mark_max_gain(ax, best)
                gain_text = f"max {best['gain_dc_dB']:.2f} dB"
            else:
                gain_text = "no saturated point"

            ax.set_title(
                f"W = {w:g}, L = {l:g}\n{gain_text}",
                fontsize=9
            )

    for ax in axes[-1]:
        ax.set_xlabel("Vin [V]")

    for ax in axes[:, 0]:
        ax.set_ylabel("Vb [V]")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.8)
        cbar.set_label("DC Gain [dB] (all transistors saturated)")

    fig.suptitle(
        "5T OTA all-saturated region (rows: W, columns: L [um])\n"
        "W: differential pair (current mirror and tail: 2W)"
    )

    fig.savefig(OVERVIEW_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Overview map saved: {OVERVIEW_PATH}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    if not all(key in df.columns for key in WL_KEYS):
        raise SystemExit(
            "CSVに W_um, L_um の列がありません。"
            "最新の 5TOTA.py でシミュレーションし直してください。"
        )

    # 浮動小数点の誤差を丸める
    for col in WL_KEYS + ["Vb_V", "Vin_V"]:
        df[col] = df[col].round(3)

    transistors = get_transistors(df)

    print(f"Transistors: {transistors}")
    print(f"Conditions: {len(df)}, all saturated: {int(df['all_sat'].sum())}")

    os.makedirs(DETAIL_DIR, exist_ok=True)

    # W, L の組ごとに描き分ける
    for (w, l), sub in df.groupby(WL_KEYS):

        print(f"\n{wl_label(w, l)}")

        plot_saturation_map(sub, w, l)
        plot_by_transistor(sub, w, l, transistors)
        plot_vin_range(sub, w, l)

    # 全組み合わせを並べた飽和マップ
    print()
    plot_overview(df)

    # W, L の組ごとに、利得が最大となる点を表示
    print()
    print("Max gain (all saturated):")

    for (w, l), sub in df.groupby(WL_KEYS):

        best = find_max_gain(sub)

        if best is not None:
            print(
                f"  {wl_label(w, l)}: {best['gain_dc_dB']:.4f} dB "
                f"at Vb = {best['Vb_V']:.2f} V, Vin = {best['Vin_V']:.2f} V"
            )
        else:
            print(f"  {wl_label(w, l)}: no all-saturated point")
