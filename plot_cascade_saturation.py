import os
import re
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

# 条件ごとの詳細図の保存先
DETAIL_DIR = os.path.join(
    STUDY_DIR,
    "cascade_saturation"
)

SUMMARY_PATH = os.path.join(
    STUDY_DIR,
    "cascade_saturation_map.png"
)


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


def saturated_ranges(vin, mask):
    """
    飽和している連続区間 [(start, end), ...] を返す
    """

    ranges = []
    start = None

    for x, m in zip(vin, mask):

        if m and start is None:
            start = x

        if not m and start is not None:
            ranges.append((start, prev))
            start = None

        prev = x

    if start is not None:
        ranges.append((start, prev))

    return ranges


def shade_saturation(ax, vin, mask):

    ax.fill_between(
        vin,
        0,
        1,
        where=mask,
        transform=ax.get_xaxis_transform(),
        color="tab:green",
        alpha=0.15,
        step="mid",
        label="All saturated"
    )


# ============================================================
# 条件ごとの詳細図
# ============================================================

def plot_detail(sub, vb, transistors):

    sub = sub.sort_values("Vin_V")

    vin = sub["Vin_V"].to_numpy()
    vout = sub["Vout_V"].to_numpy()
    mask = sub["all_sat"].to_numpy().astype(bool)

    # 小信号ゲイン dVout/dVin
    gain = np.gradient(vout, vin) if len(vin) > 1 else np.zeros_like(vin)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 10),
        sharex=True
    )

    # ---- Vout ----
    ax = axes[0]
    shade_saturation(ax, vin, mask)
    ax.plot(vin, vout, color="black")
    ax.set_ylabel("Vout [V]")
    ax.set_title(
        f"Cascade  Vb = {vb:.2f} V"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # ---- 飽和マージン ----
    ax = axes[1]
    shade_saturation(ax, vin, mask)

    for name in transistors:

        margin = (
            sub[f"{name}_vds"].abs()
            - sub[f"{name}_vdsat"].abs()
        )

        ax.plot(vin, margin, label=name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("|Vds| - |Vdsat| [V]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # ---- ゲイン ----
    ax = axes[2]
    shade_saturation(ax, vin, mask)
    ax.plot(vin, gain, color="tab:blue")
    ax.set_ylabel("dVout/dVin [V/V]")
    ax.set_xlabel("Vin [V]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = os.path.join(
        DETAIL_DIR,
        f"cascade_sat_vb_{vb:.2f}.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 飽和区間を表示
    ranges = saturated_ranges(vin, mask)

    print(f"\nVb = {vb:.2f} V")

    if ranges:

        for start, end in ranges:

            in_range = (vin >= start) & (vin <= end)

            print(
                f"  Saturated Vin: {start:.3f} ~ {end:.3f} V, "
                f"max |gain| = {np.max(np.abs(gain[in_range])):.3g}"
            )

    else:
        print("  No saturated region")

    print(f"  Saved: {path}")


# ============================================================
# Vb × Vin の飽和マップ
# ============================================================

def plot_summary(df):

    fig, ax = plt.subplots(figsize=(6, 5))

    grid = df.pivot_table(
        index="Vb_V",
        columns="Vin_V",
        values="all_sat",
        aggfunc="max"
    )

    vin = grid.columns.to_numpy()
    vb = grid.index.to_numpy()

    ax.pcolormesh(
        vin,
        vb,
        grid.to_numpy(),
        cmap="Greens",
        vmin=0,
        vmax=1.5,
        shading="nearest"
    )

    ax.set_xlabel("Vin [V]")
    ax.set_ylabel("Vb [V]")
    ax.set_title("All saturated")

    fig.tight_layout()
    fig.savefig(SUMMARY_PATH, dpi=300, bbox_inches="tight")

    print(f"\nSaturation map saved: {SUMMARY_PATH}")


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

    transistors = get_transistors(df)

    print(f"Transistors: {transistors}")

    os.makedirs(DETAIL_DIR, exist_ok=True)

    for vb, sub in df.groupby("Vb_V"):
        plot_detail(sub, vb, transistors)

    # Vb を掃引しているときだけ全体マップを描く
    if df["Vb_V"].nunique() > 1:
        plot_summary(df)
