import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# cascade 回路のヘッドルームの可視化
#
#   cascade_results.csv (cascade.py の出力) から、代表的な Vb について
#   Vin–Vout 特性を取り出し、
#     ② 出力振幅 : 全トランジスタが飽和している Vout の範囲
#     ③ 動作点の余裕 : 動作点での |Vds| - |Vdsat|
#   を図にする。新たなシミュレーションは行わない。
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
    "cascade_results.csv"
)

# 図の保存先
OUT_DIR = os.path.join(
    STUDY_DIR,
    "cascade_headroom"
)

SUMMARY_PATH = os.path.join(
    OUT_DIR,
    "cascade_headroom_summary.csv"
)

SWING_PATH = os.path.join(
    OUT_DIR,
    "cascade_headroom_swing.png"
)


# ============================================================
# 対象とする条件
# ============================================================

# 回路定数 (1組だけ見る)
RD_OHM = 2000
W1_UM = 11.2
W2_UM = 11.2

# 代表的な Vb [V]
#   1.20 : 低め
#   2.00 : 出力振幅が最大
#   2.34 : 利得 |dVout/dVin| が最大
#   2.80 : 高め
VB_LIST = [1.20, 2.00, 2.34, 2.80]

TRANSISTORS = ["M1", "M2"]


# ============================================================
# 解析
# ============================================================

def load_condition(df, vb):
    """
    指定した Vb の行を Vin 順に取り出し、gain と飽和マージンの列を追加する
    """

    sub = df[df["Vb_V"] == round(vb, 3)].sort_values("Vin_V").copy()

    if len(sub) > 1:
        sub["gain"] = np.gradient(
            sub["Vout_V"].to_numpy(),
            sub["Vin_V"].to_numpy()
        )
    else:
        sub["gain"] = 0.0

    for name in TRANSISTORS:
        sub[f"{name}_margin"] = (
            sub[f"{name}_vds"].abs() - sub[f"{name}_vdsat"].abs()
        )

    return sub.reset_index(drop=True)


def unsaturated_neighbor(sub, i):
    """
    飽和区間の端 i のすぐ外側の点で、飽和していないトランジスタを返す
    """

    for j in (i - 1, i + 1):

        if 0 <= j < len(sub) and sub.loc[j, "all_sat"] != 1:

            names = [
                name for name in TRANSISTORS
                if sub.loc[j, f"{name}_sat"] != 1
            ]

            return ",".join(names) if names else "-"

    # 掃引範囲の端まで飽和している
    return "sweep end"


def analyze(sub, vb):
    """
    出力振幅と動作点の余裕を求める
    """

    sat = sub[sub["all_sat"] == 1]

    if sat.empty:
        return None

    idx = sat.index.to_numpy()

    if idx[-1] - idx[0] + 1 != len(idx):
        print(f"WARNING: Vb = {vb:.2f} V で飽和区間が連続していません")

    i_low = sat["Vout_V"].idxmin()
    i_high = sat["Vout_V"].idxmax()

    vout_low = sub.loc[i_low, "Vout_V"]
    vout_high = sub.loc[i_high, "Vout_V"]

    # 動作点 : Vout が出力振幅の中央に最も近い飽和点
    center = (vout_low + vout_high) / 2
    i_op = (sat["Vout_V"] - center).abs().idxmin()
    op = sub.loc[i_op]

    margins = {name: op[f"{name}_margin"] for name in TRANSISTORS}
    bottleneck = min(margins, key=margins.get)

    return {
        "Vb_V": vb,
        "Vin_sat_min_V": sat["Vin_V"].min(),
        "Vin_sat_max_V": sat["Vin_V"].max(),
        "Vout_swing_min_V": vout_low,
        "Vout_swing_max_V": vout_high,
        "Vout_swing_V": vout_high - vout_low,
        "limit_at_min": unsaturated_neighbor(sub, i_low),
        "limit_at_max": unsaturated_neighbor(sub, i_high),
        "op_Vin_V": op["Vin_V"],
        "op_Vout_V": op["Vout_V"],
        "op_gain": op["gain"],
        "max_abs_gain": sat["gain"].abs().max(),
        **{f"op_{name}_margin_V": margins[name] for name in TRANSISTORS},
        "op_min_margin_V": margins[bottleneck],
        "op_bottleneck": bottleneck
    }


# ============================================================
# 図
# ============================================================

def shade_saturation(ax, sub):

    ax.fill_between(
        sub["Vin_V"],
        0,
        1,
        where=sub["all_sat"].to_numpy() == 1,
        transform=ax.get_xaxis_transform(),
        color="tab:green",
        alpha=0.15,
        step="mid",
        label="All saturated"
    )


def plot_condition(sub, res):

    vb = res["Vb_V"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True
    )

    # ---- Vout ----
    ax = axes[0]
    shade_saturation(ax, sub)
    ax.plot(sub["Vin_V"], sub["Vout_V"], color="black")

    for key, text in [("Vout_swing_max_V", "max"), ("Vout_swing_min_V", "min")]:
        ax.axhline(res[key], color="tab:red", linestyle="--", linewidth=1)
        ax.annotate(
            f"{text} {res[key]:.3f} V",
            xy=(1.0, res[key]),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="tab:red"
        )

    ax.plot(
        res["op_Vin_V"],
        res["op_Vout_V"],
        marker="o",
        color="tab:red",
        linestyle="none",
        label=f"Operating point (Vin = {res['op_Vin_V']:.2f} V)"
    )

    ax.set_ylabel("Vout [V]")
    ax.set_title(
        f"Cascade  Vb = {vb:.2f} V  "
        f"(RD = {RD_OHM:g} ohm, W1 = {W1_UM:g} um, W2 = {W2_UM:g} um)\n"
        f"Output swing {res['Vout_swing_V']:.3f} V"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 飽和マージン ----
    ax = axes[1]
    shade_saturation(ax, sub)

    for name in TRANSISTORS:
        ax.plot(sub["Vin_V"], sub[f"{name}_margin"], label=name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(res["op_Vin_V"], color="tab:red", linestyle=":", linewidth=1)
    ax.set_ylabel("|Vds| - |Vdsat| [V]")
    ax.set_title(
        f"Margin at operating point: min {res['op_min_margin_V']:.3f} V "
        f"({res['op_bottleneck']})",
        fontsize=9
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- ゲイン ----
    ax = axes[2]
    shade_saturation(ax, sub)
    ax.plot(sub["Vin_V"], sub["gain"], color="tab:blue")
    ax.axvline(res["op_Vin_V"], color="tab:red", linestyle=":", linewidth=1)
    ax.set_ylabel("dVout/dVin [V/V]")
    ax.set_xlabel("Vin [V]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        f"cascade_headroom_vb_{vb:.2f}.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_swing_summary(results):
    """
    代表点ごとの出力振幅を縦棒で並べ、動作点と余裕を書き込む
    """

    fig, ax = plt.subplots(figsize=(1.6 * len(results) + 2, 5))

    x = np.arange(len(results))

    for i, res in enumerate(results):

        ax.plot(
            [i, i],
            [res["Vout_swing_min_V"], res["Vout_swing_max_V"]],
            color="#2a6fb0",
            linewidth=10,
            solid_capstyle="butt"
        )

        ax.plot(i, res["op_Vout_V"], marker="o", color="tab:red")

        ax.annotate(
            f"swing {res['Vout_swing_V']:.2f} V\n"
            f"margin {res['op_min_margin_V']:.2f} V\n"
            f"|gain| {abs(res['op_gain']):.2f}",
            xy=(i, res["Vout_swing_min_V"]),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Vb = {res['Vb_V']:.2f}" for res in results])
    ax.set_xlim(-0.6, len(results) - 0.4)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.4)
    ax.set_ylabel("Vout [V]")
    ax.set_title(
        "Cascade output swing (bar: all saturated, dot: operating point)\n"
        f"RD = {RD_OHM:g} ohm, W1 = {W1_UM:g} um, W2 = {W2_UM:g} um"
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(SWING_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {SWING_PATH}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    # 浮動小数点の誤差を丸める
    for col in ["RD_ohm", "W1_um", "W2_um", "Vb_V", "Vin_V"]:
        df[col] = df[col].round(3)

    df = df[
        (df["RD_ohm"] == RD_OHM)
        & (df["W1_um"] == W1_UM)
        & (df["W2_um"] == W2_UM)
    ]

    if df.empty:
        raise SystemExit(
            f"RD = {RD_OHM:g}, W1 = {W1_UM:g}, W2 = {W2_UM:g} のデータが CSV にありません"
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    results = []

    for vb in VB_LIST:

        sub = load_condition(df, vb)

        if sub.empty:
            print(f"WARNING: Vb = {vb:.2f} V のデータがありません")
            continue

        res = analyze(sub, vb)

        if res is None:
            print(f"WARNING: Vb = {vb:.2f} V に全トランジスタ飽和の点がありません")
            continue

        plot_condition(sub, res)
        results.append(res)

    if not results:
        raise SystemExit("解析できる Vb がありませんでした")

    plot_swing_summary(results)

    pd.DataFrame(results).to_csv(SUMMARY_PATH, index=False)
    print(f"Saved: {SUMMARY_PATH}")

    # 結果を表示
    print()
    print("Headroom summary:")

    for res in results:
        print(
            f"  Vb = {res['Vb_V']:.2f} V: "
            f"Vout {res['Vout_swing_min_V']:.3f} ~ {res['Vout_swing_max_V']:.3f} V "
            f"(swing {res['Vout_swing_V']:.3f} V, "
            f"limited by {res['limit_at_min']} / {res['limit_at_max']}), "
            f"op margin {res['op_min_margin_V']:.3f} V ({res['op_bottleneck']}), "
            f"op |gain| {abs(res['op_gain']):.3f}"
        )
