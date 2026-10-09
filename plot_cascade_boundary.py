import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# cascade 回路の飽和マップに、二乗則から導いた3本の境界線を重ねる
#
#   ① M2 が強反転      : Vin >= Vth2
#   ③ M2 が飽和        : Vb  >= Vth1 + (1 + a)(Vin - Vth2),  a = sqrt(K2 / K1)
#   ② M1 が飽和        : Vb  <= VDD + Vth1 - RD * K2 * (Vin - Vth2)^2
#
#   K = 1/2 µn Cox (W/L) で、Id = K (Vgs - Vth)^2 。
#   Vth と K は教科書の値ではなく、シミュレーション結果 (cascade_results.csv) から取り出す。
#     Vth : そのトランジスタが飽和している点の vth の中央値
#     K   : 同じ点で Id = K (Vgs - Vth)^2 を最小二乗で当てはめた値
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
    "cascade_boundary"
)

SUMMARY_PATH = os.path.join(
    OUT_DIR,
    "cascade_boundary_summary.csv"
)

# 電源電圧 [V] (cascade.sch の .param vdd)
VDD = 3.3

# RD, W, L の組ごとに描き分けるための列
KEYS = ["RD_ohm", "W1_um", "W2_um", "L1_um", "L2_um"]

# 一覧図 (W1 × W2) を1枚ずつ分ける列
OVERVIEW_KEYS = ["RD_ohm", "L1_um", "L2_um"]

# 線の色
COLOR_1 = "tab:purple"   # ① M2 強反転
COLOR_2 = "tab:red"      # ② M1 飽和
COLOR_3 = "tab:blue"     # ③ M2 飽和


# ============================================================
# パラメータの取り出し
# ============================================================

def extract_params(sub, name):
    """
    トランジスタ name (M1 / M2) が飽和している点から Vth と K を取り出す
    (点が足りなければ None)
    """

    sat = sub[sub[f"{name}_sat"] == 1]

    if len(sat) < 3:
        return None

    vth = sat[f"{name}_vth"].median()

    # Id = K x を原点を通る直線で当てはめる (x = (Vgs - Vth)^2、Vth は各点の値)
    x = (sat[f"{name}_vgs"] - sat[f"{name}_vth"]) ** 2
    k = (sat[f"{name}_id"] * x).sum() / (x * x).sum()

    return vth, k


def boundary_params(sub, rd):
    """
    3本の境界線に必要なパラメータをまとめて返す (取り出せなければ None)
    """

    p1 = extract_params(sub, "M1")
    p2 = extract_params(sub, "M2")

    if p1 is None or p2 is None:
        return None

    vth1, k1 = p1
    vth2, k2 = p2

    return {
        "Vth1": vth1,
        "Vth2": vth2,
        "K1": k1,
        "K2": k2,
        "a": np.sqrt(k2 / k1),
        "RD": rd
    }


# ============================================================
# 境界線と予測領域
# ============================================================

def line_3(vin, p):
    """③ M2 飽和の境界 (この線より上なら飽和)"""
    return p["Vth1"] + (1 + p["a"]) * (vin - p["Vth2"])


def line_2(vin, p):
    """② M1 飽和の境界 (この線より下なら飽和)"""
    return VDD + p["Vth1"] - p["RD"] * p["K2"] * (vin - p["Vth2"]) ** 2


def predict(vin, vb, p):
    """3本の不等式をすべて満たすか (二乗則による予測)"""
    return (
        (vin >= p["Vth2"])
        & (vb >= line_3(vin, p))
        & (vb <= line_2(vin, p))
    )


def iou(sub, p):
    """
    シミュレーションの飽和領域と、予測領域の重なり具合
    (共通部分の点数 / 和集合の点数。1 なら完全一致)
    """

    sim = sub["all_sat"].to_numpy() == 1
    pred = predict(sub["Vin_V"].to_numpy(), sub["Vb_V"].to_numpy(), p)

    union = np.sum(sim | pred)

    return np.sum(sim & pred) / union if union else float("nan")


# ============================================================
# 図
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


def draw_boundaries(ax, sub, p, with_label=True):
    """
    3本の境界線を描く (軸の範囲はシミュレーションの範囲に固定する)
    """

    vin_min, vin_max = sub["Vin_V"].min(), sub["Vin_V"].max()
    vb_min, vb_max = sub["Vb_V"].min(), sub["Vb_V"].max()

    vin = np.linspace(max(p["Vth2"], vin_min), vin_max, 400)

    # ① 縦の直線
    ax.axvline(
        p["Vth2"],
        color=COLOR_1,
        linewidth=1.5,
        label=f"(1) M2 on: Vin = Vth2 = {p['Vth2']:.3f} V" if with_label else None
    )

    # ③ 斜めの直線
    ax.plot(
        vin,
        line_3(vin, p),
        color=COLOR_3,
        linewidth=1.5,
        label=(
            f"(3) M2 sat: Vb = Vth1 + (1+a)(Vin - Vth2), a = {p['a']:.2f}"
            if with_label else None
        )
    )

    # ② 放物線
    ax.plot(
        vin,
        line_2(vin, p),
        color=COLOR_2,
        linewidth=1.5,
        label=(
            "(2) M1 sat: Vb = VDD + Vth1 - RD K2 (Vin - Vth2)^2"
            if with_label else None
        )
    )

    ax.set_xlim(vin_min, vin_max)
    ax.set_ylim(vb_min, vb_max)


def a_from_size(w1, w2, l1, l2):
    """W/L の比から決まる理論上の a = sqrt((W2/L2) / (W1/L1))"""
    return np.sqrt((w2 / l2) / (w1 / l1))


def plot_condition(sub, p, rd, w1, w2, l1, l2, score):

    fig, ax = plt.subplots(figsize=(7, 5.5))

    draw_saturation_map(ax, sub)
    draw_boundaries(ax, sub, p)

    ax.set_xlabel("Vin [V]")
    ax.set_ylabel("Vb [V]")
    ax.set_title(
        f"Cascade  RD = {rd:g} ohm, W1 = {w1:g} um, W2 = {w2:g} um, "
        f"L1 = {l1:g} um, L2 = {l2:g} um\n"
        f"green: all saturated (simulation), lines: square-law prediction, "
        f"IoU = {score:.2f}",
        fontsize=10
    )
    ax.legend(loc="upper left", fontsize=7)

    # 使ったパラメータを書いておく (理論上の a は W/L の比から)
    a_theory = a_from_size(w1, w2, l1, l2)

    ax.text(
        0.98,
        0.02,
        f"Vth1 = {p['Vth1']:.3f} V\n"
        f"Vth2 = {p['Vth2']:.3f} V\n"
        f"K1 = {p['K1'] * 1e3:.3f} mA/V$^2$\n"
        f"K2 = {p['K2'] * 1e3:.3f} mA/V$^2$\n"
        f"a = {p['a']:.2f} (W/L ratio: {a_theory:.2f})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        f"cascade_boundary_rd_{rd:g}_w1_{w1:g}_w2_{w2:g}_l1_{l1:g}_l2_{l2:g}.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_overview(rd_sub, rd, l1, l2, params):
    """
    RD, L1, L2 の組1つ分について、W1 (行) × W2 (列) の全組み合わせを1枚に並べる
    """

    w1_list = sorted(rd_sub["W1_um"].unique())
    w2_list = sorted(rd_sub["W2_um"].unique())

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
            sub = rd_sub[(rd_sub["W1_um"] == w1) & (rd_sub["W2_um"] == w2)]
            p = params.get((rd, w1, w2, l1, l2))

            if sub.empty:
                ax.set_visible(False)
                continue

            draw_saturation_map(ax, sub)

            if p is None:
                ax.set_title(f"W1 = {w1:g}, W2 = {w2:g}\nno fit", fontsize=9)
                continue

            draw_boundaries(ax, sub, p, with_label=False)

            ax.set_title(
                f"W1 = {w1:g}, W2 = {w2:g}\nIoU = {iou(sub, p):.2f}",
                fontsize=9
            )

    for ax in axes[-1]:
        ax.set_xlabel("Vin [V]")

    for ax in axes[:, 0]:
        ax.set_ylabel("Vb [V]")

    fig.suptitle(
        f"Cascade saturation region vs square-law boundaries, RD = {rd:g} ohm, "
        f"L1 = {l1:g} um, L2 = {l2:g} um\n"
        "(purple: M2 on, blue: M2 sat, red: M1 sat; rows: W1, columns: W2 [um])"
    )
    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        f"cascade_boundary_all_rd_{rd:g}_l1_{l1:g}_l2_{l2:g}.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    if not all(key in df.columns for key in KEYS):
        raise SystemExit(
            "CSVに RD_ohm, W1_um, W2_um, L1_um, L2_um の列がありません。"
            "最新の cascade.py でシミュレーションし直してください。"
        )

    # 浮動小数点の誤差を丸める
    for col in KEYS + ["Vb_V", "Vin_V"]:
        df[col] = df[col].round(3)

    os.makedirs(OUT_DIR, exist_ok=True)

    params = {}
    summary = []

    # RD, W, L の組ごとに、パラメータを取り出して重ね描き
    for (rd, w1, w2, l1, l2), sub in df.groupby(KEYS):

        p = boundary_params(sub, rd)

        if p is None:
            print(
                f"WARNING: RD = {rd:g}, W1 = {w1:g}, W2 = {w2:g}, "
                f"L1 = {l1:g}, L2 = {l2:g} はパラメータを取り出せません"
            )
            continue

        score = iou(sub, p)

        params[(rd, w1, w2, l1, l2)] = p
        plot_condition(sub, p, rd, w1, w2, l1, l2, score)

        summary.append({
            "RD_ohm": rd,
            "W1_um": w1,
            "W2_um": w2,
            "L1_um": l1,
            "L2_um": l2,
            "Vth1_V": p["Vth1"],
            "Vth2_V": p["Vth2"],
            "K1_A_per_V2": p["K1"],
            "K2_A_per_V2": p["K2"],
            "a_fit": p["a"],
            "a_WL_ratio": a_from_size(w1, w2, l1, l2),
            "IoU": score
        })

    # RD, L1, L2 の組ごとに一覧図
    for (rd, l1, l2), ov_sub in df.groupby(OVERVIEW_KEYS):
        plot_overview(ov_sub, rd, l1, l2, params)

    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False)
    print(f"Saved: {SUMMARY_PATH}")

    # 結果を表示
    print()
    print("Square-law boundary fit:")

    for s in summary:
        print(
            f"  RD = {s['RD_ohm']:g}, W1 = {s['W1_um']:g}, W2 = {s['W2_um']:g}, "
            f"L1 = {s['L1_um']:g}, L2 = {s['L2_um']:g}: "
            f"Vth1 = {s['Vth1_V']:.3f}, Vth2 = {s['Vth2_V']:.3f}, "
            f"K1 = {s['K1_A_per_V2'] * 1e3:.3f}, K2 = {s['K2_A_per_V2'] * 1e3:.3f} mA/V^2, "
            f"a = {s['a_fit']:.2f} (W/L ratio {s['a_WL_ratio']:.2f}), IoU = {s['IoU']:.2f}"
        )
