import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# cascade 回路のヘッドルームの可視化
#
#   cascade_results.csv (cascade.py の出力) から、RD, W, L の組ごとに
#   代表的な Vb について Vin–Vout 特性を取り出し、
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

# 全組の結果をまとめた CSV
SUMMARY_PATH = os.path.join(
    OUT_DIR,
    "cascade_headroom_summary.csv"
)


# ============================================================
# 対象とする条件
# ============================================================

# 回路定数 (リストの全組み合わせを処理する。None なら CSV にある値すべて)
RD_LIST = [10000]
W1_LIST = [5.6, 11.2, 22.4, 44.8]
W2_LIST = [5.6, 11.2, 22.4, 44.8]
L1_LIST = [0.28, 0.56, 1.12]
L2_LIST = [0.28, 0.56, 1.12]

# 代表的な Vb [V]
#   None : 組ごとに飽和マップから自動で選ぶ (select_vb_list を参照)
#   リスト : すべての組で同じ Vb を使う (例: [1.20, 2.00, 2.34, 2.80])
VB_LIST = None

KEYS = ["RD_ohm", "W1_um", "W2_um", "L1_um", "L2_um"]

TRANSISTORS = ["M1", "M2"]


# ============================================================
# 条件の表示
# ============================================================

def cond_label(rd, w1, w2, l1, l2):
    return (
        f"RD = {rd:g} ohm, W1 = {w1:g} um, W2 = {w2:g} um, "
        f"L1 = {l1:g} um, L2 = {l2:g} um"
    )


def cond_tag(rd, w1, w2, l1, l2):
    return f"rd_{rd:g}_w1_{w1:g}_w2_{w2:g}_l1_{l1:g}_l2_{l2:g}"


# ============================================================
# 読み込み
# ============================================================

def load_csv():
    """
    CSV が大きいので、少しずつ読みながら対象の条件の行だけを残す
    """

    lists = [RD_LIST, W1_LIST, W2_LIST, L1_LIST, L2_LIST]
    chunks = []

    for chunk in pd.read_csv(CSV_PATH, chunksize=200000):

        if not all(key in chunk.columns for key in KEYS):
            raise SystemExit(
                "CSVに RD_ohm, W1_um, W2_um, L1_um, L2_um の列がありません。"
                "最新の cascade.py でシミュレーションし直してください。"
            )

        # 浮動小数点の誤差を丸める
        for col in KEYS + ["Vb_V", "Vin_V"]:
            chunk[col] = chunk[col].round(3)

        mask = np.ones(len(chunk), dtype=bool)

        for key, values in zip(KEYS, lists):
            if values is not None:
                mask &= chunk[key].isin([round(v, 3) for v in values]).to_numpy()

        chunks.append(chunk[mask])

    return pd.concat(chunks, ignore_index=True)


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


def select_vb_list(cond_df):
    """
    1つの組について、代表的な Vb を飽和マップから選ぶ

      ・飽和点がある Vb の範囲の 1/4 と 3/4 の位置 (低め・高め)
      ・出力振幅が最大の Vb
      ・利得 |dVout/dVin| が最大の Vb
    """

    swing = {}
    gain = {}

    for vb in sorted(cond_df["Vb_V"].unique()):

        sub = load_condition(cond_df, vb)
        sat = sub[sub["all_sat"] == 1]

        if sat.empty:
            continue

        swing[vb] = sat["Vout_V"].max() - sat["Vout_V"].min()
        gain[vb] = sat["gain"].abs().max()

    if not swing:
        return []

    vbs = np.array(sorted(swing))
    lo, hi = vbs.min(), vbs.max()

    def nearest(v):
        return vbs[np.abs(vbs - v).argmin()]

    picks = {
        nearest(lo + 0.25 * (hi - lo)),
        max(swing, key=swing.get),
        max(gain, key=gain.get),
        nearest(lo + 0.75 * (hi - lo)),
    }

    return sorted(float(v) for v in picks)


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
# 図 (組ごと)
# ============================================================

def plot_swing_summary(results, label, tag):
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
        f"{label}",
        fontsize=10
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    path = os.path.join(OUT_DIR, f"cascade_headroom_swing_{tag}.png")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# 図 (組どうしの比較)
# ============================================================

def plot_compare(summary):
    """
    RD, L1, L2 の組ごとに、W1 (行) × W2 (列) の最大出力振幅を並べる
    (各組で、代表 Vb のうち出力振幅が最大のものを使う)
    """

    best = (
        summary.sort_values("Vout_swing_V", ascending=False)
        .groupby(KEYS, as_index=False)
        .first()
    )

    vmax = best["Vout_swing_V"].max()

    for (rd, l1, l2), sub in best.groupby(["RD_ohm", "L1_um", "L2_um"]):

        grid = sub.pivot_table(
            index="W1_um",
            columns="W2_um",
            values="Vout_swing_V",
            aggfunc="first"
        )

        w1_list = grid.index.to_numpy()
        w2_list = grid.columns.to_numpy()

        fig, ax = plt.subplots(
            figsize=(1.6 * len(w2_list) + 2.5, 1.3 * len(w1_list) + 1.8)
        )

        # 組どうしを比べられるよう、色の範囲は全組で共通にする
        image = ax.imshow(
            grid.to_numpy(),
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            aspect="auto"
        )

        for i, w1 in enumerate(w1_list):

            for j, w2 in enumerate(w2_list):

                hit = sub[(sub["W1_um"] == w1) & (sub["W2_um"] == w2)]

                if hit.empty:
                    continue

                r = hit.iloc[0]
                color = "white" if r["Vout_swing_V"] > 0.6 * vmax else "black"

                ax.text(
                    j,
                    i,
                    f"{r['Vout_swing_V']:.2f} V\n"
                    f"Vb = {r['Vb_V']:.2f}\n"
                    f"|gain| {abs(r['op_gain']):.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color
                )

        ax.set_xticks(range(len(w2_list)))
        ax.set_xticklabels([f"{w:g}" for w in w2_list])
        ax.set_yticks(range(len(w1_list)))
        ax.set_yticklabels([f"{w:g}" for w in w1_list])
        ax.set_xlabel("W2 [um]")
        ax.set_ylabel("W1 [um]")

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Max output swing [V]")

        ax.set_title(
            "Cascade max output swing (best of representative Vb)\n"
            f"RD = {rd:g} ohm, L1 = {l1:g} um, L2 = {l2:g} um",
            fontsize=10
        )

        fig.tight_layout()

        path = os.path.join(
            OUT_DIR,
            f"cascade_headroom_compare_rd_{rd:g}_l1_{l1:g}_l2_{l2:g}.png"
        )

        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {path}")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    df = load_csv()

    if df.empty:
        raise SystemExit("指定した RD, W, L の組のデータが CSV にありません")

    os.makedirs(OUT_DIR, exist_ok=True)

    summary = []

    for (rd, w1, w2, l1, l2), cond_df in df.groupby(KEYS):

        label = cond_label(rd, w1, w2, l1, l2)

        print(f"\n{label}")

        vb_list = VB_LIST if VB_LIST is not None else select_vb_list(cond_df)

        results = []

        for vb in vb_list:

            sub = load_condition(cond_df, vb)

            if sub.empty:
                print(f"WARNING: Vb = {vb:.2f} V のデータがありません")
                continue

            res = analyze(sub, vb)

            if res is None:
                print(f"WARNING: Vb = {vb:.2f} V に全トランジスタ飽和の点がありません")
                continue

            results.append(res)

        if not results:
            print("WARNING: 解析できる Vb がありませんでした")
            continue

        plot_swing_summary(results, label, cond_tag(rd, w1, w2, l1, l2))

        for res in results:
            summary.append({
                "RD_ohm": rd, "W1_um": w1, "W2_um": w2, "L1_um": l1, "L2_um": l2,
                **res
            })

            print(
                f"  Vb = {res['Vb_V']:.2f} V: "
                f"Vout {res['Vout_swing_min_V']:.3f} ~ {res['Vout_swing_max_V']:.3f} V "
                f"(swing {res['Vout_swing_V']:.3f} V, "
                f"limited by {res['limit_at_min']} / {res['limit_at_max']}), "
                f"op margin {res['op_min_margin_V']:.3f} V ({res['op_bottleneck']}), "
                f"op |gain| {abs(res['op_gain']):.3f}"
            )

    if not summary:
        raise SystemExit("解析できる組がありませんでした")

    summary = pd.DataFrame(summary)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved: {SUMMARY_PATH}")

    # 組どうしの比較図
    plot_compare(summary)
