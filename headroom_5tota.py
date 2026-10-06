import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 5TOTA の開ループ出力振幅 (ヘッドルーム) の可視化
#
#   tb_dc_5tota.sch (開ループ) で、代表的な (Vb, Vcm) ごとに
#   Vinn = Vcm, Vinp = Vcm + Vd として差動入力 Vd を DC 掃引し、
#     ② 出力振幅 : 全トランジスタが飽和している Vout の範囲
#     ③ 動作点の余裕 : Vd = 0 (Vinp = Vinn) での |Vds| - |Vdsat|
#   を求めて図にする。
#
#   掃引範囲・保存するベクトル・書き出し (headroom.txt) は
#   回路図の .control に書いてあり、ここでは .param と W, L だけを書き換える。
# ============================================================


# ============================================================
# パス
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

STUDY_DIR = os.path.join(PROJECT_ROOT, "study")
NETLIST_DIR = os.path.join(STUDY_DIR, "netlists")

# 開ループのテストベンチ
SCH_PATH = os.path.join(STUDY_DIR, "tb_dc_5tota.sch")

# AC 解析の結果 (利得の検算に使う)
SAT_CSV_PATH = os.path.join(STUDY_DIR, "5tota_saturation_results.csv")

# 図の保存先
OUT_DIR = os.path.join(STUDY_DIR, "5tota_headroom")
SUMMARY_PATH = os.path.join(OUT_DIR, "5tota_headroom_summary.csv")
SWING_PATH = os.path.join(OUT_DIR, "5tota_headroom_swing.png")

env = os.environ.copy()


# ============================================================
# 対象とする条件
# ============================================================

# 差動対の W, L [um] (カレントミラーとテールは 2W、L は共通)
W_UM = 11.2
L_UM = 0.28

# 代表的な (Vb, Vcm) [V]
#   飽和マップ (W = 11.2, L = 0.28) では、Vb = 0.70 V のとき Vcm は 0.78 ~ 3.00 V
POINTS = [
    (0.70, 0.80),   # 最大利得の点 (飽和領域の角)
    (0.70, 0.90),   # 同じ Vb で Vcm 低
    (0.70, 2.00),   # 同じ Vb で Vcm 中
    (0.70, 2.90),   # 同じ Vb で Vcm 高
    (1.00, 2.00),   # Vb を上げる (Vcm 中)
    (1.30, 2.00),   # Vb をさらに上げる (Vcm 中)
]

# 回路図の .control で wrdata が書き出すファイル名
DATA_FILENAME = "headroom.txt"

TRANSISTORS = ["XM1", "XM2", "XM3", "XM4", "XM5"]
DEVICE_PARAMS = ["vds", "vdsat", "vgs", "vth"]


# ============================================================
# ネットリスト
# ============================================================

def export_netlist(sch_path, output_dir):
    """xschemをバッチモードで実行し、SPICEネットリストを生成する"""
    os.makedirs(output_dir, exist_ok=True)

    cmd = ["xschem", "-q", "-n", "-s", "-o", output_dir, sch_path]
    print(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, cwd=STUDY_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("xschem Error:", result.stderr)
        return None

    base_name = os.path.splitext(os.path.basename(sch_path))[0]
    return os.path.join(output_dir, f"{base_name}.spice")


def replace_mos_parameter(content, transistor, parameter, value):
    """
    指定したトランジスタのWまたはLを変更する。

    例:
        XM3 ... L=0.28u W=11.2u
    ↓
        XM3 ... L=0.50u W=20u
    """
    lines = content.splitlines()

    pattern = re.compile(
        rf'(\b{re.escape(transistor)}\b.*?\b{parameter}\s*=\s*)'
        rf'[\d.+\-eE]+u',
        flags=re.IGNORECASE
    )

    for i, line in enumerate(lines):

        if re.search(rf'^\s*{re.escape(transistor)}\b', line, flags=re.IGNORECASE):

            new_line, count = pattern.subn(rf'\g<1>{value}u', line)

            if count > 0:
                lines[i] = new_line
            else:
                print(f"WARNING: {transistor} の {parameter} が見つかりません")

            return "\n".join(lines)

    print(f"WARNING: {transistor} が ネットリスト内に見つかりません")
    return content


def set_param(content, name, value):
    """.param name=value の値を書き換える"""
    content, count = re.subn(
        rf'(\.param\s+{name}\s*=\s*)[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',
        rf'\g<1>{value}',
        content,
        flags=re.IGNORECASE
    )

    if count == 0:
        print(f"WARNING: .param {name} がネットリスト内に見つかりません")

    return content


def build_netlist(template_content, vb, vcm):
    """
    条件 (Vb, Vcm) と W, L を設定したネットリストを返す
    """
    content = template_content

    content = set_param(content, "vb", vb)
    content = set_param(content, "vcm", vcm)

    # W : 差動対 M1, M2 は W、カレントミラー M3, M4 とテール M5 は 2W
    for transistor in ["XM1", "XM2"]:
        content = replace_mos_parameter(content, transistor, "W", W_UM)

    for transistor in ["XM3", "XM4", "XM5"]:
        content = replace_mos_parameter(content, transistor, "W", 2.0 * W_UM)

    for transistor in TRANSISTORS:
        content = replace_mos_parameter(content, transistor, "L", L_UM)

    return content


# ============================================================
# シミュレーション
# ============================================================

def simulate_point(template_content, work_dir, vb, vcm):
    """
    1つの (Vb, Vcm) について Vd を DC 掃引し、結果を DataFrame で返す
    (失敗したら None)

    wrdata の書き出し先は回路図で固定 (headroom.txt) なので、
    条件ごとに別のフォルダで ngspice を実行して衝突を防ぐ。
    """
    point_dir = os.path.join(work_dir, f"vb_{vb:.2f}_vcm_{vcm:.2f}")
    os.makedirs(point_dir, exist_ok=True)

    spice_path = os.path.join(point_dir, "tb_dc_5tota.spice")
    data_path = os.path.join(point_dir, DATA_FILENAME)

    with open(spice_path, "w") as f:
        f.write(build_netlist(template_content, vb, vcm))

    result = subprocess.run(
        ["ngspice", "-b", spice_path],
        cwd=point_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0 or not os.path.exists(data_path):
        print(f"Vb = {vb:.2f}, Vcm = {vcm:.2f}: Simulation Failed")
        print(result.stderr)
        return None

    # 1行目は列名、1列目は掃引した Vd
    values = np.loadtxt(data_path, skiprows=1, ndmin=2)

    columns = ["Vd_V", "Vout_V"] + [
        f"{t.replace('XM', 'M')}_{p}" for t in TRANSISTORS for p in DEVICE_PARAMS
    ]

    if values.shape[1] != len(columns):
        print(
            f"Vb = {vb:.2f}, Vcm = {vcm:.2f}: 出力の列数が想定と違います "
            f"({values.shape[1]} 列, 想定 {len(columns)} 列)"
        )
        return None

    df = pd.DataFrame(values, columns=columns)
    df["Vinp_V"] = vcm + df["Vd_V"]

    print(f"Vb = {vb:.2f}, Vcm = {vcm:.2f}: Simulation Success ({len(df)} points)")

    return df


# ============================================================
# 解析
# ============================================================

def add_saturation(df):
    """
    |Vgs| > |Vth| かつ |Vds| >= |Vdsat| なら飽和 (5TOTA.py と同じ基準)
    """
    all_sat = np.ones(len(df), dtype=bool)

    for t in TRANSISTORS:
        name = t.replace("XM", "M")

        on = df[f"{name}_vgs"].abs() > df[f"{name}_vth"].abs()
        sat = df[f"{name}_vds"].abs() >= df[f"{name}_vdsat"].abs()

        df[f"{name}_sat"] = (on & sat).astype(int)
        df[f"{name}_margin"] = df[f"{name}_vds"].abs() - df[f"{name}_vdsat"].abs()

        all_sat &= (on & sat).to_numpy()

    df["all_sat"] = all_sat.astype(int)
    df["gain"] = np.gradient(df["Vout_V"].to_numpy(), df["Vd_V"].to_numpy())

    return df


def unsaturated_neighbor(df, i):
    """
    飽和区間の端 i のすぐ外側の点で、飽和していないトランジスタを返す
    """
    for j in (i - 1, i + 1):

        if 0 <= j < len(df) and df.loc[j, "all_sat"] != 1:

            names = [
                t.replace("XM", "M") for t in TRANSISTORS
                if df.loc[j, f"{t.replace('XM', 'M')}_sat"] != 1
            ]

            return ",".join(names) if names else "-"

    # 掃引範囲の端まで飽和している (回路図の dc vd の範囲を広げる必要がある)
    return "sweep end"


def lookup_ac_gain(ac_df, vb, vcm):
    """AC 解析 (フォロワ接続) の gain_dc [dB] を取り出す (なければ None)"""
    if ac_df is None:
        return None

    hit = ac_df[(ac_df["Vb_V"] == round(vb, 3)) & (ac_df["Vin_V"] == round(vcm, 3))]

    if hit.empty:
        return None

    return hit["gain_dc_dB"].iloc[0]


def analyze(df, vb, vcm, ac_df):
    """
    出力振幅・動作点の余裕・オフセット・利得を求める
    """
    sat = df[df["all_sat"] == 1]

    if sat.empty:
        return None

    idx = sat.index.to_numpy()

    if idx[-1] - idx[0] + 1 != len(idx):
        print(f"WARNING: Vb = {vb:.2f}, Vcm = {vcm:.2f} で飽和区間が連続していません")

    i_low = sat["Vout_V"].idxmin()
    i_high = sat["Vout_V"].idxmax()

    vout_low = df.loc[i_low, "Vout_V"]
    vout_high = df.loc[i_high, "Vout_V"]

    # 動作点 : Vinp = Vinn (差動入力 0)
    i_op = df["Vd_V"].abs().idxmin()
    op = df.loc[i_op]

    margins = {t.replace("XM", "M"): op[f"{t.replace('XM', 'M')}_margin"] for t in TRANSISTORS}
    bottleneck = min(margins, key=margins.get)

    # 入力オフセット : Vout が出力振幅の中央に来る差動入力
    center = (vout_low + vout_high) / 2
    i_center = (sat["Vout_V"] - center).abs().idxmin()

    # 利得の検算 : Vout ≈ Vcm (AC 解析と同じ動作点) での傾き
    i_follow = (df["Vout_V"] - vcm).abs().idxmin()
    dc_gain = abs(df.loc[i_follow, "gain"])
    dc_gain_db = 20 * np.log10(dc_gain) if dc_gain > 0 else float("nan")

    return {
        "Vb_V": vb,
        "Vcm_V": vcm,
        "Vout_swing_min_V": vout_low,
        "Vout_swing_max_V": vout_high,
        "Vout_swing_V": vout_high - vout_low,
        "limit_at_min": unsaturated_neighbor(df, i_low),
        "limit_at_max": unsaturated_neighbor(df, i_high),
        "op_Vout_V": op["Vout_V"],
        "op_all_sat": int(op["all_sat"]),
        **{f"op_{name}_margin_V": margins[name] for name in margins},
        "op_min_margin_V": margins[bottleneck],
        "op_bottleneck": bottleneck,
        "offset_V": df.loc[i_center, "Vd_V"],
        "dc_gain_dB_at_Vout_eq_Vcm": dc_gain_db,
        "ac_gain_dB": lookup_ac_gain(ac_df, vb, vcm),
    }


# ============================================================
# 図
# ============================================================

def shade_saturation(ax, df):

    ax.fill_between(
        df["Vd_V"] * 1e3,
        0,
        1,
        where=df["all_sat"].to_numpy() == 1,
        transform=ax.get_xaxis_transform(),
        color="tab:green",
        alpha=0.15,
        step="mid",
        label="All saturated"
    )


def plot_point(df, res):

    vb, vcm = res["Vb_V"], res["Vcm_V"]
    vd_mv = df["Vd_V"] * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ---- Vout ----
    ax = axes[0]
    shade_saturation(ax, df)
    ax.plot(vd_mv, df["Vout_V"], color="black")

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

    ax.plot(0, res["op_Vout_V"], marker="o", color="tab:red", linestyle="none",
            label="Operating point (Vinp = Vinn)")

    ax.set_ylabel("Vout [V]")
    ax.set_title(
        f"5T OTA open loop  Vb = {vb:.2f} V, Vcm = {vcm:.2f} V  "
        f"(W = {W_UM:g} um, L = {L_UM:g} um)\n"
        f"Output swing {res['Vout_swing_V']:.3f} V "
        f"(limited by {res['limit_at_min']} / {res['limit_at_max']})"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 飽和マージン ----
    ax = axes[1]
    shade_saturation(ax, df)

    for t in TRANSISTORS:
        name = t.replace("XM", "M")
        ax.plot(vd_mv, df[f"{name}_margin"], label=name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="tab:red", linestyle=":", linewidth=1)
    ax.set_ylabel("|Vds| - |Vdsat| [V]")
    ax.set_xlabel("Vinp - Vinn [mV]")
    ax.set_title(
        f"Margin at operating point: min {res['op_min_margin_V']:.3f} V "
        f"({res['op_bottleneck']})",
        fontsize=9
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = os.path.join(OUT_DIR, f"5tota_headroom_vb_{vb:.2f}_vcm_{vcm:.2f}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_swing_summary(results):
    """
    代表点ごとの出力振幅を縦棒で並べ、動作点と余裕を書き込む
    """
    fig, ax = plt.subplots(figsize=(1.6 * len(results) + 2, 5.5))

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
            f"gain {res['dc_gain_dB_at_Vout_eq_Vcm']:.1f} dB",
            xy=(i, res["Vout_swing_min_V"]),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8
        )

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(
        [f"Vb = {r['Vb_V']:.2f}\nVcm = {r['Vcm_V']:.2f}" for r in results],
        fontsize=8
    )
    ax.set_xlim(-0.6, len(results) - 0.4)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.5)
    ax.set_ylabel("Vout [V]")
    ax.set_title(
        "5T OTA open-loop output swing (bar: all saturated, dot: Vinp = Vinn)\n"
        f"W = {W_UM:g} um, L = {L_UM:g} um"
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

    print(f"Target Schematic: {SCH_PATH}")

    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)

    if not (spice_template and os.path.exists(spice_template)):
        raise SystemExit("Failed to generate base netlist. Please check xschem execution.")

    with open(spice_template, "r") as f:
        template_content = f.read()

    # 利得の検算用に AC 解析の結果を読む (なくても続行する)
    ac_df = None

    if os.path.exists(SAT_CSV_PATH):
        ac_df = pd.read_csv(SAT_CSV_PATH, usecols=["W_um", "L_um", "Vb_V", "Vin_V", "gain_dc_dB"])

        for col in ["W_um", "L_um", "Vb_V", "Vin_V"]:
            ac_df[col] = ac_df[col].round(3)

        ac_df = ac_df[(ac_df["W_um"] == W_UM) & (ac_df["L_um"] == L_UM)]

    os.makedirs(OUT_DIR, exist_ok=True)

    # 作業用ファイルはコンテナ内の一時フォルダに作る (環境変数 WORK_DIR で親フォルダを変更可能)
    work_dir = tempfile.mkdtemp(prefix="5tota_dc_", dir=os.environ.get("WORK_DIR"))

    try:
        num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            sims = list(executor.map(
                lambda p: simulate_point(template_content, work_dir, *p),
                POINTS
            ))

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    results = []

    for (vb, vcm), df in zip(POINTS, sims):

        if df is None:
            continue

        df = add_saturation(df)
        res = analyze(df, vb, vcm, ac_df)

        if res is None:
            print(f"WARNING: Vb = {vb:.2f}, Vcm = {vcm:.2f} に全トランジスタ飽和の点がありません")
            continue

        plot_point(df, res)
        results.append(res)

    if not results:
        raise SystemExit("解析できる点がありませんでした")

    plot_swing_summary(results)

    pd.DataFrame(results).to_csv(SUMMARY_PATH, index=False)
    print(f"Saved: {SUMMARY_PATH}")

    # 結果を表示
    print()
    print("Headroom summary:")

    for res in results:
        ac = res["ac_gain_dB"]
        ac_text = f"{ac:.2f} dB" if ac is not None else "n/a"

        print(
            f"  Vb = {res['Vb_V']:.2f}, Vcm = {res['Vcm_V']:.2f}: "
            f"Vout {res['Vout_swing_min_V']:.3f} ~ {res['Vout_swing_max_V']:.3f} V "
            f"(swing {res['Vout_swing_V']:.3f} V, limited by "
            f"{res['limit_at_min']} / {res['limit_at_max']}), "
            f"op margin {res['op_min_margin_V']:.3f} V ({res['op_bottleneck']}"
            f"{'' if res['op_all_sat'] else ', NOT all saturated'}), "
            f"offset {res['offset_V'] * 1e3:.1f} mV, "
            f"gain DC {res['dc_gain_dB_at_Vout_eq_Vcm']:.2f} dB / AC {ac_text}"
        )
