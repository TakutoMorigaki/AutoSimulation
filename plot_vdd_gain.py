import os
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
    "vddwl_gain_results.csv"
)

OUTPUT_DIR = os.path.join(
    STUDY_DIR,
    "vdd_gain_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CSV読み込み
# ============================================================

df = pd.read_csv(CSV_PATH)

print("CSV loaded:")
print(df.head())

print()
print("Number of data:")
print(len(df))


# ============================================================
# Wnの一覧を取得
# ============================================================

w_values = sorted(
    df["Wn_um"].dropna().unique()
)

l_values = sorted(
    df["L_um"].dropna().unique()
)

print()
print("W values:")
print(w_values)

print()
print("L values:")
print(l_values)


# ============================================================
# Wごとにグラフを作成
# ============================================================

for w in w_values:

    plt.figure(
        figsize=(9, 6)
    )

    # --------------------------------------------
    # このWだけ取り出す
    # --------------------------------------------

    w_data = df[
        df["Wn_um"] == w
    ]

    # --------------------------------------------
    # LごとにVDD-Gainを描画
    # --------------------------------------------

    for l in l_values:

        data = w_data[
            w_data["L_um"] == l
        ].copy()

        # gainが空欄のデータを除外
        data = data.dropna(
            subset=["gain_dc_dB"]
        )

        if len(data) == 0:
            continue

        # VDD順に並べる
        data = data.sort_values(
            "VDD_V"
        )

        plt.plot(
            data["VDD_V"],
            data["gain_dc_dB"],
            marker="o",
            markersize=2,
            label=f"L = {l:.2f} um"
        )


    # --------------------------------------------
    # グラフ設定
    # --------------------------------------------

    plt.xlabel(
        "VDD [V]"
    )

    plt.ylabel(
        "DC Gain [dB]"
    )

    plt.title(
        f"5T OTA Gain vs VDD "
        f"(Wn = {w:.2f} um)"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()


    # --------------------------------------------
    # 保存
    # --------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"gain_vs_vdd_W{w:.2f}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


print()
print("========================================")
print("All plots generated")
print("========================================")