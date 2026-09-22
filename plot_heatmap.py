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
    "wl_gain_results.csv"
)

OUTPUT_PATH = os.path.join(
    STUDY_DIR,
    "wl_gain_heatmap.png"
)


# ============================================================
# CSV読み込み
# ============================================================

df = pd.read_csv(CSV_PATH)

print(df)


# ============================================================
# W × L の2次元データに変換
# ============================================================

heatmap_data = df.pivot(
    index="Wn_um",
    columns="L_um",
    values="gain_dc_dB"
)

print()
print("Heatmap data:")
print(heatmap_data)


# ============================================================
# ヒートマップ作成
# ============================================================

plt.figure(figsize=(10, 7))

image = plt.imshow(
    heatmap_data,
    aspect="auto",
    origin="lower"
)

# カラーバー
cbar = plt.colorbar(image)

cbar.set_label(
    "DC Gain [dB]"
)

# 軸
plt.xticks(
    range(len(heatmap_data.columns)),
    heatmap_data.columns
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.xlabel(
    "L [um]"
)

plt.ylabel(
    "Wn (M3/M4) [um]"
)

plt.title(
    "5T OTA DC Gain vs W and L"
)

# 各セルに数値を表示
for i in range(len(heatmap_data.index)):

    for j in range(len(heatmap_data.columns)):

        value = heatmap_data.iloc[i, j]

        if pd.notna(value):

            plt.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center"
            )


plt.tight_layout()

# PNG保存
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print(f"Heatmap saved:")
print(OUTPUT_PATH)