import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from active_sim_opamp import (
    OUT_DIR,
    SAMPLES_PATH,
    ROUNDS_PATH,
    MODEL_DIR,
    TEST_CSV,
    PARAM_SPACE,
    PARAM_KEYS,
    LABEL,
    TARGET_ERR,
    to_features,
    predict_ensemble,
)
from OPAMP import DESIGN_KEYS


# ============================================================
# active_sim_opamp.py (ngspice を回しながらの能動学習) の結果をプロットする
#
#   読むもの (study/opamp_active_sim/):
#     rounds.csv, samples.csv, models/ensemble.joblib
#   評価に使う既存グリッド: study/opamp_results.csv
#
#   出力 (study/opamp_active_sim/plots/):
#     rounds.png               : ラウンドごとの change_rate / est_err / グリッドでの誤り率、
#                                飽和率、1ラウンドの時間
#     sample_hist.png          : シミュレーションした点の分布 (active / random 別)
#     grid_error_by_param.png  : 既存グリッドでの誤り率をパラメータの値ごとに集計
#     errors_per_design.csv    : 既存グリッドの設計ごとの誤り数
#     map_XX_*.png             : 誤りの多い設計の Vin-Ibias マップ
#                                (正解 + MLP の境界 / TP・TN・FP・FN)
#
#   Windows 側で実行する想定 (図に日本語を使う)。
# ============================================================


# ============================================================
# 設定
# ============================================================

PLOT_DIR = os.path.join(OUT_DIR, "plots")

# 既存グリッドのうち、OPAMP.py で振った元の範囲 (図に破線で描く)
GRID_RANGE = {
    "Win_um": (5.6, 44.8),
    "Wload_um": (5.6, 44.8),
    "Wtail_um": (5.6, 44.8),
    "W6_um": (5.6, 22.4),
    "W7_um": (5.6, 22.4),
    "Wbp_um": (5.6, 22.4),
    "Wbn_um": (5.6, 22.4),
    "L_um": (0.28, 2.24),
    "Ibias_uA": (1.0, 45.0),
    "Vin_V": (0.1, 3.3),
}

# マップを描く設計の数 (誤りの多い順)
MAX_MAPS = 12

# マップ上で MLP の境界を描くための細かい格子の点数 (Vin × Ibias)
MAP_RES = 200

ROLE_COLORS = {"active": "#d62728", "random": "#7f7f7f"}


# ============================================================
# データ
# ============================================================

def load_grid(models):
    """
    既存グリッドを読み、MLP の予測と誤りを足す。
    Vin が PARAM_SPACE の範囲外の行 (一部の設計だけにある) は除く
    """

    df = pd.read_csv(
        TEST_CSV,
        usecols=PARAM_KEYS + [LABEL],
        dtype={**{k: np.float32 for k in PARAM_KEYS}, LABEL: np.int8},
    )

    inside = np.ones(len(df), dtype=bool)
    for k, (lo, hi, _, _) in PARAM_SPACE.items():
        inside &= (df[k] >= lo - 1e-6) & (df[k] <= hi + 1e-6)
    df = df[inside].reset_index(drop=True)

    print(f"既存グリッド {len(df)} 行を予測中 ...", flush=True)
    prob = predict_ensemble(models, to_features(df[PARAM_KEYS].to_numpy()))

    df["prob"] = prob
    df["pred"] = (prob >= 0.5).astype(np.int8)
    df["err"] = df["pred"] != df[LABEL]

    return df


# ============================================================
# 図
# ============================================================

def plot_rounds(rounds, path):
    """
    左: change_rate / est_err / グリッドでの誤り率
    中: 学習に使った点の飽和率
    右: 1ラウンドの時間 (シミュレーション / 学習)
    """

    n = rounds["n_labeled"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(n, rounds["change_rate"], marker="o", ms=3, color="#d62728",
            label="change_rate (停止判定)")
    ax.plot(n, rounds["est_err"], marker="s", ms=3, color="#1f77b4",
            label="est_err (ランダム点での誤り率)")
    if rounds["test_acc"].notna().any():
        ax.plot(n, 1 - rounds["test_acc"], ls="--", color="k",
                label="既存グリッドでの誤り率")
    ax.axhline(TARGET_ERR, color="gray", ls=":",
               label=f"停止しきい値 {TARGET_ERR}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("シミュレーション回数")
    ax.set_title("停止判定の指標と誤り率")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(n, rounds["sat_ratio_labeled"], marker="o", ms=3, color="#2ca02c")
    ax.set_xscale("log")
    ax.set_xlabel("シミュレーション回数")
    ax.set_ylabel("飽和 (all_sat = 1) の割合")
    ax.set_title("シミュレーションした点の飽和率")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(n, rounds["sim_s"], marker="o", ms=3, color="#9467bd",
            label="シミュレーション")
    ax.plot(n, rounds["fit_s"], marker="s", ms=3, color="#ff7f0e",
            label="MLP の学習")
    ax.set_xscale("log")
    ax.set_xlabel("シミュレーション回数")
    ax.set_ylabel("時間 [s]")
    ax.set_title("1ラウンドの時間")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_hist(samples, path):
    """
    シミュレーションした点の分布 (パラメータごと、active / random 別)。
    W, L, Ibias は log 軸。破線は OPAMP.py のグリッドの範囲
    """

    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))

    for ax, k in zip(axes.ravel(), PARAM_KEYS):
        lo, hi, log, _ = PARAM_SPACE[k]
        bins = (np.geomspace(lo, hi, 41) if log
                else np.linspace(lo, hi, 41))

        for role, color in ROLE_COLORS.items():
            v = samples.loc[samples["role"] == role, k]
            ax.hist(v, bins=bins, color=color, alpha=0.5,
                    label=f"{role} ({len(v)})")

        for x in GRID_RANGE[k]:
            ax.axvline(x, color="k", ls="--", lw=0.8)

        if log:
            ax.set_xscale("log")
        ax.set_title(k, fontsize=10)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("シミュレーションした点の分布 (破線: 既存グリッドの範囲)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_by_param(grid, path):
    """
    既存グリッドでの誤り率を、各パラメータの値ごとに集計する
    """

    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), sharey=True)

    for ax, k in zip(axes.ravel(), PARAM_KEYS):
        e = grid.groupby(k)["err"].mean()

        if len(e) <= 6:
            ax.bar([f"{v:g}" for v in e.index], e.values, color="#1f77b4")
        else:
            ax.plot(e.index, e.values, marker="o", ms=2, color="#1f77b4")

        ax.axhline(grid["err"].mean(), color="k", ls=":", lw=0.8)
        ax.set_title(k, fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    for ax in axes[:, 0]:
        ax.set_ylabel("誤り率")

    fig.suptitle(
        f"既存グリッドでの誤り率 (パラメータの値ごと, 点線: 全体 "
        f"{grid['err'].mean():.4f})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_design_map(grid, models, design, path):
    """
    1つの設計について、Vin-Ibias 平面に
      左: 正解 + MLP の境界 (細かい格子で求めた確率 0.5 の等高線)
      右: TP / TN / FP / FN
    を並べる
    """

    mask = np.ones(len(grid), dtype=bool)
    for k in DESIGN_KEYS:
        mask &= np.isclose(grid[k].to_numpy(), design[k])
    sub = grid[mask]

    vin = np.sort(sub["Vin_V"].unique())
    ib = np.sort(sub["Ibias_uA"].unique())
    xi = np.searchsorted(vin, sub["Vin_V"].to_numpy())
    yi = np.searchsorted(ib, sub["Ibias_uA"].to_numpy())

    def to_grid(values):
        g = np.full((len(ib), len(vin)), np.nan)
        g[yi, xi] = values
        return g

    y = sub[LABEL].to_numpy()
    pred = sub["pred"].to_numpy()
    cls = np.where(pred == y, y, np.where(pred == 1, 2, 3))

    # MLP の境界を細かい格子で求める (グリッドの間も連続的に予測できるため)
    fv = np.linspace(vin.min(), vin.max(), MAP_RES)
    fi = np.geomspace(ib.min(), ib.max(), MAP_RES)
    FV, FI = np.meshgrid(fv, fi)
    P = np.column_stack(
        [np.full(FV.size, design[k]) for k in DESIGN_KEYS]
        + [FI.ravel(), FV.ravel()]
    )
    fine = predict_ensemble(models, to_features(P)).reshape(FV.shape)

    # 0: TN, 1: TP, 2: FP, 3: FN
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#dddddd", "#4c9be8", "#f28e2b", "#d62728"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    axes[0].pcolormesh(vin, ib, to_grid(y), cmap=cmap, vmin=-0.5,
                       vmax=3.5, shading="nearest")
    axes[0].contour(FV, FI, fine, levels=[0.5], colors="k", linewidths=1.0)
    axes[0].set_title("正解 (色) と MLP の境界 (黒線)")

    axes[1].pcolormesh(vin, ib, to_grid(cls), cmap=cmap, vmin=-0.5,
                       vmax=3.5, shading="nearest")
    axes[1].contour(vin, ib, to_grid(y), levels=[0.5], colors="k",
                    linewidths=0.8)
    axes[1].set_title(f"予測 (誤り {int((pred != y).sum())} / {len(y)})")

    for ax in axes:
        ax.set_xlabel("Vin [V]")
    axes[0].set_ylabel("Ibias [uA]")

    handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=lab)
        for i, lab in enumerate(["非飽和", "飽和", "FP (誤って飽和)",
                                 "FN (見逃し)"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        " ".join(f"{k[:-3]}={design[k]:g}" for k in DESIGN_KEYS) + " [um]",
        fontsize=10
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():

    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    rounds = pd.read_csv(ROUNDS_PATH)
    samples = pd.read_csv(SAMPLES_PATH).dropna(subset=PARAM_KEYS + [LABEL])

    print(f"ラウンド {len(rounds)} / シミュレーション {len(samples)} 点")

    plot_rounds(rounds, os.path.join(PLOT_DIR, "rounds.png"))
    plot_sample_hist(samples, os.path.join(PLOT_DIR, "sample_hist.png"))

    # 既存グリッドでの評価 (最後に学習した MLP を使う)
    model_path = os.path.join(MODEL_DIR, "ensemble.joblib")
    if not os.path.exists(model_path) or not os.path.exists(TEST_CSV):
        print("モデルか既存グリッドが無いので、グリッドでの評価は省略します")
        return

    # コンテナと Windows で scikit-learn の版が違うと警告が出るが、予測はできる
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = joblib.load(model_path)

    grid = load_grid(models)
    print(f"既存グリッドでの誤り率: {grid['err'].mean():.4f}")

    plot_error_by_param(
        grid, os.path.join(PLOT_DIR, "grid_error_by_param.png")
    )

    per_design = (
        grid.groupby(DESIGN_KEYS)
        .agg(n=("err", "size"), n_err=("err", "sum"),
             sat_ratio=(LABEL, "mean"))
        .reset_index()
        .sort_values("n_err", ascending=False)
    )
    per_design.to_csv(
        os.path.join(PLOT_DIR, "errors_per_design.csv"), index=False
    )

    for i, (_, d) in enumerate(per_design.head(MAX_MAPS).iterrows(), 1):
        name = "_".join(f"{k[:-3].lower()}_{d[k]:g}" for k in DESIGN_KEYS)
        plot_design_map(
            grid, models, d,
            os.path.join(PLOT_DIR, f"map_{i:02d}_{name}.png")
        )

    print(f"保存先: {PLOT_DIR}")


if __name__ == "__main__":
    main()
