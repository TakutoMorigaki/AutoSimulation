import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from OPAMP import STUDY_DIR


# ============================================================
# active_sim_opamp.py を乱数の種 (AL_SEED) を変えて回した結果をまとめ、
# 結果がどのくらいばらつくか (ランダム性に強いか) を見る
#
#   読むもの: study/opamp_active_sim_w2p5/ (種 0) と study/opamp_active_sim_w2p5_seed<N>/ の rounds.csv
#
#   出力 (study/opamp_active_sim_seeds/):
#     summary.csv  : 種ごとの停止時の値 (回数、グリッドでの accuracy、change_rate、est_err、時間)
#     curves.png   : 種ごとの グリッドでの誤り率 / change_rate と、その比 (change_rate ÷ 誤り率)
#
#   Windows 側で実行する想定 (図に日本語を使う)。
# ============================================================


BASE_NAME = "opamp_active_sim_w2p5"

OUT_DIR = os.path.join(STUDY_DIR, "opamp_active_sim_seeds")


def find_runs():
    """
    {種: rounds.csv の DataFrame}
    """

    runs = {}

    for d in sorted(glob.glob(os.path.join(STUDY_DIR, BASE_NAME + "*"))):
        name = os.path.basename(d)
        if name == BASE_NAME:
            seed = 0
        else:
            m = re.fullmatch(rf"{BASE_NAME}_seed(\d+)", name)
            if m is None:
                continue
            seed = int(m.group(1))

        path = os.path.join(d, "rounds.csv")
        if os.path.exists(path):
            runs[seed] = pd.read_csv(path)

    return dict(sorted(runs.items()))


def summarize(runs):
    """
    種ごとの最終ラウンドの値
    """

    rows = []
    for seed, r in runs.items():
        last = r.iloc[-1]
        rows.append({
            "seed": seed,
            "n_sim": int(last["n_labeled"]),
            "n_failed": int(last["n_failed"]),
            "rounds": int(last["round"]),
            "test_acc": last["test_acc"],
            "test_err": 1 - last["test_acc"],
            "change_rate": last["change_rate"],
            "est_err": last["est_err"],
            "ratio_change_to_err": last["change_rate"] / (1 - last["test_acc"]),
            "elapsed_min": last["elapsed_s"] / 60,
        })

    return pd.DataFrame(rows)


def plot_curves(runs, path):
    """
    左: 種ごとのグリッドでの誤り率 (実線) と change_rate (破線)
    右: change_rate ÷ グリッドでの誤り率
    """

    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for i, (seed, r) in enumerate(runs.items()):
        c = cmap(i % 10)
        err = 1 - r["test_acc"]

        axes[0].plot(r["n_labeled"], err, color=c, marker="o", ms=2,
                     label=f"seed {seed}")
        axes[0].plot(r["n_labeled"], r["change_rate"], color=c, ls="--",
                     lw=0.9)
        axes[1].plot(r["n_labeled"], r["change_rate"] / err, color=c,
                     marker="o", ms=2, label=f"seed {seed}")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("シミュレーション回数")
    axes[0].set_title("グリッドでの誤り率 (実線) / change_rate (破線)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    axes[1].axhline(1.0, color="gray", ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("シミュレーション回数")
    axes[1].set_title("change_rate ÷ グリッドでの誤り率")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

    runs = find_runs()
    if not runs:
        print("結果が見つかりません")
        return

    summary = summarize(runs)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    plot_curves(runs, os.path.join(OUT_DIR, "curves.png"))

    pd.set_option("display.width", 200)
    print(summary.round(5).to_string(index=False))

    cols = ["n_sim", "test_acc", "change_rate", "est_err",
            "ratio_change_to_err", "elapsed_min"]
    print(f"\n{len(summary)} 個の種の 平均 / 標準偏差 / 最小 / 最大:")
    print(summary[cols].agg(["mean", "std", "min", "max"]).round(5)
          .to_string())

    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
