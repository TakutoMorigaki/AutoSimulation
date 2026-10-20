import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from OPAMP import STUDY_DIR


# ============================================================
# 能動学習 (active) とランダムサンプリング (random) を、
# 「既存グリッドでの accuracy が目標に届くまでのシミュレーション回数と時間」で比べる
#
#   読むもの (active_sim_opamp.py の出力):
#     active : study/opamp_active_sim_w2p5/, study/opamp_active_sim_w2p5_seed<N>/
#     random : study/opamp_active_sim_w2p5_random/, study/opamp_active_sim_w2p5_random_seed<N>/
#
#   時間の数え方 (実際に使うときにかかる時間で比べる):
#     active : 経過時間そのもの (毎ラウンドの学習も点選びに必要なので含める)
#     random : それまでのシミュレーション時間の合計 + 最後に1回だけ学習する時間
#              (random は途中の学習が要らないので、途中ラウンドの学習時間は含めない)
#
#   出力 (study/opamp_active_vs_random/):
#     reach.csv   : 実行ごと・目標ごとの 届いた回数と時間 (届かなければ空欄)
#     curves.png  : 回数 vs 誤り率 / 時間 vs 誤り率
#
#   Windows 側で実行する想定 (図に日本語を使う)。
# ============================================================


BASE_NAME = "opamp_active_sim_w2p5"

OUT_DIR = os.path.join(STUDY_DIR, "opamp_active_vs_random")

# 目標の accuracy (0.9967 = 全データ MLP / active 5 回の平均と同程度)
TARGETS = [0.99, 0.995, 0.996, 0.9967]


def find_runs():
    """
    [(方法, 種, rounds の DataFrame)]
    """

    runs = []
    pattern = re.compile(rf"{BASE_NAME}(_random)?(?:_seed(\d+))?")

    for d in sorted(glob.glob(os.path.join(STUDY_DIR, BASE_NAME + "*"))):
        m = pattern.fullmatch(os.path.basename(d))
        path = os.path.join(d, "rounds.csv")
        if m is None or not os.path.exists(path):
            continue

        strategy = "random" if m.group(1) else "active"
        seed = int(m.group(2) or 0)
        r = pd.read_csv(path)

        # 時間の列を足す (途中から再開した実行では、最初のラウンドの sim_s が空になる)
        sim = r["sim_s"].fillna(0).cumsum()
        r["time_s"] = (r["elapsed_s"] if strategy == "active"
                       else sim + r["fit_s"])
        r["test_err"] = 1 - r["test_acc"]

        runs.append((strategy, seed, r))

    return runs


def reach_table(runs):
    """
    実行ごと・目標ごとに、初めて目標に届いたラウンドの回数と時間
    """

    rows = []
    for strategy, seed, r in runs:
        for t in TARGETS:
            hit = r[r["test_acc"] >= t]
            row = {"strategy": strategy, "seed": seed, "target": t,
                   "n_sim": np.nan, "time_min": np.nan}
            if len(hit):
                row["n_sim"] = int(hit["n_labeled"].iloc[0])
                row["time_min"] = hit["time_s"].iloc[0] / 60
            rows.append(row)

        # 最後のラウンドの値 (届かなかった目標の参考)
        last = r.iloc[-1]
        print(f"{strategy} seed {seed}: 最後 n={int(last['n_labeled'])}, "
              f"accuracy={last['test_acc']:.4f}, "
              f"時間 {last['time_s'] / 60:.1f} 分")

    return pd.DataFrame(rows)


def plot_curves(runs, path):
    """
    左: 回数 vs 誤り率  右: 時間 vs 誤り率 (破線: 目標の誤り率)
    """

    colors = {"active": "#d62728", "random": "#7f7f7f"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    labeled = set()
    for strategy, seed, r in runs:
        label = strategy if strategy not in labeled else None
        labeled.add(strategy)

        axes[0].plot(r["n_labeled"], r["test_err"], color=colors[strategy],
                     marker="o", ms=2, lw=1, alpha=0.8, label=label)
        axes[1].plot(r["time_s"] / 60, r["test_err"], color=colors[strategy],
                     marker="o", ms=2, lw=1, alpha=0.8, label=label)

    for ax in axes:
        for t in TARGETS:
            ax.axhline(1 - t, color="k", ls=":", lw=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    axes[0].set_xlabel("シミュレーション回数")
    axes[0].set_ylabel("既存グリッドでの誤り率")
    axes[0].set_title("回数 vs 誤り率 (点線: 目標)")
    axes[1].set_xlabel("時間 [分] (random は シミュレーション合計 + 最後の学習)")
    axes[1].set_title("時間 vs 誤り率")

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

    reach = reach_table(runs)
    reach.to_csv(os.path.join(OUT_DIR, "reach.csv"), index=False)

    plot_curves(runs, os.path.join(OUT_DIR, "curves.png"))

    # 方法・目標ごとの平均 (届いた実行だけで平均。reached = 届いた実行の数)
    pd.set_option("display.width", 200)
    print("\n目標の accuracy に届いた回数と時間 (届いた実行の平均):")
    print(reach.groupby(["target", "strategy"]).agg(
        runs=("seed", "size"),
        reached=("n_sim", "count"),
        n_sim=("n_sim", "mean"),
        time_min=("time_min", "mean"),
    ).round(1).to_string())

    print(f"\n保存先: {OUT_DIR}")


if __name__ == "__main__":
    main()
