import os

import plot_active_sim_opamp as base
import active_sim_opamp_svm as svm


# ============================================================
# active_sim_opamp_svm.py (SVM での能動学習) の結果をプロットする
# (plot_active_sim_opamp.py の SVM 版)
#
#   図の種類・描き方は plot_active_sim_opamp.py をそのまま使い、
#   読むファイル、保存先、予測に使うモデル、図に書くモデル名だけを差し替える。
#   SVM の「確率」は、決定関数 f(x) をシグモイドで 0～1 にした値
#   (active_sim_opamp_svm.py と同じ。0.5 が境界)。
#
#   読むもの (study/opamp_active_sim_svm/):
#     rounds.csv, samples.csv, models/ensemble.joblib
#   評価に使う既存グリッド: study/opamp_results.csv
#
#   出力 (study/opamp_active_sim_svm/plots/): MLP 版と同じ構成
#     rounds.png, sample_hist.png, grid_error_by_param.png,
#     errors_per_design.csv, map_XX_*.png
#
#   Windows 側で実行する想定 (図に日本語を使う)。
#   既存グリッド (約49万行) の予測は SVM だと重いので、SVM_JOBS の数だけ並列に計算する。
# ============================================================


def use_svm():
    """
    plot_active_sim_opamp.py が読む名前を SVM 版のものに差し替える
    (関数は実行するときにこれらの名前を読みに行くので、差し替えた後に base.main() を呼ぶ)
    """

    base.OUT_DIR = svm.OUT_DIR
    base.SAMPLES_PATH = svm.SAMPLES_PATH
    base.ROUNDS_PATH = svm.ROUNDS_PATH
    base.MODEL_DIR = svm.MODEL_DIR
    base.PLOT_DIR = os.path.join(svm.OUT_DIR, "plots")

    base.predict_ensemble = svm.predict_svm
    base.MODEL_NAME = "SVM"


if __name__ == "__main__":
    use_svm()
    base.main()
