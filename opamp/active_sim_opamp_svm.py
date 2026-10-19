import os
import numpy as np
from joblib import Parallel, delayed

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import active_sim_opamp as base
from OPAMP import STUDY_DIR


# ============================================================
# active_sim_opamp.py (MLP での能動学習) の SVM 版
#
#   ngspice の回し方、設計空間 (PARAM_SPACE)、点の選び方、停止条件、再開の仕組みは
#   active_sim_opamp.py をそのまま使い、「モデル」と「保存先」だけを差し替える。
#   (active_sim_opamp.py の設定を変えると、こちらにも同じように効く)
#
#   MLP 版との違い:
#     - モデルは RBF カーネルの SVM 1個 (MLP のように初期値を変えたアンサンブルにはしない。
#       SVM は同じデータなら同じ結果になるため)
#     - 次に回す点のスコアは「決定関数 f(x) の絶対値が小さい = 境界に近い」点
#       (マージンサンプリング)。f(x) をシグモイドで 0～1 に変換した値を
#       「飽和する確率」の代わりに使うので、MLP 版と同じエントロピーの計算で
#       |f(x)| が小さい点ほどスコアが高くなる。0.5 が境界 (f(x) = 0) になるのも同じ
#     - 予測は点数と support vector の数に比例して重いので、候補点を分けて並列に計算する
#
#   ハイパーパラメータ (SVM_C, SVM_GAMMA) は、MLP 版で集めた点 (opamp_active_sim_w2p5) の
#   うち 20000 点で学習し、ランダムに選ばれた点 3000 点での正解率が高かった値:
#     gamma=0.05, C=100 : 0.948   (gamma=0.1, C=10 : 0.939 / gamma=0.2 : 0.93 前後)
#
#   最初のランダムな N_INIT 点は MLP 版と同じ乱数で作るので、同じ点から始まる。
#
#   出力 (study/opamp_active_sim_svm/): MLP 版と同じ構成
#     samples.csv, failed.csv, rounds.csv, rounds.png, stop_pred/
#     models/ensemble.joblib : 最後に学習した SVM (1個だけのリスト。MLP 版と同じ形)
#
#   環境変数 (MLP 版と共通のもの以外):
#     SVM_JOBS : 予測の並列数 (既定は CPU コア数)
# ============================================================


# ============================================================
# 設定
# ============================================================

SVM_C = 100.0
SVM_GAMMA = 0.05

# 予測を何行ずつに分けて並列に計算するか
SVM_CHUNK = 25_000
SVM_JOBS = int(os.environ.get("SVM_JOBS", os.cpu_count() or 1))

# libsvm のカーネル値のキャッシュ [MB] (大きいほど学習が速い)
SVM_CACHE_MB = 2000


# ============================================================
# 保存先 (MLP 版の結果を上書きしないよう、別のフォルダにする)
# ============================================================

OUT_DIR = os.path.join(
    STUDY_DIR,
    "opamp_active_sim_svm"
)

SAMPLES_PATH = os.path.join(OUT_DIR, "samples.csv")
FAILED_PATH = os.path.join(OUT_DIR, "failed.csv")
ROUNDS_PATH = os.path.join(OUT_DIR, "rounds.csv")
STOP_PRED_DIR = os.path.join(OUT_DIR, "stop_pred")
MODEL_DIR = os.path.join(OUT_DIR, "models")


# ============================================================
# モデル
# ============================================================

def build_svm():
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=SVM_C,
            gamma=SVM_GAMMA,
            cache_size=SVM_CACHE_MB,
        ),
    )


def fit_svm(X, y, seed, n_models):
    """
    MLP 版の fit_ensemble と同じ呼び方で、SVM を1個学習してリストで返す
    (seed, n_models は使わない。SVM は同じデータなら同じ結果になる)
    """

    model = build_svm()
    model.fit(X, y)

    return [model]


def decision_svm(model, X):
    """
    決定関数 f(x) を SVM_CHUNK 行ずつ並列に計算する
    """

    if len(X) <= SVM_CHUNK:
        return model.decision_function(X)

    return np.concatenate(
        Parallel(n_jobs=SVM_JOBS)(
            delayed(model.decision_function)(X[i:i + SVM_CHUNK])
            for i in range(0, len(X), SVM_CHUNK)
        )
    )


def predict_svm(models, X):
    """
    MLP 版の predict_ensemble と同じ呼び方で、「飽和する確率」の代わりの値を返す。
    f(x) をシグモイドで 0～1 にしたもの (0.5 が境界、境界から遠いほど 0 か 1 に近づく)
    """

    f = decision_svm(models[0], X)

    return 1.0 / (1.0 + np.exp(-f))


# ============================================================
# active_sim_opamp.py のモデルと保存先を差し替える
#   (active_sim_opamp.py の関数は、実行するときにこれらの名前を読みに行くので、
#    差し替えた後に base.main() を呼べば SVM 版として動く)
# ============================================================

def use_svm():

    base.OUT_DIR = OUT_DIR
    base.SAMPLES_PATH = SAMPLES_PATH
    base.FAILED_PATH = FAILED_PATH
    base.ROUNDS_PATH = ROUNDS_PATH
    base.STOP_PRED_DIR = STOP_PRED_DIR
    base.MODEL_DIR = MODEL_DIR

    base.fit_ensemble = fit_svm
    base.predict_ensemble = predict_svm


if __name__ == "__main__":
    use_svm()

    print(
        f"SVM 版 (RBF, C={SVM_C}, gamma={SVM_GAMMA}, 予測の並列数 {SVM_JOBS})\n"
        f"保存先: {OUT_DIR}"
    )

    base.main()
