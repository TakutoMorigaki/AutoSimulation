import os
import subprocess
import re
import numpy as np
import csv
import time
import shutil
import itertools
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 2段オペアンプ (opamp.sch) の飽和シミュレーション
#
#   1段目 : PMOS 差動対 M1, M2 / NMOS カレントミラー負荷 M3, M4 / PMOS テール M5
#   2段目 : NMOS ソース接地 M7 / PMOS 電流源 M6
#   バイアス : M8 (ibias を受けるダイオード接続), M9～M11 (M12 のゲート電圧を作る)
#   位相補償 : M12 (抵抗として使う NMOS) + C1
#
#   テストベンチ tb_ac_opamp.sch は 5TOTA と同じく、
#   出力を 100MΩ で vinn に戻した DC ボルテージフォロワ (AC では開ループ)。
#
#   掃引:
#     W   : 7種類を独立に振る (M12 は回路図の値のまま)
#             Win   : PMOS 差動対 M1, M2
#             Wload : NMOS 負荷ミラー M3, M4
#             Wtail : PMOS テール M5
#             W6    : 2段目 PMOS 電流源 M6
#             W7    : 2段目 NMOS ソース接地 M7
#             Wbp   : バイアス PMOS M8, M9 (M8 はミラーの基準)
#             Wbn   : バイアス NMOS M10, M11
#     L   : M1～M12 で共通
#     Ibias, Vin (同相電圧)
#
#   M12 は DC 電流が流れず線形領域で使う素子なので、all_sat には含めない。
#
#   点数が多いので、設計 (W, L の組) ごとに計算して CSV に追記する。
#   終わった設計は opamp_results.done に記録し、途中で止めても続きから再開できる。
#   (再開時は、最後に記録した位置より後ろの書きかけの行を CSV から消してから続ける)
# ============================================================

# パスの定義（sourceフォルダから見た相対パス）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

STUDY_DIR = os.path.join(PROJECT_ROOT, "study")
NETLIST_DIR = os.path.join(STUDY_DIR, "netlists") # 自動生成されるネットリストの保存先

# 対象のxschem回路図ファイル名
SCH_FILENAME = "tb_ac_opamp.sch"
SCH_PATH = os.path.join(STUDY_DIR, SCH_FILENAME)

env = os.environ.copy()



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

    found = False

    for i, line in enumerate(lines):

        # 対象トランジスタの先頭行だけを検索
        if re.search(
            rf'^\s*{re.escape(transistor)}\b',
            line,
            flags=re.IGNORECASE
        ):
            new_line, count = pattern.subn(
                rf'\g<1>{value}u',
                line
            )

            if count > 0:
                lines[i] = new_line
                found = True
            else:
                print(
                    f"WARNING: {transistor} の "
                    f"{parameter} が見つかりません"
                )

            break

    if not found:
        print(
            f"WARNING: {transistor} が "
            f"ネットリスト内に見つかりません"
        )

    return "\n".join(lines)

def modify_netlist_params(netlist_path, **kwargs):
    """ネットリストファイル内の電源電圧などのパラメータを書き換える (引数は modify_netlist_content と同じ)"""
    with open(netlist_path, 'r') as f:
        content = f.read()

    content = modify_netlist_content(content, **kwargs)

    with open(netlist_path, 'w') as f:
        f.write(content)

def modify_netlist_content(
        content,
        vdd_val=None,
        ibias_val=None,
        vin_val=None,
        sizes=None
    ):
    """
    ネットリストの文字列内の電源電圧などのパラメータを書き換えて返す。
    (ファイルを介さないので、並列実行時のファイル入出力を減らせる)

    sizes: {"Win_um": 11.2, ..., "L_um": 0.56} のような W, L の辞書
           (キーは SIZE_GROUPS と "L_um")
    """

    # modify vdd
    if vdd_val is not None:
        # \g<1> を使ってグループ1と数値の結合ミス（グループ10の誤認）を防ぐ
        content = re.sub(
            r'(\.param\s+vdd\s*=\s*)[\d\.]+',
            rf'\g<1>{vdd_val}',
            content,
            flags=re.IGNORECASE
        )

    # modify ibias [uA] (回路図では "10u" のように単位付きで書かれているので、値全体を置き換える)
    if ibias_val is not None:
        content = re.sub(
            r'(\.param\s+ibias\s*=\s*)\S+',
            rf'\g<1>{ibias_val}u',
            content,
            flags=re.IGNORECASE
        )

    # modify vin (入力の直流電圧 = 同相電圧)
    if vin_val is not None:
        content = re.sub(
            r'(\.param\s+vin\s*=\s*)[\d\.]+',
            rf'\g<1>{vin_val}',
            content,
            flags=re.IGNORECASE
        )

    if sizes is None:
        return content

    # modify W (グループごとに同じ値を入れる)
    for key, transistors in SIZE_GROUPS.items():
        if sizes.get(key) is None:
            continue
        for transistor in transistors:
            content = replace_mos_parameter(content, transistor, "W", sizes[key])

    # modify L
    if sizes.get("L_um") is not None:
        for transistor in ALL_TRANSISTORS:
            content = replace_mos_parameter(content, transistor, "L", sizes["L_um"])

    return content

def run_ngspice(netlist_path):
    """Ngspiceをバッチモードで実行する"""
    cmd = ["ngspice", "-b", netlist_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Ngspice Error:", result.stderr)
        return False, result.stdout
    return True, result.stdout

def extract_measure_results(stdout_text):
    results = {}

    pattern = r'([a-zA-Z0-9_]+)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'

    matches = re.findall(pattern, stdout_text)
    for name, value in matches:
        results[name] = float(value)

    return results

def device_to_transistor(device):
    """
    m.x1.xm1.m0 などのデバイス名を XM1 に正規化する。
    MOS以外 (vin, vdd などの電圧源) は None を返す。
    """
    match = re.search(r'(xm\d+)', device, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    return None

def extract_device_parameters(stdout_text):
    """
    ngspiceの show 出力 (トランジスタが数個ずつ横並びのブロック) から
    各トランジスタの id, vds, vdsat, vgs, vth を取得する
    """
    data = {}
    current_devices = []

    for line in stdout_text.splitlines():

        tokens = line.split()

        if not tokens:
            continue

        # device行: 以降の値がどのトランジスタのものかを切り替える
        if tokens[0].lower() == "device":

            current_devices = tokens[1:]

            for device in current_devices:
                transistor = device_to_transistor(device)
                if transistor is not None and transistor not in data:
                    data[transistor] = {}

        # 各パラメータ行
        elif current_devices:

            parameter = tokens[0].lower()

            if parameter not in DEVICE_PARAMS:
                continue

            values = tokens[1:]

            if len(values) != len(current_devices):
                continue

            for device, value in zip(current_devices, values):

                transistor = device_to_transistor(device)

                if transistor is None:
                    continue

                try:
                    data[transistor][parameter] = float(value)
                except ValueError:
                    pass

    return data

DEVICE_PARAMS = ["id", "vds", "vdsat", "vgs", "vth"]

# 回路内の全トランジスタ (L はすべてに同じ値を入れる)
ALL_TRANSISTORS = [f"XM{i}" for i in range(1, 13)]

# all_sat の判定に含めないトランジスタ (M12 は位相補償の抵抗として線形領域で使う)
SAT_EXCLUDE = ["M12"]

# W を振るグループ (CSV の列名 → 同じ W を入れるトランジスタ)
SIZE_GROUPS = {
    "Win_um": ["XM1", "XM2"],      # PMOS 差動対
    "Wload_um": ["XM3", "XM4"],    # NMOS 負荷ミラー
    "Wtail_um": ["XM5"],           # PMOS テール
    "W6_um": ["XM6"],              # 2段目 PMOS 電流源
    "W7_um": ["XM7"],              # 2段目 NMOS ソース接地
    "Wbp_um": ["XM8", "XM9"],      # バイアス PMOS
    "Wbn_um": ["XM10", "XM11"],    # バイアス NMOS
}

# 設計を区別する列 (CSV の先頭に並べる)
DESIGN_KEYS = list(SIZE_GROUPS) + ["L_um"]

# CSV の列 (設計 → 掃引 → ゲイン → 各トランジスタ → all_sat)
FIELDNAMES = (
    DESIGN_KEYS
    + ["Ibias_uA", "Vin_V", "gain_dc_dB"]
    + [
        f"M{i}_{p}"
        for i in range(1, 13)
        for p in DEVICE_PARAMS + ["sat"]
    ]
    + ["all_sat"]
)

def is_saturated(params):
    """
    |Vgs| > |Vth| かつ |Vds| >= |Vdsat| なら飽和とみなす。
    (PMOSは負の値で出力されることがあるため絶対値で比較)
    """
    if not all(p in params for p in ["vds", "vdsat", "vgs", "vth"]):
        return None

    on = abs(params["vgs"]) > abs(params["vth"])
    sat = abs(params["vds"]) >= abs(params["vdsat"])

    return on and sat

def simulate_point(template_content, work_dir, index, sizes, ibias, vin):
    """
    1つの (W, L, Ibias, Vin) についてネットリスト作成 → ngspice実行 → 解析を行う。

    作業用ネットリストは work_dir (コンテナ内の一時フォルダ) に作る。
    Windows側のマウントフォルダに作ると、ファイル入出力の待ちで遅くなるため。

    戻り値: (row または None, 失敗時のメッセージ または None)
    """
    label = (
        ", ".join(f"{k[:-3]} = {v}u" for k, v in sizes.items())
        + f", Ibias = {ibias:g}uA, Vin = {vin:.2f}V"
    )

    # 条件ごとに番号が異なるので並列でも衝突しない
    work_spice = os.path.join(work_dir, f"opamp_{index}.spice")

    try:
        # パラメータをメモリ上で書き換えてから、1回だけ書き込む
        content = modify_netlist_content(
            template_content, ibias_val=ibias, vin_val=vin, sizes=sizes
        )

        with open(work_spice, 'w') as dst:
            dst.write(content)

        # シミュレーション実行
        success, stdout = run_ngspice(work_spice)

        if not success:
            return None, f"{label}: Simulation Failed"

        meas_results = extract_measure_results(stdout)
        data = extract_device_parameters(stdout)

        if 'gain_dc' not in meas_results:
            return None, f"{label}: gain_dc が取得できませんでした"

        row = {
            **sizes,
            "Ibias_uA": ibias,
            "Vin_V": vin,
            "gain_dc_dB": meas_results['gain_dc'],
        }

        # 各トランジスタの飽和判定
        all_sat = True

        for transistor in sorted(data):
            params = data[transistor]
            sat = is_saturated(params)
            name = transistor.replace("XM", "M")

            for p in DEVICE_PARAMS:
                row[f"{name}_{p}"] = params.get(p)

            row[f"{name}_sat"] = None if sat is None else int(sat)

            if not sat and name not in SAT_EXCLUDE:
                all_sat = False

        row["all_sat"] = int(bool(data) and all_sat)

        return row, None

    finally:
        if os.path.exists(work_spice):
            os.remove(work_spice)

def design_key(sizes):
    """終了記録ファイルに書く設計の識別子"""
    return ",".join(str(sizes[k]) for k in DESIGN_KEYS)

def load_progress(csv_path, done_path):
    """
    終了記録ファイル (1行 = 設計の識別子 + タブ + その設計を書き終えた時点の CSV のバイト数) を読み、
    終わった設計の集合を返す。
    CSV は最後に記録した位置で切り詰め、書きかけの設計の行を消す。
    """
    done = set()
    end = None

    with open(done_path) as f:
        for line in f:
            if "\t" not in line:
                continue
            key, offset = line.rstrip("\n").split("\t")
            done.add(key)
            end = int(offset)

    # 1つも終わっていなければヘッダの直後まで戻す
    if end is None:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    else:
        with open(csv_path, "r+b") as f:
            f.truncate(end)

    return done

if __name__ == "__main__":
    print(f"Working Directory: {BASE_DIR}")
    print(f"Target Schematic: {SCH_PATH}")

    # 1. 基準となるネットリストを study/netlists/ に書き出し
    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)

    if spice_template and os.path.exists(spice_template):
        print(f"Base netlist generated: {spice_template}")

        # W [um] (グループごとに独立に振る)
        size_lists = {
            "Win_um": [5.6, 11.2, 22.4, 44.8],     # PMOS 差動対 M1, M2
            "Wload_um": [5.6, 11.2, 22.4, 44.8],   # NMOS 負荷ミラー M3, M4
            "Wtail_um": [5.6, 11.2, 22.4, 44.8],   # PMOS テール M5
            "W6_um": [5.6, 22.4],                  # 2段目 PMOS 電流源 M6
            "W7_um": [5.6, 22.4],                  # 2段目 NMOS ソース接地 M7
            "Wbp_um": [5.6, 22.4],                 # バイアス PMOS M8, M9
            "Wbn_um": [5.6, 22.4],                 # バイアス NMOS M10, M11
            "L_um": [0.28, 0.56, 1.12, 2.24],      # L (M1～M12 で共通)
        }

        # VDD は回路図の設定値 (.param vdd=3.3) をそのまま使うので、Vin はその範囲内
        # Ibias [uA] (M8 に流す基準電流。テール M5 と 2段目 M6 にミラーされる)
        ibias_sweep = [round(float(v), 2) for v in np.arange(1, 46, 1)]
        vin_sweep = [round(float(v), 2) for v in np.arange(0.1, 3.4, 0.1)]

        # 並列数 (環境変数 NUM_WORKERS で変更可能。既定はCPUコア数)
        num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))

        # テンプレートは一度だけ読み込む
        with open(spice_template, 'r') as src:
            template_content = src.read()

        # 全設計の組み合わせ (CSV と同じ順)
        designs = [
            dict(zip(DESIGN_KEYS, values))
            for values in itertools.product(
                *(size_lists[k] for k in DESIGN_KEYS)
            )
        ]

        csv_path = os.path.join(STUDY_DIR, "opamp_results.csv")
        done_path = os.path.join(STUDY_DIR, "opamp_results.done")

        # 前回までに終わった設計 (CSV と記録ファイルの両方がある場合だけ再開する)
        done = set()
        if os.path.exists(csv_path) and os.path.exists(done_path):
            done = load_progress(csv_path, done_path)
        else:
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
            open(done_path, "w").close()

        todo = [d for d in designs if design_key(d) not in done]
        points_per_design = len(ibias_sweep) * len(vin_sweep)

        # 作業用ネットリストはコンテナ内の一時フォルダ (/tmp/opamp_xxxx) に作る
        # (環境変数 WORK_DIR で親フォルダを変更可能)
        work_dir = tempfile.mkdtemp(prefix="opamp_", dir=os.environ.get("WORK_DIR"))

        print(
            f"Designs: {len(designs)} (done {len(designs) - len(todo)}, "
            f"remaining {len(todo)}), points/design: {points_per_design}, "
            f"workers: {num_workers}"
        )
        print(f"Work directory: {work_dir}")

        # 計測開始
        start_time = time.perf_counter()
        n_rows = 0
        n_sat = 0
        n_fail = 0

        try:
            # ngspiceは別プロセスで動くので、スレッドでも並列に実行される
            with ThreadPoolExecutor(max_workers=num_workers) as executor:

                for d_idx, sizes in enumerate(todo, 1):

                    futures = [
                        executor.submit(
                            simulate_point, template_content, work_dir,
                            i, sizes, ibias, vin
                        )
                        for i, (ibias, vin) in enumerate(
                            itertools.product(ibias_sweep, vin_sweep)
                        )
                    ]

                    rows = []
                    for future in as_completed(futures):
                        try:
                            row, err = future.result()
                        except Exception as e:
                            row, err = None, f"ERROR: {e}"

                        if row is not None:
                            rows.append(row)
                        else:
                            n_fail += 1
                            print(err)

                    # 完了順はバラバラなので、条件順に並べ直してから追記する
                    rows.sort(key=lambda r: (r["Ibias_uA"], r["Vin_V"]))

                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=FIELDNAMES).writerows(rows)
                        f.flush()
                        os.fsync(f.fileno())
                        offset = f.tell()

                    # CSV に書き終えてから、その位置と一緒に記録する
                    with open(done_path, "a") as f:
                        f.write(f"{design_key(sizes)}\t{offset}\n")

                    n_rows += len(rows)
                    n_sat += sum(r["all_sat"] for r in rows)

                    elapsed = time.perf_counter() - start_time
                    eta = elapsed / d_idx * (len(todo) - d_idx)
                    print(
                        f"[{d_idx}/{len(todo)}] {design_key(sizes)} : "
                        f"{len(rows)}/{points_per_design} ok, "
                        f"all_sat {sum(r['all_sat'] for r in rows)}, "
                        f"elapsed {elapsed:.0f} s, ETA {eta / 3600:.1f} h",
                        flush=True
                    )

        finally:
            # 途中で止めた場合も一時フォルダを消す
            shutil.rmtree(work_dir, ignore_errors=True)

        # 計測終了
        elapsed_time = time.perf_counter() - start_time

        print(f"CSV saved: {csv_path}")
        print(f"Number of successful simulations (this run): {n_rows}")
        print(f"Number of failed simulations (this run): {n_fail}")
        print(f"Number of all-saturated conditions (this run): {n_sat}")
        print(f"Total simulation time: {elapsed_time:.2f} s")

    else:
        print("Failed to generate base netlist. Please check xschem execution.")
