import os
import subprocess
import re
import numpy as np
import csv
import time
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# パスの定義（sourceフォルダから見た相対パス）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

STUDY_DIR = os.path.join(PROJECT_ROOT, "study")
NETLIST_DIR = os.path.join(STUDY_DIR, "netlists") # 自動生成されるネットリストの保存先

# 対象のxschem回路図ファイル名
SCH_FILENAME = "tb_ac_5tota.sch"
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

        # XM1～XM5の先頭行だけを検索
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
        vb_val=None,
        vin_val=None,
        wn_val=None,
        wp_val=None,
        wtail_val=None,
        l_val=None
    ):
    """
    ネットリストの文字列内の電源電圧などのパラメータを書き換えて返す。
    (ファイルを介さないので、並列実行時のファイル入出力を減らせる)
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

    # modify vb
    if vb_val is not None:
        content = re.sub(
            r'(\.param\s+vb\s*=\s*)[\d\.]+',
            rf'\g<1>{vb_val}',
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

    # modify W (3種類のトランジスタで別々に指定する)
    #   差動対 M1, M2 (NMOS)          : wn_val
    #   カレントミラー M3, M4 (PMOS)  : wp_val
    #   テール電流源 M5 (NMOS)        : wtail_val
    if wn_val is not None:
        content = replace_mos_parameter(content, "XM1", "W", wn_val)
        content = replace_mos_parameter(content, "XM2", "W", wn_val)

    if wp_val is not None:
        content = replace_mos_parameter(content, "XM3", "W", wp_val)
        content = replace_mos_parameter(content, "XM4", "W", wp_val)

    if wtail_val is not None:
        content = replace_mos_parameter(content, "XM5", "W", wtail_val)

    # modify L
    if l_val is not None:

        for transistor in ["XM1", "XM2", "XM3", "XM4", "XM5"]:

            content = replace_mos_parameter(content, transistor, "L", l_val)

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

def simulate_point(template_content, work_dir, wn, wp, wtail, l, vb, vin):
    """
    1つの (Wn, Wp, Wtail, L, Vb, Vin) についてネットリスト作成 → ngspice実行 → 解析を行う。
    並列実行時に表示が混ざらないよう、ログは文字列で返す。

    作業用ネットリストは work_dir (コンテナ内の一時フォルダ) に作る。
    Windows側のマウントフォルダに作ると、ファイル入出力の待ちで遅くなるため。

    戻り値: (row または None, ログ文字列)
    """
    label = (
        f"Wn = {wn}u, Wp = {wp}u, Wtail = {wtail}u, L = {l}u, "
        f"Vb = {vb:.2f}, Vin = {vin:.2f}V"
    )
    log = []

    # 条件ごとにファイル名が異なるので並列でも衝突しない
    work_spice = os.path.join(
        work_dir,
        f"5tota_wn_{wn}_wp_{wp}_wt_{wtail}_l_{l}_vb_{vb:.2f}_vin_{vin:.2f}.spice"
    )

    try:
        # パラメータをメモリ上で書き換えてから、1回だけ書き込む
        content = modify_netlist_content(
            template_content, vb_val=vb, vin_val=vin,
            wn_val=wn, wp_val=wp, wtail_val=wtail, l_val=l
        )

        with open(work_spice, 'w') as dst:
            dst.write(content)

        # シミュレーション実行
        success, stdout = run_ngspice(work_spice)

        if not success:
            log.append(f"{label}: Simulation Failed")
            return None, "\n".join(log)

        log.append(f"{label}: Simulation Success")

        meas_results = extract_measure_results(stdout)
        data = extract_device_parameters(stdout)

        if 'gain_dc' not in meas_results:
            log.append(" -> WARNING: gain_dc が取得できませんでした")
            return None, "\n".join(log)

        gain_value = meas_results['gain_dc']
        log.append(f" -> gain_dc = {gain_value:.4f} dB")

        row = {
            "Wn_um": wn,
            "Wp_um": wp,
            "Wtail_um": wtail,
            "L_um": l,
            "Vb_V": vb,
            "Vin_V": vin,
            "gain_dc_dB": gain_value
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

            if not sat:
                all_sat = False

            log.append(
                f" -> {name}: vds = {params.get('vds')}V, "
                f"vdsat = {params.get('vdsat')}V, "
                f"vgs = {params.get('vgs')}V, "
                f"vth = {params.get('vth')}V, "
                f"sat = {sat}"
            )

        row["all_sat"] = int(bool(data) and all_sat)
        log.append(f" -> all saturated = {bool(row['all_sat'])}")

        return row, "\n".join(log)

    finally:
        if os.path.exists(work_spice):
            os.remove(work_spice)

if __name__ == "__main__":
    print(f"Working Directory: {BASE_DIR}")
    print(f"Target Schematic: {SCH_PATH}")

    # 1. 基準となるネットリストを study/netlists/ に書き出し
    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)

    if spice_template and os.path.exists(spice_template):
        print(f"Base netlist generated: {spice_template}")

        # W [um] (3種類のトランジスタで独立に振る)
        wn_list = [5.6, 11.2, 22.4, 44.8]      # 差動対 M1, M2
        wp_list = [5.6, 11.2, 22.4, 44.8]      # PMOS カレントミラー M3, M4
        wtail_list = [5.6, 11.2, 22.4, 44.8]   # テール電流源 M5

        # L [um] (5つのトランジスタで共通)
        l_list = [0.28, 0.56, 1.12, 2.24]

        # VDD は回路図の設定値 (.param vdd=3.3) をそのまま使うので、Vin はその範囲内
        vb_sweep = np.arange(0.6, 1.5, 0.02)
        vin_sweep = np.arange(0.1, 3.4, 0.02)
        results = []

        # 並列数 (環境変数 NUM_WORKERS で変更可能。既定はCPUコア数)
        num_workers = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))

        # テンプレートは一度だけ読み込む
        with open(spice_template, 'r') as src:
            template_content = src.read()

        # 全条件の組み合わせ
        conditions = [
            (wn, wp, wtail, l, round(float(vb), 2), round(float(vin), 2))
            for wn in wn_list
            for wp in wp_list
            for wtail in wtail_list
            for l in l_list
            for vb in vb_sweep
            for vin in vin_sweep
        ]

        num_iterations = len(conditions)

        # 作業用ネットリストはコンテナ内の一時フォルダ (/tmp/5tota_xxxx) に作る
        # (環境変数 WORK_DIR で親フォルダを変更可能)
        work_dir = tempfile.mkdtemp(prefix="5tota_", dir=os.environ.get("WORK_DIR"))

        print(f"Total conditions: {num_iterations}, workers: {num_workers}")
        print(f"Work directory: {work_dir}")

        # 計測開始
        start_time = time.perf_counter()

        try:
            # ngspiceは別プロセスで動くので、スレッドでも並列に実行される
            with ThreadPoolExecutor(max_workers=num_workers) as executor:

                futures = [
                    executor.submit(
                        simulate_point, template_content, work_dir,
                        wn, wp, wtail, l, vb, vin
                    )
                    for wn, wp, wtail, l, vb, vin in conditions
                ]

                for done, future in enumerate(as_completed(futures), 1):

                    try:
                        row, log = future.result()
                    except Exception as e:
                        row, log = None, f"ERROR: {e}"

                    elapsed = time.perf_counter() - start_time
                    print(f"[{done}/{num_iterations}] elapsed {elapsed:.1f} s")
                    print(log)

                    if row is not None:
                        results.append(row)

        finally:
            # 途中で止めた場合も一時フォルダを消す
            shutil.rmtree(work_dir, ignore_errors=True)

        # 完了順はバラバラなので、条件順に並べ直す
        results.sort(key=lambda r: (
            r["Wn_um"], r["Wp_um"], r["Wtail_um"], r["L_um"], r["Vb_V"], r["Vin_V"]
        ))

        # 計測終了
        elapsed_time = time.perf_counter() - start_time

        # W を共通にした旧データ (5tota_saturation_results.csv) は上書きしない
        csv_path = os.path.join(STUDY_DIR, "5tota_wsplit_results.csv")

        with open(csv_path, "w", newline="") as f:

            # 全行のキーを出現順に集めてヘッダにする (all_sat は末尾)
            fieldnames = []
            for row in results:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)

            if "all_sat" in fieldnames:
                fieldnames.remove("all_sat")
                fieldnames.append("all_sat")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"CSV saved: {csv_path}")
        print(f"Number of successful simulations: {len(results)} / {num_iterations}")
        print(f"Number of all-saturated conditions: {sum(r['all_sat'] for r in results)}")
        print(f"Total simulation time: {elapsed_time:.2f} s")

    else:
        print("Failed to generate base netlist. Please check xschem execution.")
