import os
import subprocess
import re
import numpy as np
import csv
import time
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# パスの定義
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

STUDY_DIR = os.path.join(PROJECT_ROOT, "study")
NETLIST_DIR = os.path.join(STUDY_DIR, "netlists")

# 対象のXschem回路図
SCH_FILENAME = "cascade.sch"
SCH_PATH = os.path.join(STUDY_DIR, SCH_FILENAME)

env = os.environ.copy()


# ============================================================
# Xschem → SPICEネットリスト生成
# ============================================================

def export_netlist(sch_path, output_dir):
    """
    xschemをバッチモードで実行し、
    SPICEネットリストを生成する
    """

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "xschem",
        "-q",
        "-n",
        "-s",
        "-o",
        output_dir,
        sch_path
    ]

    print(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=STUDY_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("xschem Error:")
        print(result.stderr)
        return None

    base_name = os.path.splitext(
        os.path.basename(sch_path)
    )[0]

    return os.path.join(
        output_dir,
        f"{base_name}.spice"
    )


# ============================================================
# MOSパラメータ変更
# ============================================================

def replace_mos_parameter(
        content,
        transistor,
        parameter,
        value
    ):
    """
    指定したトランジスタのWまたはLを変更する。

    例:

        XM3 ... L=0.28u W=11.2u

    ↓

        XM3 ... L=0.50u W=20u
    """

    lines = content.splitlines()

    pattern = re.compile(
        rf'(\b{re.escape(transistor)}\b.*?\b'
        rf'{re.escape(parameter)}\s*=\s*)'
        rf'[\d.+\-eE]+u',
        flags=re.IGNORECASE
    )

    found = False

    for i, line in enumerate(lines):

        # XM1～XM5の行を検索
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


# ============================================================
# .param の変更
# ============================================================

def modify_netlist_params(netlist_path, **kwargs):
    """
    ネットリストファイル内の各種パラメータを書き換える。
    (引数は modify_netlist_content と同じ)
    """

    with open(netlist_path, "r") as f:
        content = f.read()

    content = modify_netlist_content(content, **kwargs)

    with open(netlist_path, "w") as f:
        f.write(content)


def modify_netlist_content(
        content,
        vdd_val=None,
        vb_val=None,
        vin_val=None,
        w1_val=None,
        w2_val=None,
        l_val=None,
        rd_val=None
    ):
    """
    ネットリストの文字列内の各種パラメータを書き換えて返す。
    (ファイルを介さないので、並列実行時のファイル入出力を減らせる)
    """


    # --------------------------------------------------------
    # VDD
    # --------------------------------------------------------

    if vdd_val is not None:

        content = re.sub(
            r'(\.param\s+vdd\s*=\s*)'
            r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',

            rf'\g<1>{vdd_val}',

            content,

            flags=re.IGNORECASE
        )


    # --------------------------------------------------------
    # Vb
    # --------------------------------------------------------

    if vb_val is not None:

        content = re.sub(
            r'(\.param\s+vb\s*=\s*)'
            r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',

            rf'\g<1>{vb_val}',

            content,

            flags=re.IGNORECASE
        )


    # --------------------------------------------------------
    # Vin
    # --------------------------------------------------------

    if vin_val is not None:

        content = re.sub(
            r'(\.param\s+vin\s*=\s*)'
            r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',

            rf'\g<1>{vin_val}',

            content,

            flags=re.IGNORECASE
        )


    # --------------------------------------------------------
    # RD (負荷抵抗 R1 [Ω]。回路図で value={R1} としている)
    # --------------------------------------------------------

    if rd_val is not None:

        content, count = re.subn(
            r'(\.param\s+R1\s*=\s*)'
            r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',

            rf'\g<1>{rd_val}',

            content,

            flags=re.IGNORECASE
        )

        if count == 0:
            print("WARNING: .param R1 がネットリスト内に見つかりません")


    # --------------------------------------------------------
    # W (M1, M2 は別々に指定する)
    # --------------------------------------------------------

    if w1_val is not None:

        content = replace_mos_parameter(
            content,
            "XM1",
            "W",
            w1_val
        )

    if w2_val is not None:

        content = replace_mos_parameter(
            content,
            "XM2",
            "W",
            w2_val
        )


    # --------------------------------------------------------
    # L
    # --------------------------------------------------------

    if l_val is not None:

        for transistor in [
            "XM1",
            "XM2",
        ]:

            content = replace_mos_parameter(
                content,
                transistor,
                "L",
                l_val
            )


    return content


# ============================================================
# ngspice実行
# ============================================================

def run_ngspice(netlist_path):
    """
    ngspiceをバッチモードで実行する
    """

    cmd = [
        "ngspice",
        "-b",
        netlist_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("Ngspice Error:")
        print(result.stderr)

        return False, result.stdout

    return True, result.stdout


# ============================================================
# ngspiceの出力から v(out) を取得
# ============================================================

def extract_vout(stdout_text):
    """
    ngspiceの

        v(out) = 1.23456

    のような出力から値を取り出す。
    """

    pattern = (
        r'v\(out\)\s*=\s*'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)'
        r'(?:[eE][+-]?\d+)?)'
    )

    match = re.search(
        pattern,
        stdout_text,
        flags=re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None

# ============================================================
# ngspiceの出力から各トランジスタの id,vds,vdsat,vgs,vth を取得
# ============================================================

def device_to_transistor(device):
    """
    m.x1.xm1.m0 / m.xm1.m0 などのデバイス名を XM1 に正規化する。
    MOS以外 (vin, vdd などの電圧源) は None を返す。
    """

    match = re.search(
        r'(xm\d+)',
        device,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return None


def extract_device_parameters(stdout_text):

    data = {}

    lines = stdout_text.splitlines()

    current_devices = []

    for line in lines:

        stripped = line.strip()

        # --------------------------------------------
        # device行
        # --------------------------------------------

        if stripped.lower().startswith("device"):

            tokens = stripped.split()
            current_devices = tokens[1:]

            for device in current_devices:

                transistor = device_to_transistor(device)

                if transistor is not None and transistor not in data:
                    data[transistor] = {}


        # --------------------------------------------
        # 各パラメータ
        # --------------------------------------------

        elif current_devices:

            tokens = stripped.split()

            if len(tokens) < 2:
                continue

            parameter = tokens[0].lower()

            if parameter not in [
                "id",
                "vds",
                "vdsat",
                "vgs",
                "vth"
            ]:
                continue

            values = tokens[1:]

            if len(values) != len(current_devices):
                continue

            for device, value in zip(
                current_devices,
                values
            ):

                transistor = device_to_transistor(device)

                if transistor is None:
                    continue

                try:
                    data[transistor][parameter] = float(value)
                except ValueError:
                    pass

    # print(data)

    return data


# ============================================================
# 飽和判定
# ============================================================

DEVICE_PARAMS = ["id", "vds", "vdsat", "vgs", "vth"]


def is_saturated(params):
    """
    |Vgs| > |Vth| かつ |Vds| >= |Vdsat| なら飽和とみなす。
    (PMOSは負の値で出力されるため絶対値で比較)
    """

    if not all(p in params for p in ["vds", "vdsat", "vgs", "vth"]):
        return None

    on = abs(params["vgs"]) > abs(params["vth"])
    sat = abs(params["vds"]) >= abs(params["vdsat"])

    return on and sat


# ============================================================
# 1条件分のシミュレーション (並列実行の単位)
# ============================================================

def simulate_point(template_content, work_dir, rd, w1, w2, vb, vin):
    """
    1つの (RD, W1, W2, Vb, Vin) についてネットリスト作成 → ngspice実行 → 解析を行う。
    並列実行時に表示が混ざらないよう、ログは文字列で返す。

    作業用ネットリストは work_dir (コンテナ内の一時フォルダ) に作る。
    Windows側のマウントフォルダに作ると、ファイル入出力の待ちで遅くなるため。

    戻り値: (row または None, ログ文字列)
    """

    log = [
        f"====================================\n"
        f"RD  = {rd:g} ohm\n"
        f"W1  = {w1} um, W2 = {w2} um\n"
        f"Vb  = {vb:.2f} V\n"
        f"Vin = {vin:.2f} V\n"
        f"===================================="
    ]

    # ----------------------------------------
    # 作業用ネットリスト
    # (条件ごとにファイル名が異なるので並列でも衝突しない)
    # ----------------------------------------

    work_spice = os.path.join(
        work_dir,
        (
            f"cascade_"
            f"rd_{rd:g}_"
            f"w1_{w1}_"
            f"w2_{w2}_"
            f"vb_{vb:.2f}_"
            f"vin_{vin:.2f}.spice"
        )
    )

    row = None

    try:

        # .param, W, RD をメモリ上で書き換えてから、1回だけ書き込む
        content = modify_netlist_content(
            template_content,
            vb_val=vb,
            vin_val=vin,
            w1_val=w1,
            w2_val=w2,
            rd_val=rd
        )

        with open(work_spice, "w") as dst:
            dst.write(content)

        # ngspice実行
        success, stdout = run_ngspice(work_spice)

        if not success:
            log.append("Simulation Failed")
            return None, "\n".join(log)

        vout = extract_vout(stdout)
        data = extract_device_parameters(stdout)

        if vout is None:
            log.append("WARNING: v(out) が取得できませんでした")
            log.append(stdout)
            return None, "\n".join(log)

        log.append(
            f"SUCCESS: "
            f"Vout = {vout:.6g} V, Av = {vout/vin:.6g}"
        )

        row = {
            "RD_ohm": rd,
            "W1_um": w1,
            "W2_um": w2,
            "Vb_V": vb,
            "Vin_V": vin,
            "Vout_V": vout,
            "Av": vout / vin
        }

        all_sat = True

        for transistor in sorted(data):

            params = data[transistor]
            sat = is_saturated(params)
            name = transistor.replace("XM", "M")

            for p in DEVICE_PARAMS:
                row[f"{name}_{p}"] = params.get(p)

            row[f"{name}_sat"] = (
                None if sat is None else int(sat)
            )

            if not sat:
                all_sat = False

            log.append(
                f"{name}: id = {params.get('id')}A, "
                f"vds = {params.get('vds')}V, "
                f"vdsat = {params.get('vdsat')}V, "
                f"vgs = {params.get('vgs')}V, "
                f"vth = {params.get('vth')}V, "
                f"sat = {sat}"
            )

        row["all_sat"] = int(bool(data) and all_sat)

    finally:

        if os.path.exists(work_spice):
            os.remove(work_spice)

    return row, "\n".join(log)


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    print(f"Working Directory: {BASE_DIR}")
    print(f"Target Schematic: {SCH_PATH}")


    # ========================================================
    # 1. 基準ネットリストを生成
    # ========================================================

    spice_template = export_netlist(
        SCH_PATH,
        NETLIST_DIR
    )


    if spice_template and os.path.exists(spice_template):

        print(
            f"Base netlist generated: "
            f"{spice_template}"
        )


        # ====================================================
        # 2. 掃引条件
        # ====================================================

        # VDD は回路図の設定値 (.param vdd) をそのまま使う

        # RD [Ω] (負荷抵抗 R1。複数指定するとそれぞれで掃引する)
        rd_list = [2000, 3000, 4000, 5000]

        # W [um] (M1, M2 それぞれ独立に振る)
        w1_list = [5.6, 11.2, 22.4, 44.8]
        w2_list = [5.6, 11.2, 22.4, 44.8]

        # Vb
        vb_sweep = np.arange(
            0.5,
            3.0,
            0.02
        )

        # Vin
        vin_sweep = np.arange(
            0.3,
            1.7,
            0.02
        )


        # 並列数 (環境変数 NUM_WORKERS で変更可能。既定はCPUコア数)
        num_workers = int(
            os.environ.get("NUM_WORKERS", os.cpu_count() or 1)
        )

        # テンプレートは一度だけ読み込む
        with open(spice_template, "r") as src:
            template_content = src.read()

        # 全条件の組み合わせ
        conditions = [
            (
                rd,
                w1,
                w2,
                round(float(vb), 2),
                round(float(vin), 2)
            )
            for rd in rd_list
            for w1 in w1_list
            for w2 in w2_list
            for vb in vb_sweep
            for vin in vin_sweep
        ]

        num_iterations = len(conditions)

        # 結果保存用
        results = []


        # ====================================================
        # 3. RD × W1 × W2 × Vb × Vin の全組み合わせを並列実行
        # ====================================================

        # 作業用ネットリストはコンテナ内の一時フォルダ (/tmp/cascade_xxxx) に作る
        # (環境変数 WORK_DIR で親フォルダを変更可能)
        work_dir = tempfile.mkdtemp(
            prefix="cascade_",
            dir=os.environ.get("WORK_DIR")
        )

        print(
            f"\nTotal conditions: {num_iterations}, "
            f"workers: {num_workers}\n"
            f"Work directory: {work_dir}"
        )

        # 計測開始
        start_time = time.perf_counter()

        try:

            # ngspiceは別プロセスで動くので、スレッドでも並列に実行される
            with ThreadPoolExecutor(max_workers=num_workers) as executor:

                futures = [
                    executor.submit(
                        simulate_point,
                        template_content,
                        work_dir,
                        rd,
                        w1,
                        w2,
                        vb,
                        vin
                    )
                    for rd, w1, w2, vb, vin in conditions
                ]

                for done, future in enumerate(as_completed(futures), 1):

                    try:
                        row, log = future.result()
                    except Exception as e:
                        row, log = None, f"ERROR: {e}"

                    elapsed = time.perf_counter() - start_time

                    print(
                        f"\n[{done}/{num_iterations}] "
                        f"elapsed {elapsed:.1f} s\n{log}"
                    )

                    if row is not None:
                        results.append(row)

        finally:

            # 途中で止めた場合も一時フォルダを消す
            shutil.rmtree(work_dir, ignore_errors=True)

        # 完了順はバラバラなので、条件順に並べ直す
        results.sort(
            key=lambda r: (r["RD_ohm"], r["W1_um"], r["W2_um"], r["Vb_V"], r["Vin_V"])
        )


        # 計測終了
        elapsed_time = time.perf_counter() - start_time

        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)


        # ====================================================
        # 4. CSV保存
        # ====================================================

        csv_path = os.path.join(
            STUDY_DIR,
            "cascade_results.csv"
        )


        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            # 全行のキーを出現順に集めてヘッダにする
            fieldnames = []

            for row in results:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)

            # all_sat は末尾に置く
            if "all_sat" in fieldnames:
                fieldnames.remove("all_sat")
                fieldnames.append("all_sat")

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)


        print("\n====================================")
        print("CSV saved:")
        print(csv_path)
        print(
            f"Number of successful simulations: "
            f"{len(results)} / {num_iterations}"
        )
        print(
            f"Total simulation time: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} "
            f"({elapsed_time:.2f} s)"
        )

        if num_iterations > 0:
            print(
                f"Average time per iteration: "
                f"{elapsed_time / num_iterations:.3f} s"
            )

        print("====================================")


    else:

        print(
            "Failed to generate base netlist. "
            "Please check xschem execution."
        )