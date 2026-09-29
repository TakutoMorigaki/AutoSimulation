import os
import subprocess
import re
import numpy as np
import csv
import time


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

def modify_netlist_params(
        netlist_path,
        vdd_val=None,
        vb_val=None,
        vin_val=None,
        w_val=None,
        wtail_val=None,
        l_val=None
    ):
    """
    ネットリスト内の各種パラメータを書き換える。
    """

    with open(netlist_path, "r") as f:
        content = f.read()


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
    # W
    # --------------------------------------------------------

    if w_val is not None:

        # M1
        content = replace_mos_parameter(
            content,
            "XM1",
            "W",
            w_val
        )

        # M2
        content = replace_mos_parameter(
            content,
            "XM2",
            "W",
            w_val
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


    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    with open(netlist_path, "w") as f:
        f.write(content)


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

        # VDD
        vdd_sweep = np.arange(
            1.0,
            5.0,
            0.1
        )

        # Vb
        vb_sweep = np.arange(
            0.5,
            3.0,
            0.1
        )

        # Vin
        vin_sweep = np.arange(
            0.3,
            1.7,
            0.1
        )


        # 結果保存用
        results = []


        # ====================================================
        # 3. VDD × Vb × Vin の全組み合わせを実行
        # ====================================================

        # 計測開始
        start_time = time.perf_counter()
        num_iterations = 0

        for vdd in vdd_sweep:

            vdd = round(float(vdd), 2)

            for vb in vb_sweep:

                vb = round(float(vb), 2)

                for vin in vin_sweep:

                    vin = round(float(vin), 2)

                    num_iterations += 1

                    print(
                        f"\n"
                        f"====================================\n"
                        f"VDD = {vdd:.2f} V\n"
                        f"Vb  = {vb:.2f} V\n"
                        f"Vin = {vin:.2f} V\n"
                        f"===================================="
                    )


                    # ----------------------------------------
                    # 作業用ネットリスト
                    # ----------------------------------------

                    work_spice = os.path.join(
                        NETLIST_DIR,
                        (
                            f"cascade_"
                            f"vdd_{vdd:.2f}_"
                            f"vb_{vb:.2f}_"
                            f"vin_{vin:.2f}.spice"
                        )
                    )


                    # ----------------------------------------
                    # テンプレートをコピー
                    # ----------------------------------------

                    with open(
                        spice_template,
                        "r"
                    ) as src:

                        with open(
                            work_spice,
                            "w"
                        ) as dst:

                            dst.write(src.read())


                    # ----------------------------------------
                    # .paramを書き換える
                    # ----------------------------------------

                    modify_netlist_params(
                        work_spice,
                        vdd_val=vdd,
                        vb_val=vb,
                        vin_val=vin
                    )


                    # ----------------------------------------
                    # ngspice実行
                    # ----------------------------------------

                    success, stdout = run_ngspice(
                        work_spice
                    )


                    if success:

                        # ------------------------------------
                        # Voutを取得
                        # ------------------------------------

                        vout = extract_vout(stdout)
                        data = extract_device_parameters(stdout)


                        if vout is not None:

                            print(
                                f"SUCCESS: "
                                f"Vout = {vout:.6g} V, Av = {vout/vin:.6g}"
                            )
                            row = {
                                "VDD_V": vdd,
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

                                print(
                                    f"{name}: id = {params.get('id')}A, "
                                    f"vds = {params.get('vds')}V, "
                                    f"vdsat = {params.get('vdsat')}V, "
                                    f"vgs = {params.get('vgs')}V, "
                                    f"vth = {params.get('vth')}V, "
                                    f"sat = {sat}"
                                )

                            row["all_sat"] = int(bool(data) and all_sat)

                            # print(stdout)

                            results.append(row)

                        else:

                            print(
                                "WARNING: "
                                "v(out) が取得できませんでした"
                            )

                            print(stdout)

                    else:

                        print(
                            "Simulation Failed"
                        )

                    if os.path.exists(work_spice):
                        os.remove(work_spice)


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