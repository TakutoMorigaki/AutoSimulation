import os
import subprocess
import re
import numpy as np
import csv


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
            0.6,
            1.0,
            0.1
        )

        # Vb
        vb_sweep = np.arange(
            0.3,
            1.01,
            0.1
        )

        # Vin
        vin_sweep = np.arange(
            0.3,
            1.01,
            0.1
        )


        # 結果保存用
        results = []


        # ====================================================
        # 3. VDD × Vb × Vin の全組み合わせを実行
        # ====================================================

        for vdd in vdd_sweep:

            vdd = round(float(vdd), 2)

            for vb in vb_sweep:

                vb = round(float(vb), 2)

                for vin in vin_sweep:

                    vin = round(float(vin), 2)


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


                        if vout is not None:

                            print(
                                f"SUCCESS: "
                                f"Vout = {vout:.6g} V"
                            )

                            results.append([
                                vdd,
                                vb,
                                vin,
                                vout
                            ])

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

            writer = csv.writer(f)

            writer.writerow([
                "VDD_V",
                "Vb_V",
                "Vin_V",
                "Vout_V"
            ])

            writer.writerows(results)


        print("\n====================================")
        print("CSV saved:")
        print(csv_path)
        print(
            f"Number of successful simulations: "
            f"{len(results)}"
        )
        print("====================================")


    else:

        print(
            "Failed to generate base netlist. "
            "Please check xschem execution."
        )