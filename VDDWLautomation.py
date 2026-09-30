import os
import subprocess
import re
import numpy as np
import csv

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

    # XM3などの行だけを対象にする。
    #
    # 現在のネットリストでは、
    #
    # XM3 ... L=0.28u W=11.2u nf=4 ...
    # + ...
    #
    # となっているため、XM3がある先頭行の中から
    # WまたはLを置換する。

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

def modify_netlist_params(
        netlist_path,
        vdd_val=None,
        w_val=None,
        wtail_val=None,
        l_val=None
    ):
    """ネットリスト内の電源電圧などのパラメータを書き換える"""
    with open(netlist_path, 'r') as f:
        content = f.read()

    # modify vdd
    if vdd_val is not None:
        # \g<1> を使ってグループ1と数値の結合ミス（グループ10の誤認）を防ぐ
        content = re.sub(
            r'(\.param\s+vdd\s*=\s*)[\d\.]+',
            rf'\g<1>{vdd_val}',
            content,
            flags=re.IGNORECASE
        )

    # modify W
    if w_val is not None:
        wp_value = 2.0 * w_val
        wn_value = w_val
        wtail_value = 2.0 * w_val

        # M1
        content = replace_mos_parameter(
            content,
            "XM1",
            "W",
            wn_value
        )

        # M2
        content = replace_mos_parameter(
            content,
            "XM2",
            "W",
            wn_value
        )

        # M3
        content = replace_mos_parameter(
            content,
            "XM3",
            "W",
            wp_value
        )

        # M4
        content = replace_mos_parameter(
            content,
            "XM4",
            "W",
            wp_value
        )

        # M5
        content = replace_mos_parameter(
            content,
            "XM5",
            "W",
            wtail_value
        )

    # modify L
    if l_val is not None:

        for transistor in [
            "XM1",
            "XM2",
            "XM3",
            "XM4",
            "XM5"
        ]:

            content = replace_mos_parameter(
                content,
                transistor,
                "L",
                l_val
            )

    with open(netlist_path, 'w') as f:
        f.write(content)

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
        # print(results)

    return results

W_SWEEP = [
    5.6,
    8.4,
    11.2,
    16.8,
    22.4,
    33.6,
    44.8
]

L_SWEEP = [
    0.28,
    0.35,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00
]

VDD_SWEEP = np.arange(
    0.5,
    4.001,
    0.01
)

if __name__ == "__main__":
    print(f"Working Directory: {BASE_DIR}")
    print(f"Target Schematic: {SCH_PATH}")
    
    # 1. 基準となるネットリストを study/netlists/ に書き出し
    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    
    if spice_template and os.path.exists(spice_template):
        print(f"Base netlist generated: {spice_template}")

        csv_path = os.path.join(STUDY_DIR, "vddwl_gain_results.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "Wn_um",
                "Wp_um",
                "Wtail_um",
                "L_um",
                "VDD_V",
                "gain_dc_dB"
            ])

            results = []
            
            for w in W_SWEEP:
                for l in L_SWEEP:
                    for vdd in VDD_SWEEP:
                        vdd = round(float(vdd), 2)
                        work_spice = os.path.join(NETLIST_DIR, f"5tota_W{w:.2f}_L{l:.2f}_VDD{vdd:.2f}.spice")
                        
                        # テンプレートから作業用ネットリストをコピー
                        with open(spice_template, 'r') as src, open(work_spice, 'w') as dst:
                            dst.write(src.read())
                        
                        # パラメータ書き換え
                        modify_netlist_params(work_spice, vdd_val=vdd, w_val=w, l_val=l)
                        print(work_spice)
                        
                        # シミュレーション実行
                        success, stdout = run_ngspice(work_spice)
                        gain_value = ""
                        if success:
                            print(f"W/L = {w:.2f}/{l:.2f}: Simulation Success")

                            meas_results = extract_measure_results(stdout)

                            if 'gain_dc' in meas_results:
                                gain_value = meas_results['gain_dc']
                                print(f"VDD = {vdd:.2f} -> gain_dc = {gain_value:.4f} dB")

                        else:
                            print(f"W/L = {w:.2f}/{l:.2f}: Simulation Failed")

                        writer.writerow([
                            w,
                            2.0*w,
                            2.0*w,
                            l,
                            vdd,
                            gain_value
                        ])

                        f.flush()

        print(f"CSV saved: {csv_path}")

    else:
        print("Failed to generate base netlist. Please check xschem execution.")