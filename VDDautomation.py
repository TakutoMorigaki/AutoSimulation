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

if __name__ == "__main__":
    print(f"Working Directory: {BASE_DIR}")
    print(f"Target Schematic: {SCH_PATH}")
    
    # 1. 基準となるネットリストを study/netlists/ に書き出し
    spice_template = export_netlist(SCH_PATH, NETLIST_DIR)
    
    if spice_template and os.path.exists(spice_template):
        print(f"Base netlist generated: {spice_template}")
        
        # VDDが0VだとDC収束エラーになりやすいため、動作下限付近（例: 0.6V〜3.0V）からスイープする
        vdd_sweep = np.arange(0.6, 4.0, 0.1)
        results = {}
        
        for vdd in vdd_sweep:
            vdd = round(float(vdd), 2)
            work_spice = os.path.join(NETLIST_DIR, f"5tota_vdd_{vdd:.2f}.spice")
            
            # テンプレートから作業用ネットリストをコピー
            with open(spice_template, 'r') as src, open(work_spice, 'w') as dst:
                dst.write(src.read())
            
            # パラメータ書き換え
            modify_netlist_params(work_spice, vdd_val=vdd)
            print(work_spice)
            
            # シミュレーション実行
            success, stdout = run_ngspice(work_spice)
            if success:
                print(f"VDD = {vdd:.2f}V: Simulation Success")
                print(stdout)

                meas_results = extract_measure_results(stdout)

                if 'gain_dc' in meas_results:
                    gain_value = meas_results['gain_dc']
                    results[vdd] = gain_value
                    print(f" -> gain_dc = {gain_value:.4f} dB")

            else:
                print(f"VDD = {vdd:.2f}V: Simulation Failed")

            # if os.path.exists(work_spice):
            #     os.remove(work_spice)

        csv_path = os.path.join(STUDY_DIR, "vdd_gain_results.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "VDD_V",
                "gain_dc_dB"
            ])

            for vdd, gain in results.items():
                writer.writerow([
                    vdd,
                    gain
                ])

            print(f"CSV saved: {csv_path}")

    else:
        print("Failed to generate base netlist. Please check xschem execution.")