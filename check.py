import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

STUDY_DIR = os.path.join(PROJECT_ROOT, "study")
NETLIST_DIR = os.path.join(STUDY_DIR, "netlists")
SCH_PATH = os.path.join(STUDY_DIR, "tb_ac_5tota.sch")

env = os.environ.copy()

print("===== Environment =====")
print("PDK      =", env.get("PDK"))
print("PDK_ROOT =", env.get("PDK_ROOT"))
print("PDKPATH  =", env.get("PDKPATH"))
print("=======================")

cmd = [
    "xschem",
    "-q",
    "-n",
    "-s",
    "-o",
    NETLIST_DIR,
    SCH_PATH
]

print("Executing:")
print(" ".join(cmd))

result = subprocess.run(
    cmd,
    env=env,
    cwd=STUDY_DIR,
    capture_output=True,
    text=True
)

print("===== stdout =====")
print(result.stdout)

print("===== stderr =====")
print(result.stderr)

print("returncode =", result.returncode)
