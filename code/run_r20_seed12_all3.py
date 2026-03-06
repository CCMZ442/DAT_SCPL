import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def exists_final(method_root: str, tag: str) -> bool:
    return (ROOT / method_root / f"{tag}_final_val" / "final_metrics.csv").exists()

def main():
    # 用你真实脚本路径（在根目录，不在 code）
    script_scpl = ROOT / "run_one_ssod_full_win_v2_memsafe_autoteacher.py"
    script_dat  = ROOT / "run_one_ssod_full_win_v2_memsafe_datonly.py"
    script_fix  = ROOT / "run_one_ssod_full_win_v2_memsafe_fixedtau.py"

    for seed in [1, 2]:
        tag = f"r20_seed{seed}"

        # 1) 先保证 teacher+DAT-SCPL 跑完（它会创建 teacher）
        if not exists_final("runs_ssod_paper3_one", tag):
            run([PY, str(script_scpl), "--tag", tag])
        else:
            print(f"[SKIP] {tag} DAT-SCPL final exists")

        # 2) 再跑 DAT-only（假设复用 teacher）
        if not exists_final("runs_ssod_paper3_one_datonly", tag):
            run([PY, str(script_dat), "--tag", tag])
        else:
            print(f"[SKIP] {tag} DAT-only final exists")

        # 3) 再跑 FixedTau（假设复用 teacher）
        if not exists_final("runs_ssod_paper3_one_fixedtau", tag):
            run([PY, str(script_fix), "--tag", tag])
        else:
            print(f"[SKIP] {tag} FixedTau final exists")

    print("\nDONE.")

if __name__ == "__main__":
    main()
