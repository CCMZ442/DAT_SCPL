import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

SCRIPT_SCPL = ROOT / "run_one_ssod_full_win_v2_memsafe_autoteacher.py"
SCRIPT_DAT  = ROOT / "run_one_ssod_full_win_v2_memsafe_datonly.py"
SCRIPT_FIX  = ROOT / "run_one_ssod_full_win_v2_memsafe_fixedtau.py"

def final_csv(method_root: str, tag: str) -> Path:
    return ROOT / method_root / f"{tag}_final_val" / "final_metrics.csv"

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_if_needed(tag: str, method_root: str, script: Path):
    out = final_csv(method_root, tag)
    if out.exists():
        print(f"[SKIP] {tag} -> exists: {out}")
        return
    run([PY, str(script), "--tag", tag])

def main():
    for seed in [0, 1, 2]:
        tag = f"r20_seed{seed}"
        run_if_needed(tag, "runs_ssod_paper3_one",          SCRIPT_SCPL)
        run_if_needed(tag, "runs_ssod_paper3_one_datonly",  SCRIPT_DAT)
        run_if_needed(tag, "runs_ssod_paper3_one_fixedtau", SCRIPT_FIX)

    print("\nDONE.")

if __name__ == "__main__":
    main()
