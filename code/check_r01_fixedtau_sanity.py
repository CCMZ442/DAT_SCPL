from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))

def count_labels(labels_dir: Path) -> int:
    if not labels_dir.exists():
        return -1
    total = 0
    for txt in labels_dir.glob("*.txt"):
        s = txt.read_text(encoding="utf-8-sig").strip()
        if not s:
            continue
        total += len(s.splitlines())
    return total

def main():
    tags = ["r01_seed0", "r01_seed1"]
    for tag in tags:
        print("\n==============================")
        print("TAG:", tag)
        print("==============================")

        scpl_thr = ROOT / "runs_ssod_paper3_one" / f"{tag}_pseudo_labels" / "scpl_thresholds.json"
        fix_tau  = ROOT / "runs_ssod_paper3_one_fixedtau" / f"{tag}_pseudo_labels" / "fixed_tau.json"

        scpl_labels = ROOT / "runs_ssod_paper3_one" / f"{tag}_pseudo_labels" / "labels_train"
        fix_labels  = ROOT / "runs_ssod_paper3_one_fixedtau" / f"{tag}_pseudo_labels" / "labels_train"

        scpl_thr_j = load_json(scpl_thr)
        fix_tau_j = load_json(fix_tau)

        print("[SCPL] thresholds json:", scpl_thr)
        print("  exists:", scpl_thr.exists())
        print("  content:", scpl_thr_j)

        print("[FIX ] fixed_tau json:", fix_tau)
        print("  exists:", fix_tau.exists())
        print("  content:", fix_tau_j)

        scpl_cnt = count_labels(scpl_labels)
        fix_cnt = count_labels(fix_labels)
        print("[COUNT] SCPL kept labels:", scpl_cnt, "dir:", scpl_labels)
        print("[COUNT] FIX  kept labels:", fix_cnt,  "dir:", fix_labels)

        if scpl_thr_j is not None:
            # 是否全部=0.25（或非常接近）
            vals = [float(v) for v in scpl_thr_j.values()]
            all_025 = all(abs(v - 0.25) < 1e-9 for v in vals)
            print("[SCPL] all thresholds == 0.25 ?", all_025)

if __name__ == "__main__":
    main()
