from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
methods = ["runs_ssod_paper3_one", "runs_ssod_paper3_one_datonly", "runs_ssod_paper3_one_fixedtau"]
seeds = [0, 1, 2]

print("ROOT =", ROOT)
for s in seeds:
    tag = f"r20_seed{s}"
    for m in methods:
        p = ROOT / m / f"{tag}_final_val" / "final_metrics.csv"
        print(("[OK] " if p.exists() else "[MISSING] ") + str(p))
        if p.exists():
            print(p.read_text(encoding="utf-8-sig"))
            print("-" * 60)
