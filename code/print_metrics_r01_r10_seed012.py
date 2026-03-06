from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
methods = [
    "runs_ssod_paper3_one",
    "runs_ssod_paper3_one_datonly",
    "runs_ssod_paper3_one_fixedtau",
]
ratios = ["r01", "r10"]
seeds = [0, 1, 2]

print("ROOT =", ROOT)

for ratio in ratios:
    print("\n" + "=" * 20, ratio, "=" * 20)
    for seed in seeds:
        tag = f"{ratio}_seed{seed}"
        for m in methods:
            p = ROOT / m / f"{tag}_final_val" / "final_metrics.csv"
            print(("[OK] " if p.exists() else "[MISSING] ") + str(p))
            if p.exists():
                print(p.read_text(encoding="utf-8-sig"))
                print("-" * 60)
