"""
CortiX — Z-Threshold Sweep

Runs the benchmark suite across a range of ANOMALY_Z_THRESHOLD values
(via the existing --tune_file mechanism) and aggregates results into
a single comparison table + tradeoff plot, so you can pick a threshold
that satisfies FPR/detection/F1 targets jointly rather than guessing.

Usage:
    python evaluation/sweep_threshold.py --dataset nslkdd

Assumes benchmark.py already has the encoding-bug fix and the full
train-derived warmup wired in — this sweep tunes calibration, it does
not substitute for fixing the pipeline bug.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Sweep range
# ──────────────────────────────────────────────
# Start wide, then narrow once you see where FPR/recall cross your targets.
# z=2.0 ~ flags the outer ~5% under a normal assumption; z=3.0 ~ outer ~0.3%
# (which not coincidentally lines up with your FPR ≤0.3% target — worth
# treating that as your first checkpoint, not just a coincidence).
Z_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

ACCEPTANCE = {
    "latency_p50_ms": ("<=", 9.0),
    "fpr": ("<=", 0.003),
    "detection_rate": (">=", 0.70),
    "f1_score": (">=", 0.70),
}


def check(value: float, op: str, target: float) -> bool:
    return value <= target if op == "<=" else value >= target


def run_one(z: float, dataset: str, base_output: Path) -> dict:
    tune_path = base_output / f"tune_z{z}.json"
    out_dir = base_output / f"z_{z}"
    tune_path.write_text(json.dumps({"z_threshold": z}))

    cmd = [
        sys.executable, "evaluation/benchmark.py",
        "--dataset", dataset,
        "--tune_file", str(tune_path),
        "--output", str(out_dir),
    ]
    print(f"\n{'=' * 70}\nRunning z_threshold = {z}\n{'=' * 70}")
    subprocess.run(cmd, check=True)

    with open(out_dir / "benchmark_results.json") as f:
        result = json.load(f)[0]
    result["z_threshold"] = z
    return result


def summarize(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("Z-THRESHOLD SWEEP — SUMMARY")
    print("=" * 100)
    header = f"{'z':>6} | {'Accuracy':>9} | {'FPR':>9} | {'Detection':>10} | {'F1':>8} | {'p50 (ms)':>9} | Pass?"
    print(header)
    print("-" * 100)

    best = None
    for r in results:
        checks = {k: check(r[k], op, tgt) for k, (op, tgt) in ACCEPTANCE.items()}
        all_pass = all(checks.values())
        n_pass = sum(checks.values())
        if best is None or n_pass > best[1]:
            best = (r["z_threshold"], n_pass, r)

        print(
            f"{r['z_threshold']:>6.1f} | "
            f"{r['accuracy']*100:>8.2f}% | "
            f"{r['fpr']*100:>8.4f}% | "
            f"{r['detection_rate']*100:>9.2f}% | "
            f"{r['f1_score']:>8.4f} | "
            f"{r['latency_p50_ms']:>8.2f}ms | "
            f"{'ALL PASS ✅' if all_pass else f'{n_pass}/4'}"
        )
    print("=" * 100)
    if best:
        print(f"\nBest candidate so far: z_threshold = {best[0]} ({best[1]}/4 criteria met)")
        if best[1] < 4:
            print("No threshold cleared all four — narrow the sweep around the best "
                  "candidate, or revisit warmup size / feature selection before "
                  "widening the search further.")
    print()


def plot_tradeoff(results: list[dict], output_path: Path) -> None:
    """FPR and detection rate vs z_threshold — the core calibration tradeoff."""
    zs = [r["z_threshold"] for r in results]
    fprs = [r["fpr"] * 100 for r in results]
    dets = [r["detection_rate"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(zs, fprs, "o-", color="#ef4444", label="FPR (%)")
    ax1.axhline(0.3, color="#ef4444", linestyle="--", alpha=0.4, label="FPR target (0.3%)")
    ax1.set_xlabel("z_threshold")
    ax1.set_ylabel("False Positive Rate (%)", color="#ef4444")
    ax1.tick_params(axis="y", labelcolor="#ef4444")

    ax2 = ax1.twinx()
    ax2.plot(zs, dets, "s-", color="#3b82f6", label="Detection rate (%)")
    ax2.axhline(70, color="#3b82f6", linestyle="--", alpha=0.4, label="Detection target (70%)")
    ax2.set_ylabel("Detection Rate (%)", color="#3b82f6")
    ax2.tick_params(axis="y", labelcolor="#3b82f6")

    fig.suptitle("FPR vs. Detection Rate across z_threshold", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tradeoff plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep ANOMALY_Z_THRESHOLD for CortiX SNN")
    parser.add_argument("--dataset", type=str, default="nslkdd")
    parser.add_argument("--output", type=str, default="evaluation/results/z_sweep")
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="Override the default z_threshold list, e.g. --values 2.6 2.8 3.0 3.2",
    )
    args = parser.parse_args()

    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)

    thresholds = args.values if args.values else Z_THRESHOLDS
    results = [run_one(z, args.dataset, base_output) for z in thresholds]

    summarize(results)
    plot_tradeoff(results, base_output / "fpr_vs_detection_tradeoff.png")

    with open(base_output / "sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
