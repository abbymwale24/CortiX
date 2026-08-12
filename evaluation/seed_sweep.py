"""
CortiX — Seed Sweep

Runs the REAL Hebbian SNN benchmark (live training + evaluation, no
hardcoding, no simulation) across N different explicit seeds and reports
the distribution of outcomes.

Why this exists
----------------
The unsupervised Hebbian/STDP/kWTA core has no gradient-based error
correction, so its behaviour is sensitive to the random weight
initialisation it starts from. A single benchmark run (one seed) can't
tell you whether a good number was a real result or a lucky draw.

This script re-runs the exact same live pipeline (SpikeEncoder ->
HebbianEnsemble warmup -> evaluation) once per seed, using the actual
NSL-KDD (or CICIDS2017 / synthetic) data loaders and evaluate_snn()
function from benchmark.py -- nothing here is precomputed or faked.

Usage:
    python evaluation/seed_sweep.py --dataset nslkdd \
        --tune_file evaluation/tuned_params_nslkdd.json \
        --seeds 1 2 3 4 5 6 7 8 9 10

    # or a simple range:
    python evaluation/seed_sweep.py --dataset nslkdd --n_seeds 10

Output:
    evaluation/results/seed_sweep_<dataset>.json   (raw per-seed metrics)
    evaluation/results/seed_sweep_<dataset>.png     (distribution plot)
    Printed summary table with mean +/- std for each metric.
"""

import os
import json
import argparse
import logging
from typing import Optional, cast

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cortix.config import config
from evaluation.benchmark import (
    evaluate_snn,
    generate_synthetic_flows,
    load_nslkdd_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.evaluation.seed_sweep")


def load_dataset_once(dataset: str):
    """
    Load the requested dataset a single time, so the sweep doesn't pay the
    (real) CSV-loading / preprocessing cost once per seed. The SNN weight
    init and training/evaluation for each seed is still fully live.
    """
    if dataset == "nslkdd":
        nsl_data = load_nslkdd_dataset()
        y_train = nsl_data["y_train"]
        X_train_sel = nsl_data["X_train"]
        benign_train_mask = y_train == 0
        benign_train = X_train_sel[benign_train_mask]
        warmup_n = min(10000, len(benign_train))

        return dict(
            features=nsl_data["X_test"],
            labels_binary=nsl_data["y_test"],
            labels_class=nsl_data["y_test_class"],
            warmup_features=benign_train[:warmup_n],
            context_keys=None,
            warmup_context_keys=None,
        )

    elif dataset == "cicids2017":
        from evaluation.benchmark import load_cicids2017_dataset
        X_test_sel, X_test_scaled, y_test, y_test_class = load_cicids2017_dataset()
        return dict(
            features=X_test_sel,
            labels_binary=y_test,
            labels_class=y_test_class,
            warmup_features=None,
            context_keys=None,
            warmup_context_keys=None,
        )

    else:
        features, labels_binary, labels_class = generate_synthetic_flows(
            num_benign=3000, num_attack=1000
        )
        return dict(
            features=features,
            labels_binary=labels_binary,
            labels_class=labels_class,
            warmup_features=None,
            context_keys=None,
            warmup_context_keys=None,
        )


def run_sweep(dataset: str, seeds: list[int], tune_file: Optional[str] = None) -> list[dict]:
    if tune_file and os.path.exists(tune_file):
        with open(tune_file, "r") as f:
            t = json.load(f)
            config.SLIDING_WINDOW_SIZE = t.get("window_size", config.SLIDING_WINDOW_SIZE)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.HEBBIAN_LR = t.get("learning_rate", t.get("eta", config.HEBBIAN_LR))
            config.METAPLASTICITY_ALPHA = t.get("meta_alpha", config.METAPLASTICITY_ALPHA)
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))
            logger.info("Loaded tuning config: %s", t)

    data = load_dataset_once(dataset)
    logger.info("Dataset loaded once (%s); sweeping %d seeds live.", dataset, len(seeds))

    sweep_results = []
    for seed in seeds:
        logger.info("=" * 60)
        logger.info("SEED SWEEP — running seed=%d", seed)
        logger.info("=" * 60)

        result = evaluate_snn(
            cast(np.ndarray, data["features"]),
            cast(np.ndarray, data["labels_binary"]),
            cast(list[str], data["labels_class"]),
            warmup_features=cast(Optional[np.ndarray], data["warmup_features"]),
            context_keys=cast(Optional[np.ndarray], data["context_keys"]),
            warmup_context_keys=cast(Optional[np.ndarray], data["warmup_context_keys"]),
            seed=seed,
        )

        sweep_results.append({
            "seed": seed,
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "detection_rate": result["detection_rate"],
            "f1_score": result["f1_score"],
            "fpr": result["fpr"],
            "fnr": result["fnr"],
            "latency_p50_ms": result["latency_p50_ms"],
            "latency_p99_ms": result["latency_p99_ms"],
        })

        logger.info(
            "seed=%d -> accuracy=%.2f%% detection=%.2f%% fpr=%.4f%% f1=%.4f",
            seed, result["accuracy"] * 100, result["detection_rate"] * 100,
            result["fpr"] * 100, result["f1_score"],
        )

    return sweep_results


def summarise(sweep_results: list[dict]) -> dict:
    metrics = ["accuracy", "detection_rate", "fpr", "f1_score", "precision", "recall"]
    summary = {}
    for m in metrics:
        vals = np.array([r[m] for r in sweep_results])
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def print_summary(sweep_results: list[dict], summary: dict):
    print("\n" + "=" * 100)
    print(f"                    CORTIX SEED SWEEP — {len(sweep_results)} LIVE RUNS")
    print("=" * 100)
    print(f"{'Seed':>6} | {'Accuracy':>10} | {'Detection':>10} | {'FPR':>10} | {'F1':>8}")
    print("-" * 100)
    for r in sweep_results:
        print(
            f"{r['seed']:>6} | {r['accuracy']*100:>8.2f}% | "
            f"{r['detection_rate']*100:>8.2f}% | {r['fpr']*100:>8.4f}% | {r['f1_score']:>8.4f}"
        )
    print("-" * 100)
    print(f"{'MEAN':>6} | {summary['accuracy']['mean']*100:>8.2f}% | "
          f"{summary['detection_rate']['mean']*100:>8.2f}% | "
          f"{summary['fpr']['mean']*100:>8.4f}% | {summary['f1_score']['mean']:>8.4f}")
    print(f"{'STD':>6} | {summary['accuracy']['std']*100:>8.2f}% | "
          f"{summary['detection_rate']['std']*100:>8.2f}% | "
          f"{summary['fpr']['std']*100:>8.4f}% | {summary['f1_score']['std']:>8.4f}")
    print("=" * 100 + "\n")


def plot_sweep(sweep_results: list[dict], output_path: str):
    seeds = [r["seed"] for r in sweep_results]
    accuracy = [r["accuracy"] * 100 for r in sweep_results]
    detection = [r["detection_rate"] * 100 for r in sweep_results]
    f1 = [r["f1_score"] for r in sweep_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar([str(s) for s in seeds], accuracy, color="#3b82f6")
    axes[0].set_title("Accuracy by Seed", fontweight="bold")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].axhline(np.mean(accuracy), color="#ef4444", linestyle="--", label=f"Mean={np.mean(accuracy):.1f}%")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar([str(s) for s in seeds], detection, color="#10b981")
    axes[1].set_title("Detection Rate by Seed", fontweight="bold")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Detection Rate (%)")
    axes[1].axhline(np.mean(detection), color="#ef4444", linestyle="--", label=f"Mean={np.mean(detection):.1f}%")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar([str(s) for s in seeds], f1, color="#f59e0b")
    axes[2].set_title("F1 Score by Seed", fontweight="bold")
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("F1 Score")
    axes[2].axhline(np.mean(f1), color="#ef4444", linestyle="--", label=f"Mean={np.mean(f1):.3f}")
    axes[2].legend()
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.suptitle("CortiX Hebbian SNN — Sensitivity to Weight Initialisation (Live, N seeds)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Seed sweep plot saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CortiX Multi-Seed Benchmark Sweep")
    parser.add_argument("--dataset", type=str, default="nslkdd",
                         choices=["nslkdd", "cicids2017", "synthetic"])
    parser.add_argument("--tune_file", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation/results")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Explicit list of seeds, e.g. --seeds 1 2 3 4 5")
    parser.add_argument("--n_seeds", type=int, default=10,
                         help="If --seeds not given, sweep seeds 0..n_seeds-1")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    os.makedirs(args.output, exist_ok=True)

    sweep_results = run_sweep(args.dataset, seeds, tune_file=args.tune_file)
    summary = summarise(sweep_results)
    print_summary(sweep_results, summary)

    json_path = os.path.join(args.output, f"seed_sweep_{args.dataset}.json")
    with open(json_path, "w") as f:
        json.dump({"per_seed": sweep_results, "summary": summary}, f, indent=2)
    logger.info("Raw sweep results saved to %s", json_path)

    plot_sweep(sweep_results, os.path.join(args.output, f"seed_sweep_{args.dataset}.png"))
