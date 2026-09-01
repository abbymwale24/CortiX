
import os
import json
import argparse
import logging
import time
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from cortix.config import config
from evaluation.benchmark import evaluate_snn, load_nslkdd_dataset

logging.basicConfig(level=logging.WARNING)  # quiet -- this sweep prints its own summary
logger = logging.getLogger("cortix.evaluation.representation_sweep")


def mode_aware_score(z_scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "lower":
        return -z_scores
    elif mode == "bilateral":
        return np.abs(z_scores)
    else:
        return z_scores


def run_one(features, labels_binary, labels_class, warmup_features, seed, eta) -> dict:
    config.HEBBIAN_LR = eta
    t0 = time.time()
    result = evaluate_snn(
        features, labels_binary, labels_class,
        warmup_features=warmup_features,
        seed=seed,
    )
    elapsed = time.time() - t0
    scored = mode_aware_score(result["z_scores"], config.ANOMALY_MODE)
    auc = float(roc_auc_score(result["y_true"], scored))
    return {"auc": auc, "elapsed_sec": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_n_stage1", type=int, default=10000)
    parser.add_argument("--output", type=str, default="evaluation/results")
    args = parser.parse_args()

    if args.tune_file and os.path.exists(args.tune_file):
        with open(args.tune_file, "r") as f:
            t = json.load(f)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))

    print(f"ANOMALY_MODE = {config.ANOMALY_MODE}  (fix should already be in place)")
    os.makedirs(args.output, exist_ok=True)

    print("Loading NSL-KDD (full dataset)...")
    nsl_data = load_nslkdd_dataset()
    y_train = nsl_data["y_train"]
    X_train_sel = nsl_data["X_train"]
    benign_train_mask = y_train == 0
    benign_train = X_train_sel[benign_train_mask]
    total_benign_available = len(benign_train)
    print(f"Total benign training samples available: {total_benign_available}")

    features = nsl_data["X_test"]
    labels_binary = nsl_data["y_test"]
    labels_class = nsl_data["y_test_class"]

    # ── Stage 1: eta sweep at fixed warmup_n ──
    warmup_n1 = min(args.warmup_n_stage1, total_benign_available)
    warmup_features1 = benign_train[:warmup_n1]

    eta_candidates = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01]
    stage1_results = []

    print("\n" + "=" * 80)
    print(f"STAGE 1: eta sweep (warmup_n={warmup_n1} fixed)")
    print("=" * 80)
    for eta in eta_candidates:
        r = run_one(features, labels_binary, labels_class, warmup_features1, args.seed, eta)
        stage1_results.append({"eta": eta, **r})
        print(f"  eta={eta:<8} -> AUC={r['auc']:.4f}  ({r['elapsed_sec']:.1f}s)")

    best_eta_row = max(stage1_results, key=lambda r: r["auc"])
    best_eta = best_eta_row["eta"]
    print(f"\nBest eta: {best_eta} (AUC={best_eta_row['auc']:.4f})")

    # ── Stage 2: warmup_n sweep at best eta ──
    warmup_candidates = sorted(set([
        min(5000, total_benign_available),
        min(10000, total_benign_available),
        min(20000, total_benign_available),
        min(40000, total_benign_available),
        total_benign_available,
    ]))
    stage2_results = []

    print("\n" + "=" * 80)
    print(f"STAGE 2: warmup_n sweep (eta={best_eta} fixed, best from Stage 1)")
    print("=" * 80)
    for wn in warmup_candidates:
        warmup_features2 = benign_train[:wn]
        r = run_one(features, labels_binary, labels_class, warmup_features2, args.seed, best_eta)
        stage2_results.append({"warmup_n": wn, **r})
        print(f"  warmup_n={wn:<7} -> AUC={r['auc']:.4f}  ({r['elapsed_sec']:.1f}s)")

    best_warmup_row = max(stage2_results, key=lambda r: r["auc"])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Best eta (Stage 1):      eta={best_eta}, warmup_n={warmup_n1}, AUC={best_eta_row['auc']:.4f}")
    print(f"Best warmup_n (Stage 2): eta={best_eta}, warmup_n={best_warmup_row['warmup_n']}, "
          f"AUC={best_warmup_row['auc']:.4f}")
    print("=" * 80)
    print("\nNext step: take the winning (eta, warmup_n) combo and run")
    print("threshold_diagnostic.py / hybrid diagnostics with those values wired")
    print("into your tune file, to see the actual detection/FPR tradeoff, not just AUC.")

    json_path = os.path.join(args.output, "representation_sweep.json")
    with open(json_path, "w") as f:
        json.dump({"stage1_eta_sweep": stage1_results, "stage2_warmup_sweep": stage2_results}, f, indent=2)
    print(f"\nFull results saved to {json_path}")


if __name__ == "__main__":
    main()