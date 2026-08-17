"""
CortiX — Threshold / Separability Diagnostic

Answers the question the seed sweep raised: is 43.26% accuracy / 0.64%
detection a THRESHOLD problem (fixable by picking a different
ANOMALY_Z_THRESHOLD) or a SEPARABILITY problem (the SNN's activation
signal doesn't distinguish benign from attack traffic well enough, no
matter where you cut)?

This is a single LIVE run (seed sweep already proved seed doesn't matter,
so one run's z-scores are representative). Nothing here is hardcoded or
simulated -- it reuses evaluate_snn() exactly as benchmark.py does, then
does extra analysis on the real z_scores/y_true it returns.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/threshold_diagnostic.py \
        --dataset nslkdd --tune_file evaluation/tuned_params_nslkdd.json --seed 42

Output:
    1. AUC of z_scores vs ground truth (the ceiling on what ANY threshold
       can achieve with this representation).
    2. Percentile breakdown of z-scores for benign vs attack samples
       (where's the overlap?).
    3. A full threshold sweep: for each candidate threshold, the resulting
       accuracy / detection / FPR / F1, with the acceptance-criteria
       columns flagged PASS/FAIL exactly like benchmark.py's report.
    4. The best achievable threshold under two objectives:
         (a) max F1
         (b) tightest threshold that still keeps FPR <= 0.3%, reporting
             what detection rate that actually buys you.
    5. Saves a z-score distribution plot (benign vs attack histograms)
       and a threshold-sweep curve, evaluation/results/threshold_diagnostic.png
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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from cortix.config import config
from evaluation.benchmark import (
    evaluate_snn,
    generate_synthetic_flows,
    load_nslkdd_dataset,
    load_cicids2017_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.evaluation.threshold_diagnostic")


def load_dataset(dataset: str):
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
        X_test_sel, X_test_raw, y_test, y_test_class = load_cicids2017_dataset(max_samples=0)
        # Take up to 10000 benign samples for warmup (same as NSL-KDD)
        benign_indices = np.where(y_test == 0)[0]
        warmup_n = min(10000, len(benign_indices))
        warmup_features = X_test_sel[benign_indices[:warmup_n]]
        return dict(
            features=X_test_sel,
            labels_binary=y_test,
            labels_class=y_test_class,
            warmup_features=warmup_features,
            context_keys=None,
            warmup_context_keys=None,
        )
    else:
        features, labels_binary, labels_class = generate_synthetic_flows(
            num_benign=3000, num_attack=1000
        )
        return dict(
            features=features, labels_binary=labels_binary, labels_class=labels_class,
            warmup_features=None, context_keys=None, warmup_context_keys=None,
        )


def sweep_thresholds(z_scores: np.ndarray, y_true: np.ndarray, n_points: int = 60) -> list[dict]:
    """Evaluate detection/FPR/F1/accuracy across a range of candidate thresholds.
    
    When bilateral=True, uses |z| > threshold (two-tailed) instead of z > threshold.
    """
    bilateral = config.ANOMALY_MODE == "bilateral"
    signal = np.abs(z_scores) if bilateral else z_scores
    lo, hi = np.percentile(signal, 1), np.percentile(signal, 99.5)
    candidates = np.linspace(lo, hi, n_points)

    rows = []
    for thresh in candidates:
        y_pred = (signal > thresh).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        detection = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        rows.append({
            "threshold": float(thresh), "accuracy": acc, "detection_rate": detection,
            "fpr": fpr, "f1_score": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nslkdd", choices=["nslkdd", "cicids2017", "synthetic"])
    parser.add_argument("--tune_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation/results")
    args = parser.parse_args()

    if args.tune_file and os.path.exists(args.tune_file):
        with open(args.tune_file, "r") as f:
            t = json.load(f)
            config.SLIDING_WINDOW_SIZE = t.get("window_size", config.SLIDING_WINDOW_SIZE)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.HEBBIAN_LR = t.get("learning_rate", t.get("eta", config.HEBBIAN_LR))
            config.METAPLASTICITY_ALPHA = t.get("meta_alpha", config.METAPLASTICITY_ALPHA)
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))
            config.ANOMALY_MODE = t.get("anomaly_mode", config.ANOMALY_MODE)
            logger.info("Loaded tuning config: %s", t)

    os.makedirs(args.output, exist_ok=True)

    data = load_dataset(args.dataset)
    logger.info("Running live SNN evaluation (seed=%d) to collect real z-scores...", args.seed)

    result = evaluate_snn(
        cast(np.ndarray, data["features"]),
        cast(np.ndarray, data["labels_binary"]),
        cast(list[str], data["labels_class"]),
        warmup_features=cast(Optional[np.ndarray], data["warmup_features"]),
        context_keys=cast(Optional[np.ndarray], data["context_keys"]),
        warmup_context_keys=cast(Optional[np.ndarray], data["warmup_context_keys"]),
        seed=args.seed,
    )

    z_scores = result["z_scores"]
    y_true = result["y_true"]

    # For bilateral mode, use |z| as the scoring signal
    bilateral = config.ANOMALY_MODE == "bilateral"
    scoring_signal = np.abs(z_scores) if bilateral else z_scores

    # ── 1. AUC: the ceiling on what ANY threshold can achieve ──
    auc = float(roc_auc_score(y_true, scoring_signal))

    print("\n" + "=" * 100)
    print("                    SEPARABILITY CHECK — z-score vs ground truth")
    print("=" * 100)
    print(f"AUC = {auc:.4f}")
    if auc < 0.55:
        print("  -> Essentially NO separation. z-scores carry almost no signal about")
        print("     benign vs attack. No threshold choice will fix this -- the SNN's")
        print("     representation/warmup/config needs to change, not the cutoff.")
    elif auc < 0.70:
        print("  -> Weak separation. Some signal exists; threshold tuning will help")
        print("     somewhat, but hitting 99% detection at 0.3% FPR is unlikely without")
        print("     also improving the representation (more warmup, different eta, etc).")
    elif auc < 0.90:
        print("  -> Moderate-to-good separation. Threshold tuning should meaningfully")
        print("     improve results; whether it clears the strict acceptance criteria")
        print("     depends on the specific overlap -- see the sweep below.")
    else:
        print("  -> Strong separation. If current results are poor, it's very likely")
        print("     a threshold-selection problem, not a representation problem.")
    print("=" * 100)

    # ── 2. Percentile breakdown by class ──
    benign_z = z_scores[y_true == 0]
    attack_z = z_scores[y_true == 1]

    def pct_row(name, arr):
        p = np.percentile(arr, [5, 25, 50, 75, 95, 99])
        print(f"{name:<10} n={len(arr):>6} | p5={p[0]:>8.3f} p25={p[1]:>8.3f} "
              f"median={p[2]:>8.3f} p75={p[3]:>8.3f} p95={p[4]:>8.3f} p99={p[5]:>8.3f}")

    print("\nZ-SCORE DISTRIBUTION BY CLASS")
    print("-" * 100)
    pct_row("BENIGN", benign_z)
    pct_row("ATTACK", attack_z)
    print(f"\nCurrent configured threshold: {config.ANOMALY_Z_THRESHOLD}")
    print(f"Benign z-scores exceeding current threshold (-> false positives): "
          f"{np.mean(benign_z > config.ANOMALY_Z_THRESHOLD) * 100:.4f}%")
    print(f"Attack z-scores exceeding current threshold (-> true positives):  "
          f"{np.mean(attack_z > config.ANOMALY_Z_THRESHOLD) * 100:.4f}%")

    # ── 3. Full threshold sweep ──
    sweep = sweep_thresholds(z_scores, y_true, n_points=60)

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP (sample of points; full data in JSON)")
    print("=" * 100)
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Detection':>10} | {'FPR':>9} | {'F1':>7}")
    print("-" * 100)
    for row in sweep[::max(1, len(sweep) // 20)]:
        print(f"{row['threshold']:>10.3f} | {row['accuracy']*100:>8.2f}% | "
              f"{row['detection_rate']*100:>9.2f}% | {row['fpr']*100:>8.4f}% | {row['f1_score']:>7.4f}")

    # ── 4a. Best F1 ──
    best_f1_row = max(sweep, key=lambda r: r["f1_score"])
    print("\n" + "-" * 100)
    print(f"BEST F1 threshold = {best_f1_row['threshold']:.3f}  ->  "
          f"accuracy={best_f1_row['accuracy']*100:.2f}%  detection={best_f1_row['detection_rate']*100:.2f}%  "
          f"fpr={best_f1_row['fpr']*100:.4f}%  f1={best_f1_row['f1_score']:.4f}")

    # ── 4b. Tightest threshold keeping FPR <= 0.3% ──
    fpr_ok = [r for r in sweep if r["fpr"] <= 0.003]
    if fpr_ok:
        best_under_fpr = max(fpr_ok, key=lambda r: r["detection_rate"])
        print(f"BEST under FPR<=0.3% = threshold {best_under_fpr['threshold']:.3f}  ->  "
              f"detection={best_under_fpr['detection_rate']*100:.2f}%  "
              f"fpr={best_under_fpr['fpr']*100:.4f}%  f1={best_under_fpr['f1_score']:.4f}")
        if best_under_fpr["detection_rate"] < 0.99:
            print("  -> Even the best FPR<=0.3% threshold does NOT reach 99% detection.")
            print("     Acceptance criteria as currently defined are not jointly achievable")
            print("     with this representation. Consider: more warmup samples, tuning eta,")
            print("     more hidden neurons, or relaxing which criterion is primary.")
    else:
        print("No threshold in the swept range achieves FPR <= 0.3%.")
    print("=" * 100 + "\n")

    # ── 5. Save plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(benign_z, bins=60, alpha=0.6, label="Benign", color="#3b82f6", density=True)
    axes[0].hist(attack_z, bins=60, alpha=0.6, label="Attack", color="#ef4444", density=True)
    axes[0].axvline(config.ANOMALY_Z_THRESHOLD, color="black", linestyle="--",
                     label=f"Current threshold ({config.ANOMALY_Z_THRESHOLD})")
    axes[0].set_xlabel("z-score")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Z-Score Distribution by Class (AUC={auc:.3f})", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    thresholds = [r["threshold"] for r in sweep]
    detections = [r["detection_rate"] * 100 for r in sweep]
    fprs = [r["fpr"] * 100 for r in sweep]
    f1s = [r["f1_score"] * 100 for r in sweep]

    axes[1].plot(thresholds, detections, label="Detection Rate (%)", color="#10b981")
    axes[1].plot(thresholds, fprs, label="FPR (%)", color="#ef4444")
    axes[1].plot(thresholds, f1s, label="F1 x100", color="#f59e0b")
    axes[1].axvline(config.ANOMALY_Z_THRESHOLD, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("%")
    axes[1].set_title("Threshold Sweep", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output, "threshold_diagnostic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Diagnostic plot saved to %s", plot_path)

    # Save raw sweep + summary JSON
    out = {
        "auc": auc,
        "current_threshold": config.ANOMALY_Z_THRESHOLD,
        "best_f1": best_f1_row,
        "best_under_fpr_0.3pct": (max(fpr_ok, key=lambda r: r["detection_rate"]) if fpr_ok else None),
        "sweep": sweep,
    }
    json_path = os.path.join(args.output, "threshold_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Full sweep data saved to %s", json_path)


if __name__ == "__main__":
    main()
