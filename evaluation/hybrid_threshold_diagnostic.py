"""
CortiX — Hybrid (Stage 2) Confidence Threshold Diagnostic

evaluate_hybrid() currently hardcodes `benign_prob > 0.85` as the cutoff for
overturning an SNN anomaly flag. That number was never tuned -- same mistake
we found and fixed for the SNN's ANOMALY_Z_THRESHOLD. This script runs
Stage 2 inference ONCE (live, real forward passes through the trained
CortixLSTMCNN checkpoint), collects the raw benign_prob for every
SNN-flagged event, then sweeps the cutoff to find the best achievable
operating point -- without retraining anything.

This answers: was 0.85 just the wrong cutoff (fixable in minutes), or is
even the best cutoff still far from the acceptance criteria (meaning the
classifier itself, or the sequence construction feeding it, needs work)?

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/hybrid_threshold_diagnostic.py \
        --dataset nslkdd --tune_file evaluation/tuned_params_nslkdd.json --seed 42
"""

import os
import json
import argparse
import logging
from typing import cast

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from cortix.config import config
from evaluation.benchmark import evaluate_snn, load_nslkdd_dataset
from evaluation.classifier_eval_utils import (
    load_nslkdd_features_for_classifier,
    get_benign_class_index,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.evaluation.hybrid_threshold_diagnostic")


def collect_benign_probs(snn_results: dict, classifier_features: np.ndarray,
                          benign_idx: int, model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Stage 2 inference once over every SNN-flagged event and return:
      - flagged_indices: positions in the eval set that the SNN flagged
      - benign_probs: the classifier's P(BENIGN) for each flagged event
    (Everything the SNN did NOT flag stays y_pred=0 regardless of Stage 2 --
    unchanged from evaluate_hybrid's existing behaviour.)
    """
    import torch
    from cortix.classifier.model import CortixLSTMCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = ckpt["fc2.bias"].shape[0] if "fc2.bias" in ckpt else config.CLASSIFIER_NUM_CLASSES
    num_features = ckpt["conv1.weight"].shape[1] if "conv1.weight" in ckpt else classifier_features.shape[1]

    # Infer seq_len from checkpoint
    if "pool.output_size" in ckpt:
        seq_len = max(1, int(ckpt["pool.output_size"]) * 2)
    elif "nslkdd" in model_path.lower():
        seq_len = 1
    else:
        seq_len = config.CLASSIFIER_SEQ_LEN

    model = CortixLSTMCNN(num_classes=num_classes, num_features=num_features, seq_len=seq_len).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    y_pred_snn = snn_results["y_pred"]
    n_classifier = len(classifier_features)

    flagged_indices = np.where(y_pred_snn == 1)[0]
    # Filter to valid indices within classifier_features length
    flagged_indices = flagged_indices[flagged_indices < n_classifier]
    benign_probs = np.zeros(len(flagged_indices), dtype=np.float32)

    logger.info("Running Stage 2 inference on %d SNN-flagged events (seq_len=%d)...", len(flagged_indices), seq_len)

    with torch.no_grad():
        for j, i in enumerate(flagged_indices):
            if seq_len == 1:
                seq_features = classifier_features[i:i + 1]  # shape (1, num_features)
            else:
                start_idx = max(0, i - seq_len + 1)
                seq_features = classifier_features[start_idx:i + 1]
                if len(seq_features) < seq_len:
                    padding = np.zeros(
                        (seq_len - len(seq_features), classifier_features.shape[1]),
                        dtype=np.float32,
                    )
                    seq_features = np.vstack([padding, seq_features])

            x_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=-1)
            benign_probs[j] = probs[0, benign_idx].item()

    return flagged_indices, benign_probs


def sweep_confidence_thresholds(
    y_true: np.ndarray, y_pred_snn: np.ndarray,
    flagged_indices: np.ndarray, benign_probs: np.ndarray,
    n_points: int = 50,
) -> list[dict]:
    """For each candidate confidence cutoff, recompute full-set metrics."""
    candidates = np.linspace(0.0, 1.0, n_points)
    rows = []

    for cutoff in candidates:
        y_pred = y_pred_snn.copy()
        overturn_mask = benign_probs > cutoff
        y_pred[flagged_indices[overturn_mask]] = 0

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        detection = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        rows.append({
            "cutoff": float(cutoff), "accuracy": acc, "detection_rate": detection,
            "fpr": fpr, "f1_score": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nslkdd", choices=["nslkdd"])
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
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))
            logger.info("Loaded tuning config: %s", t)

    os.makedirs(args.output, exist_ok=True)

    # 1. Live SNN pass (identical to benchmark.py's nslkdd path)
    nsl_data = load_nslkdd_dataset()
    y_train = nsl_data["y_train"]
    X_train_sel = nsl_data["X_train"]
    benign_train_mask = y_train == 0
    benign_train = X_train_sel[benign_train_mask]
    warmup_n = min(10000, len(benign_train))

    snn_results = evaluate_snn(
        cast(np.ndarray, nsl_data["X_test"]),
        cast(np.ndarray, nsl_data["y_test"]),
        cast(list[str], nsl_data["y_test_class"]),
        warmup_features=benign_train[:warmup_n],
        seed=args.seed,
    )

    # 2. Load classifier-consistent NSL-KDD features (StandardScaler, not tanh/MinMax)
    classifier_features = load_nslkdd_features_for_classifier()
    if classifier_features is None:
        logger.error("Classifier features unavailable -- train Stage 2 on NSL-KDD first.")
        return

    benign_idx = get_benign_class_index()
    if benign_idx is None:
        logger.error("Could not resolve BENIGN class index.")
        return

    model_path = config.MODEL_PATH_NSLKDD
    if not os.path.exists(model_path):
        logger.error("No checkpoint at %s -- train Stage 2 first.", model_path)
        return

    # 3. Collect real benign_prob for every SNN-flagged event, ONCE
    flagged_indices, benign_probs = collect_benign_probs(
        snn_results, classifier_features, benign_idx, model_path
    )

    y_true = snn_results["y_true"]
    y_pred_snn = snn_results["y_pred"]

    # ── Baseline (SNN alone) for reference ──
    snn_fp = int(np.sum((y_true == 0) & (y_pred_snn == 1)))
    snn_tn = int(np.sum((y_true == 0) & (y_pred_snn == 0)))
    snn_tp = int(np.sum((y_true == 1) & (y_pred_snn == 1)))
    snn_fn = int(np.sum((y_true == 1) & (y_pred_snn == 0)))
    print("\n" + "=" * 100)
    print("BASELINE (SNN pre-filter alone, before Stage 2 overturns anything)")
    print("=" * 100)
    print(f"Detection: {snn_tp/(snn_tp+snn_fn)*100:.2f}%  FPR: {snn_fp/(snn_fp+snn_tn)*100:.4f}%  "
          f"({len(flagged_indices)} events flagged for Stage 2 review)")

    # 4. Sweep confidence cutoffs
    sweep = sweep_confidence_thresholds(y_true, y_pred_snn, flagged_indices, benign_probs)

    print("\n" + "=" * 100)
    print("STAGE 2 CONFIDENCE CUTOFF SWEEP (benign_prob > cutoff -> overturn to benign)")
    print("=" * 100)
    print(f"{'Cutoff':>8} | {'Accuracy':>9} | {'Detection':>10} | {'FPR':>10} | {'F1':>7}")
    print("-" * 100)
    for row in sweep[::max(1, len(sweep) // 25)]:
        print(f"{row['cutoff']:>8.3f} | {row['accuracy']*100:>8.2f}% | "
              f"{row['detection_rate']*100:>9.2f}% | {row['fpr']*100:>9.4f}% | {row['f1_score']:>7.4f}")

    best_f1_row = max(sweep, key=lambda r: r["f1_score"])
    print("\n" + "-" * 100)
    print(f"BEST F1 cutoff = {best_f1_row['cutoff']:.3f}  ->  "
          f"accuracy={best_f1_row['accuracy']*100:.2f}%  detection={best_f1_row['detection_rate']*100:.2f}%  "
          f"fpr={best_f1_row['fpr']*100:.4f}%  f1={best_f1_row['f1_score']:.4f}")

    fpr_ok = [r for r in sweep if r["fpr"] <= 0.003]
    if fpr_ok:
        best_under_fpr = max(fpr_ok, key=lambda r: r["detection_rate"])
        print(f"BEST under FPR<=0.3% = cutoff {best_under_fpr['cutoff']:.3f}  ->  "
              f"detection={best_under_fpr['detection_rate']*100:.2f}%  "
              f"fpr={best_under_fpr['fpr']*100:.4f}%  f1={best_under_fpr['f1_score']:.4f}")
        if best_under_fpr["detection_rate"] < 0.99:
            print("  -> Even the best FPR<=0.3% cutoff does not reach 99% detection.")
            print("     This points to a ceiling in the classifier/sequence construction,")
            print("     not just cutoff selection -- worth reporting as a measured limit.")
    else:
        print("No cutoff in [0,1] achieves FPR <= 0.3% -- Stage 2 cannot hit that target")
        print("alone at this SNN pre-filter operating point. Report the best F1 trade-off")
        print("as your honest operating point, and discuss the ceiling explicitly.")
    print("=" * 100 + "\n")

    # 5. Plot + save
    fig, ax = plt.subplots(figsize=(10, 6))
    cutoffs = [r["cutoff"] for r in sweep]
    ax.plot(cutoffs, [r["detection_rate"] * 100 for r in sweep], label="Detection Rate (%)", color="#10b981")
    ax.plot(cutoffs, [r["fpr"] * 100 for r in sweep], label="FPR (%)", color="#ef4444")
    ax.plot(cutoffs, [r["f1_score"] * 100 for r in sweep], label="F1 x100", color="#f59e0b")
    ax.axvline(0.85, color="black", linestyle="--", alpha=0.5, label="Original hardcoded cutoff (0.85)")
    ax.set_xlabel("Stage 2 benign_prob cutoff")
    ax.set_ylabel("%")
    ax.set_title("Hybrid Stage 2 — Confidence Cutoff Sweep", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output, "hybrid_threshold_diagnostic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to %s", plot_path)

    json_path = os.path.join(args.output, "hybrid_threshold_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump({
            "snn_baseline": {"detection_rate": snn_tp/(snn_tp+snn_fn), "fpr": snn_fp/(snn_fp+snn_tn),
                              "flagged_count": int(len(flagged_indices))},
            "best_f1": best_f1_row,
            "best_under_fpr_0.3pct": (max(fpr_ok, key=lambda r: r["detection_rate"]) if fpr_ok else None),
            "sweep": sweep,
        }, f, indent=2)
    logger.info("Full sweep data saved to %s", json_path)


if __name__ == "__main__":
    main()
