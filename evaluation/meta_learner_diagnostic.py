"""
CortiX — Meta-Learner (Stacking) Diagnostic

Tests a specific idea: instead of retraining the SNN or the classifier,
learn a small combiner ON TOP of their existing (frozen) outputs --
[SNN z_score, Stage 2 benign_prob] -> P(attack) -- for events the SNN
already flagged. Everything the SNN did NOT flag stays negative, same as
the existing cascade design; this only changes how flagged events get
adjudicated.

Why this might help: the SNN and Stage 2 each make an independent
threshold decision on their OWN score alone. If their two scores disagree
usefully on the ambiguous boundary cases (even if neither alone separates
cleanly there), a learned 2D combiner can sometimes carve out a decision
boundary neither 1D threshold could reach. This is standard "stacked
generalization" -- not a new heavy model, just a 2-parameter (+intercept)
logistic regression on two numbers.

Honesty safeguards:
  - Uses cross_val_predict (5-fold, out-of-fold) so every flagged event's
    predicted probability comes from a fold that never saw its label --
    no train/test leakage inflating the result.
  - Reports the correlation between the two input scores directly: if
    they're highly correlated, combining them can't help much, and this
    script will show that plainly rather than let a lucky split hide it.
  - Sweeps the combiner's own threshold exactly like the two earlier
    diagnostics, so results are directly comparable to the SNN-alone and
    Stage-2-alone sweeps already on file.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/meta_learner_diagnostic.py \
        --dataset nslkdd --tune_file evaluation/tuned_params_nslkdd.json --seed 42
"""

import os
import json
import argparse
import logging
from typing import cast, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from cortix.config import config
from evaluation.benchmark import evaluate_snn, load_nslkdd_dataset
from evaluation.classifier_eval_utils import (
    load_nslkdd_features_for_classifier,
    get_benign_class_index,
)
from evaluation.hybrid_threshold_diagnostic import collect_benign_probs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.evaluation.meta_learner_diagnostic")


def sweep_meta_thresholds(y_true_full, y_pred_snn, flagged_indices, meta_probs, n_points=50):
    candidates = np.linspace(0.0, 1.0, n_points)
    rows = []
    for cutoff in candidates:
        y_pred = np.zeros_like(y_pred_snn)
        keep_mask = meta_probs > cutoff  # meta_probs = P(attack); keep flag if model agrees it's likely attack
        y_pred[flagged_indices[keep_mask]] = 1

        tp = int(np.sum((y_true_full == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true_full == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true_full == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true_full == 1) & (y_pred == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        detection = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = float(f1_score(y_true_full, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true_full, y_pred))
        rows.append({"cutoff": float(cutoff), "accuracy": acc, "detection_rate": detection,
                     "fpr": fpr, "f1_score": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
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

    # 1. Live SNN pass
    nsl_data = load_nslkdd_dataset()
    y_train = nsl_data["y_train"]
    X_train_sel = nsl_data["X_train"]
    benign_train_mask = y_train == 0
    benign_train = X_train_sel[benign_train_mask]
    warmup_n = min(10000, len(benign_train))

    snn_results = evaluate_snn(
        cast(Any, nsl_data["X_test"]),
        cast(Any, nsl_data["y_test"]),
        cast(list[str], nsl_data["y_test_class"]),
        warmup_features=benign_train[:warmup_n],
        seed=args.seed,
    )

    y_true = snn_results["y_true"]
    y_pred_snn = snn_results["y_pred"]
    z_scores = snn_results["z_scores"]

    # 2. Stage 2 benign_prob for every SNN-flagged event
    classifier_features = load_nslkdd_features_for_classifier()
    if classifier_features is None:
        logger.error("Classifier features unavailable -- train Stage 2 first.")
        return
    benign_idx = get_benign_class_index()
    if benign_idx is None:
        logger.error("Could not resolve BENIGN class index.")
        return
    model_path = config.MODEL_PATH_NSLKDD
    if not os.path.exists(model_path):
        logger.error("No checkpoint at %s.", model_path)
        return

    flagged_indices, benign_probs = collect_benign_probs(
        snn_results, classifier_features, benign_idx, model_path
    )
    attack_probs = 1.0 - benign_probs  # P(attack) from Stage 2, for a consistent "higher = more suspicious" direction

    z_flagged = z_scores[flagged_indices]
    y_flagged = y_true[flagged_indices]

    # ── Sanity check: are the two signals actually independent enough to combine usefully? ──
    corr = float(np.corrcoef(z_flagged, attack_probs)[0, 1])
    print("\n" + "=" * 100)
    print("SIGNAL INDEPENDENCE CHECK")
    print("=" * 100)
    print(f"Pearson correlation between SNN z-score and Stage-2 P(attack), on flagged events: {corr:.4f}")
    if abs(corr) > 0.8:
        print("  -> Highly correlated. The two signals carry mostly redundant information;")
        print("     combining them is unlikely to meaningfully beat either alone.")
    elif abs(corr) > 0.4:
        print("  -> Moderately correlated. Some redundancy, but room for the combiner to help.")
    else:
        print("  -> Low correlation. The two signals are largely independent -- good sign")
        print("     that combining them could genuinely improve separation.")
    print("=" * 100)

    # 3. Fit the combiner with OUT-OF-FOLD predictions (no leakage)
    X = np.column_stack([z_flagged, attack_probs])
    y = y_flagged

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    cv_preds = cast(np.ndarray, cross_val_predict(
        LogisticRegression(class_weight="balanced"), cast(cast(Any, np.ndarray), X), cast(cast(Any, np.ndarray), y),
        cv=skf, method="predict_proba"
    ))
    meta_probs = cv_preds[:, 1]  # P(attack)

    meta_auc = float(roc_auc_score(y, meta_probs))
    snn_only_auc = float(roc_auc_score(y, z_flagged))
    stage2_only_auc = float(roc_auc_score(y, attack_probs))

    print("\n" + "=" * 100)
    print("AUC COMPARISON ON FLAGGED SUBSET (out-of-fold for the combiner)")
    print("=" * 100)
    print(f"SNN z-score alone:      AUC = {snn_only_auc:.4f}")
    print(f"Stage 2 P(attack) alone: AUC = {stage2_only_auc:.4f}")
    print(f"Combiner [z, p_attack]:  AUC = {meta_auc:.4f}")
    if meta_auc <= max(snn_only_auc, stage2_only_auc) + 0.01:
        print("  -> Combiner does NOT meaningfully beat the better single signal.")
        print("     The two scores are too redundant on this data for stacking to help.")
    else:
        print("  -> Combiner beats both single signals -- genuine complementary information found.")
    print("=" * 100)

    # 4. Fit final combiner on ALL flagged data (deployable weights) for reporting
    final_model = LogisticRegression(class_weight="balanced")
    final_model.fit(X, y)
    print(f"\nLearned combiner: P(attack) = sigmoid({final_model.coef_[0][0]:.4f} * z_score + "
          f"{final_model.coef_[0][1]:.4f} * p_attack_stage2 + {final_model.intercept_[0]:.4f})")

    # 5. Sweep the combiner's threshold on the FULL test set (comparable to earlier diagnostics)
    sweep = sweep_meta_thresholds(y_true, y_pred_snn, flagged_indices, meta_probs)

    print("\n" + "=" * 100)
    print("META-LEARNER THRESHOLD SWEEP (full test set, out-of-fold predictions)")
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
    else:
        print("No cutoff achieves FPR <= 0.3% even with the combiner.")
    print("=" * 100 + "\n")

    # 6. Plot: sweep curve + score scatter colored by true label
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cutoffs = [r["cutoff"] for r in sweep]
    axes[0].plot(cutoffs, [r["detection_rate"] * 100 for r in sweep], label="Detection Rate (%)", color="#10b981")
    axes[0].plot(cutoffs, [r["fpr"] * 100 for r in sweep], label="FPR (%)", color="#ef4444")
    axes[0].plot(cutoffs, [r["f1_score"] * 100 for r in sweep], label="F1 x100", color="#f59e0b")
    axes[0].set_xlabel("Meta-learner P(attack) cutoff")
    axes[0].set_ylabel("%")
    axes[0].set_title("Meta-Learner Threshold Sweep", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    benign_mask = y_flagged == 0
    axes[1].scatter(z_flagged[benign_mask], attack_probs[benign_mask], alpha=0.3, s=8,
                     color="#3b82f6", label="Benign (false alarm)")
    axes[1].scatter(z_flagged[~benign_mask], attack_probs[~benign_mask], alpha=0.3, s=8,
                     color="#ef4444", label="Attack (true positive)")
    axes[1].set_xlabel("SNN z-score")
    axes[1].set_ylabel("Stage 2 P(attack)")
    axes[1].set_title(f"Flagged Events in 2D Score Space (corr={corr:.2f})", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output, "meta_learner_diagnostic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to %s", plot_path)

    json_path = os.path.join(args.output, "meta_learner_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump({
            "correlation_z_vs_attack_prob": corr,
            "auc_snn_only": snn_only_auc,
            "auc_stage2_only": stage2_only_auc,
            "auc_combiner": meta_auc,
            "learned_weights": {
                "z_score_coef": float(final_model.coef_[0][0]),
                "p_attack_coef": float(final_model.coef_[0][1]),
                "intercept": float(final_model.intercept_[0]),
            },
            "best_f1": best_f1_row,
            "best_under_fpr_0.3pct": (max(fpr_ok, key=lambda r: r["detection_rate"]) if fpr_ok else None),
            "sweep": sweep,
        }, f, indent=2)
    logger.info("Full results saved to %s", json_path)


if __name__ == "__main__":
    main()
