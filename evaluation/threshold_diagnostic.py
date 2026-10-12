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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.evaluation.threshold_diagnostic")


def mode_aware_score(z_scores: np.ndarray, mode: str) -> np.ndarray:
    """
    Return a 'higher = more anomalous' score array regardless of mode,
    so downstream AUC/sweep logic can stay mode-agnostic after this call.
    """
    if mode == "lower":
        return -z_scores
    elif mode == "bilateral":
        return np.abs(z_scores)
    else:  # "upper"
        return z_scores


def predict_anomaly(z_scores: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    """Apply the SAME decision rule the live AnomalyScorer uses for this mode."""
    if mode == "lower":
        return (z_scores < threshold).astype(int)
    elif mode == "bilateral":
        return (np.abs(z_scores) > threshold).astype(int)
    else:  # "upper"
        return (z_scores > threshold).astype(int)


def load_dataset(dataset: str, holdout_class: Optional[str] = None):
    if dataset == "nslkdd":
        nsl_data = load_nslkdd_dataset(
            exclude_train_classes=[holdout_class] if holdout_class else None
        )
        y_train = nsl_data["y_train"]
        X_train_sel = nsl_data["X_train"]
        benign_train_mask = y_train == 0
        benign_train = X_train_sel[benign_train_mask]
        warmup_n = min(10000, len(benign_train))
        # Test split returned in full; holdout splitting is done in run_clean_holdout().
        X_test = nsl_data["X_test"]
        y_test = nsl_data["y_test"]
        y_test_class = list(nsl_data["y_test_class"])
        return dict(
            features=X_test,
            labels_binary=y_test,
            labels_class=y_test_class,
            warmup_features=benign_train[:warmup_n],
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


def sweep_thresholds(z_scores: np.ndarray, y_true: np.ndarray, mode: str, n_points: int = 60) -> list[dict]:
    """Evaluate detection/FPR/F1/accuracy across candidate thresholds, in the correct direction for `mode`."""
    lo, hi = np.percentile(z_scores, 1), np.percentile(z_scores, 99.5)
    candidates = np.linspace(lo, hi, n_points)

    rows = []
    for thresh in candidates:
        y_pred = predict_anomaly(z_scores, thresh, mode)
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


def _run_snn(data: dict, X: np.ndarray, y: np.ndarray, cls: list[str], seed: int) -> dict:
    """Fresh ensemble, same seed + same benign warmup -> identical weights every call."""
    return evaluate_snn(
        X, y, cls,
        warmup_features=cast(Optional[np.ndarray], data["warmup_features"]),
        context_keys=None, warmup_context_keys=None, seed=seed,
    )


def _metrics_at(z: np.ndarray, y: np.ndarray, thr: float, mode: str) -> dict:
    p = predict_anomaly(z, thr, mode)
    tp = int(np.sum((y == 1) & (p == 1))); fn = int(np.sum((y == 1) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1))); tn = int(np.sum((y == 0) & (p == 0)))
    return {"detection_rate": tp / (tp + fn) if tp + fn else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def run_clean_holdout(args, data: dict, holdout: str, original_thr: float) -> None:
    """
    Leave-one-attack-out zero-day test with ZERO exposure of `holdout`:
      - scaler fit:      train split with `holdout` rows removed (load_nslkdd_dataset)
      - SNN warmup:      benign-only train rows
      - threshold pick:  Phase 1 = BENIGN half A + all OTHER attack classes (best F1)
      - evaluation:      Phase 2 = BENIGN half B + ONLY `holdout`, at the frozen Phase-1 threshold
    Benign halves A/B are disjoint so Phase-2 FPR is not measured on selection data.
    """
    mode = config.ANOMALY_MODE
    X = cast(np.ndarray, data["features"])
    y = cast(np.ndarray, data["labels_binary"])
    cls = np.asarray(data["labels_class"])

    available = sorted(set(cls.tolist()))
    if holdout not in available or holdout == "BENIGN":
        raise SystemExit(f"holdout class {holdout!r} invalid; test labels: {available}")

    rng = np.random.default_rng(args.seed)
    benign_idx = rng.permutation(np.where(cls == "BENIGN")[0])
    half = len(benign_idx) // 2
    benign_sel, benign_eval = benign_idx[:half], benign_idx[half:]

    other_idx = np.where((cls != "BENIGN") & (cls != holdout))[0]
    hold_idx = np.where(cls == holdout)[0]
    sel_idx = np.sort(np.concatenate([benign_sel, other_idx]))
    eval_idx = np.sort(np.concatenate([benign_eval, hold_idx]))

    # Hard guarantees — abort rather than silently leak.
    assert holdout not in set(cls[sel_idx].tolist()), "LEAK: holdout class in selection set"
    assert len(np.intersect1d(sel_idx, eval_idx)) == 0, "LEAK: selection/eval overlap"
    assert set(cls[eval_idx].tolist()) == {"BENIGN", holdout}
    other_classes = sorted(set(cls[other_idx].tolist()))

    # ── Phase 1: threshold selection without the holdout class ──
    logger.info("PHASE 1 (selection): %d BENIGN-A + %d attacks from %s",
                len(benign_sel), len(other_idx), other_classes)
    r1 = _run_snn(data, X[sel_idx], y[sel_idx], cls[sel_idx].tolist(), args.seed)
    z1, y1 = r1["z_scores"], r1["y_true"]
    sweep = sweep_thresholds(z1, y1, mode, n_points=60)
    best_f1_row = max(sweep, key=lambda r: r["f1_score"])
    clean_thr_f1 = float(best_f1_row["threshold"])
    auc1 = float(roc_auc_score(y1, mode_aware_score(z1, mode)))

    # Also find thresholds for realistic FPR targets (5% and 10%) on Phase 1 data
    fpr_5_rows = [r for r in sweep if r["fpr"] <= 0.05]
    best_fpr_5_row = max(fpr_5_rows, key=lambda r: r["detection_rate"]) if fpr_5_rows else None

    fpr_10_rows = [r for r in sweep if r["fpr"] <= 0.10]
    best_fpr_10_row = max(fpr_10_rows, key=lambda r: r["detection_rate"]) if fpr_10_rows else None

    print("\n" + "=" * 100)
    print(f"PHASE 1 — THRESHOLD SELECTION (holdout '{holdout}' EXCLUDED)   mode={mode}")
    print("=" * 100)
    print(f"Selection set: {len(benign_sel)} BENIGN (half A) + {len(other_idx)} attacks {other_classes}")
    print(f"AUC on selection set = {auc1:.4f}")
    print(f"{'Threshold':>10} | {'Detection':>10} | {'FPR':>9} | {'F1':>7}")
    for row in sweep[::max(1, len(sweep) // 20)]:
        print(f"{row['threshold']:>10.3f} | {row['detection_rate']*100:>9.2f}% | "
              f"{row['fpr']*100:>8.4f}% | {row['f1_score']:>7.4f}")
    thr_10 = float(best_fpr_10_row["threshold"]) if best_fpr_10_row else None
    thr_5 = float(best_fpr_5_row["threshold"]) if best_fpr_5_row else None

    if best_fpr_10_row and thr_10 is not None:
        print(f"-> Strategy B (FPR <= 10%): Threshold = {thr_10:.4f} "
              f"(det={best_fpr_10_row['detection_rate']*100:.2f}%, fpr={best_fpr_10_row['fpr']*100:.4f}%)")
    if best_fpr_5_row and thr_5 is not None:
        print(f"-> Strategy C (FPR <= 5%):  Threshold = {thr_5:.4f} "
              f"(det={best_fpr_5_row['detection_rate']*100:.2f}%, fpr={best_fpr_5_row['fpr']*100:.4f}%)")

    # ── Phase 2: zero-day evaluation at the frozen threshold(s) ──
    config.ANOMALY_Z_THRESHOLD = clean_thr_f1
    logger.info("PHASE 2 (zero-day eval): %d BENIGN-B + %d %s at frozen Max-F1 threshold %.4f",
                len(benign_eval), len(hold_idx), holdout, clean_thr_f1)
    r2 = _run_snn(data, X[eval_idx], y[eval_idx], cls[eval_idx].tolist(), args.seed)
    z2, y2 = r2["z_scores"], r2["y_true"]
    
    m_f1 = _metrics_at(z2, y2, clean_thr_f1, mode)
    auc2 = float(roc_auc_score(y2, mode_aware_score(z2, mode)))
    ref = _metrics_at(z2, y2, original_thr, mode)

    m_fpr10 = _metrics_at(z2, y2, thr_10, mode) if thr_10 is not None else None
    m_fpr5 = _metrics_at(z2, y2, thr_5, mode) if thr_5 is not None else None

    print("\n" + "#" * 100)
    print(f"  ZERO-DAY HOLDOUT RESULT — {holdout}")
    print("#" * 100)
    print(f"  Exposure audit for '{holdout}':")
    print(f"    scaler fit ........... EXCLUDED (train rows of {holdout} dropped before MinMaxScaler.fit)")
    print(f"    SNN warmup ........... EXCLUDED (benign-only train rows)")
    print(f"    threshold selection .. EXCLUDED (Phase 1 classes: BENIGN + {other_classes})")
    print(f"    adaptive calibration . OFF")
    print(f"    benign overlap ....... NONE (Phase 1 and Phase 2 use disjoint benign halves)")
    print("-" * 100)
    print("  RESULTS AT FROZEN THRESHOLDS (derived in Phase 1, ZERO exposure to held-out attack):")
    print(f"  [1] Max-F1 Threshold ({clean_thr_f1:.4f}):")
    print(f"      {holdout} Detection Rate: {m_f1['detection_rate']*100:.2f}%   ({m_f1['tp']}/{m_f1['tp']+m_f1['fn']})")
    print(f"      BENIGN False Positive Rate: {m_f1['fpr']*100:.4f}% ({m_f1['fp']}/{m_f1['fp']+m_f1['tn']})")
    if m_fpr10 and thr_10 is not None:
        print(f"  [2] Target FPR <= 10% Threshold ({thr_10:.4f}):")
        print(f"      {holdout} Detection Rate: {m_fpr10['detection_rate']*100:.2f}%   ({m_fpr10['tp']}/{m_fpr10['tp']+m_fpr10['fn']})")
        print(f"      BENIGN False Positive Rate: {m_fpr10['fpr']*100:.4f}% ({m_fpr10['fp']}/{m_fpr10['fp']+m_fpr10['tn']})")
    if m_fpr5 and thr_5 is not None:
        print(f"  [3] Target FPR <= 5% Threshold ({thr_5:.4f}):")
        print(f"      {holdout} Detection Rate: {m_fpr5['detection_rate']*100:.2f}%   ({m_fpr5['tp']}/{m_fpr5['tp']+m_fpr5['fn']})")
        print(f"      BENIGN False Positive Rate: {m_fpr5['fpr']*100:.4f}% ({m_fpr5['fp']}/{m_fpr5['fp']+m_fpr5['tn']})")
    print(f"  AUC (benign vs {holdout}): {auc2:.4f}  (threshold-free separability measure)")
    print(f"  [reference only] legacy threshold {original_thr}: det={ref['detection_rate']*100:.2f}% "
          f"fpr={ref['fpr']*100:.4f}%")
    print("#" * 100 + "\n")

    out = {
        "holdout_class": holdout, "seed": args.seed, "anomaly_mode": mode,
        "clean_threshold_f1": clean_thr_f1, "legacy_threshold": original_thr,
        "exposure": {"scaler_fit": "excluded", "warmup": "benign_only",
                     "threshold_selection": "excluded", "adaptive_threshold": False,
                     "benign_split": "disjoint_halves"},
        "phase1": {"classes": ["BENIGN"] + other_classes, "n_benign": len(benign_sel),
                   "n_attack": len(other_idx), "auc": auc1, "best_f1": best_f1_row,
                   "best_fpr_10": best_fpr_10_row, "best_fpr_5": best_fpr_5_row, "sweep": sweep},
        "phase2": {"n_benign": len(benign_eval), "n_holdout": len(hold_idx),
                   "auc": auc2, "max_f1_results": m_f1, "target_fpr10_results": m_fpr10,
                   "target_fpr5_results": m_fpr5, "legacy_threshold_reference": ref},
    }
    path = os.path.join(args.output, f"zero_day_holdout_{holdout}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Zero-day holdout results saved to %s", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nslkdd", choices=["nslkdd", "synthetic"])
    parser.add_argument("--tune_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation/results")
    parser.add_argument("--holdout_class", type=str, default=None,
                        help="Leave-one-attack-out zero-day test: exclude this class from scaler fit, "
                             "warmup AND threshold selection; evaluate on BENIGN + this class only.")
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
            if "adaptive_threshold" in t:
                config.ADAPTIVE_THRESHOLD = t.get("adaptive_threshold", config.ADAPTIVE_THRESHOLD)
            logger.info("Loaded tuning config: %s", t)

    os.makedirs(args.output, exist_ok=True)

    holdout = args.holdout_class
    if holdout:
        if args.dataset != "nslkdd":
            raise SystemExit("--holdout_class is only supported with --dataset nslkdd")
        # No per-run recalibration in a zero-day test; threshold comes from Phase 1.
        config.ADAPTIVE_THRESHOLD = False
        logger.info("CLEAN HOLDOUT MODE: class=%s, adaptive calibration OFF", holdout)
        data = load_dataset(args.dataset, holdout_class=holdout)
        run_clean_holdout(args, data, holdout, original_thr=float(config.ANOMALY_Z_THRESHOLD))
        return

    mode = config.ANOMALY_MODE
    logger.info("Using ANOMALY_MODE=%s for scoring/sweep direction", mode)

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

    # ── 1. AUC: the ceiling on what ANY threshold can achieve, IN THIS MODE'S DIRECTION ──
    scored = mode_aware_score(z_scores, mode)
    auc = float(roc_auc_score(y_true, scored))

    print("\n" + "=" * 100)
    print(f"                    SEPARABILITY CHECK — z-score vs ground truth (mode={mode})")
    print("=" * 100)
    print(f"AUC = {auc:.4f}")
    if auc < 0.55:
        print("  -> Essentially NO separation in this direction. z-scores carry almost no")
        print("     signal about benign vs attack under this mode. No threshold choice will")
        print("     fix this -- try a different ANOMALY_MODE (see direction_check.py) or")
        print("     change the SNN's representation/warmup/config.")
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

    # ── 2. Percentile breakdown by class (raw z, not mode-transformed --
    #      this is about the underlying activation distribution, independent of mode) ──
    benign_z = z_scores[y_true == 0]
    attack_z = z_scores[y_true == 1]

    def pct_row(name, arr):
        p = np.percentile(arr, [5, 25, 50, 75, 95, 99])
        print(f"{name:<10} n={len(arr):>6} | p5={p[0]:>8.3f} p25={p[1]:>8.3f} "
              f"median={p[2]:>8.3f} p75={p[3]:>8.3f} p95={p[4]:>8.3f} p99={p[5]:>8.3f}")

    print("\nZ-SCORE DISTRIBUTION BY CLASS (raw z, mode-independent)")
    print("-" * 100)
    pct_row("BENIGN", benign_z)
    pct_row("ATTACK", attack_z)

    print(f"\nCurrent configured threshold: {config.ANOMALY_Z_THRESHOLD}  (mode={mode})")
    benign_flagged = predict_anomaly(benign_z, config.ANOMALY_Z_THRESHOLD, mode)
    attack_flagged = predict_anomaly(attack_z, config.ANOMALY_Z_THRESHOLD, mode)
    print(f"Benign z-scores flagged as anomalous (-> false positives): "
          f"{np.mean(benign_flagged) * 100:.4f}%")
    print(f"Attack z-scores flagged as anomalous (-> true positives):  "
          f"{np.mean(attack_flagged) * 100:.4f}%")

    # ── 3. Full threshold sweep, mode-aware ──
    sweep = sweep_thresholds(z_scores, y_true, mode, n_points=60)

    print("\n" + "=" * 100)
    print(f"THRESHOLD SWEEP (mode={mode}; sample of points; full data in JSON)")
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
            print("     Consider: more warmup samples, tuning eta, more hidden neurons,")
            print("     a different encoder config, or relaxing which criterion is primary.")
    else:
        print("No threshold in the swept range achieves FPR <= 0.3%.")
    print("=" * 100 + "\n")

    # ── 5. Save plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(benign_z, bins=60, alpha=0.6, label="Benign", color="#3b82f6", density=True)
    axes[0].hist(attack_z, bins=60, alpha=0.6, label="Attack", color="#ef4444", density=True)
    axes[0].axvline(config.ANOMALY_Z_THRESHOLD, color="black", linestyle="--",
                     label=f"Current threshold ({config.ANOMALY_Z_THRESHOLD:.3f}, mode={mode})")
    axes[0].set_xlabel("z-score (raw)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Z-Score Distribution by Class (AUC={auc:.3f}, mode={mode})", fontweight="bold")
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
    axes[1].set_title(f"Threshold Sweep (mode={mode})", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output, "threshold_diagnostic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Diagnostic plot saved to %s", plot_path)

    # Save raw sweep + summary JSON
    out = {
        "anomaly_mode": mode,
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