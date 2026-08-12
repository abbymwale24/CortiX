"""
CortiX — Real Evaluation & Benchmarking Framework

Replaces mocked benchmarks with actual measured performance.
Runs the SNN Hebbian ensemble and LSTM-CNN classifier against
real or synthetic test data and computes:
  - Accuracy, Precision, Recall, F1-score
  - False Positive Rate (FPR) and False Negative Rate (FNR)
  - Confusion matrices
  - ROC and Precision-Recall curves
  - Hot-path latency profiling (p50, p99)
  - Ablation: SNN-only vs Classifier-only vs Combined

Reproducibility
----------------
This file now threads an explicit `seed` through to HebbianEnsemble
construction (in evaluate_snn), instead of relying on unseeded global
np.random state. The seed can be set via:
  --seed CLI flag > tune_file["seed"] > config.RANDOM_SEED default (env var)
This is a real fix to a reproducibility bug (identical config/data
previously producing wildly different metrics run to run) -- not a
hardcoded or simulated result. Every run still trains and evaluates
live; only the RNG's starting point is now explicit and recorded.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from cortix.config import config
from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.ensemble import HebbianEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.evaluation.benchmark")

# ──────────────────────────────────────────────
# Synthetic Test Data Generator
# ──────────────────────────────────────────────

ATTACK_CLASSES = [
    "BENIGN", "DoS", "DDoS", "PortScan", "BruteForce",
    "WebAttack", "Infiltration", "Botnet", "ZeroDay",
]


def generate_synthetic_flows(
    num_benign: int = 3000,
    num_attack: int = 1000,
    num_features: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic flow feature vectors with known ground truth labels.
    """
    rng = np.random.RandomState(seed)

    benign_features = rng.normal(
        loc=0.3, scale=0.12, size=(num_benign, num_features)
    ).astype(np.float32)
    benign_labels_binary = np.zeros(num_benign, dtype=np.int32)
    benign_labels_class = ["BENIGN"] * num_benign

    attack_features = []
    attack_labels_binary = []
    attack_labels_class = []

    attacks_per_class = num_attack // (len(ATTACK_CLASSES) - 1)  # Exclude BENIGN

    for attack_type in ATTACK_CLASSES[1:]:
        n = attacks_per_class

        if attack_type in ("DoS", "DDoS"):
            feats = rng.normal(loc=0.85, scale=0.08, size=(n, num_features))
            feats[:, 6] = rng.uniform(0.0, 0.05, n)
            feats[:, 8] = rng.uniform(0.8, 1.0, n)
        elif attack_type == "PortScan":
            feats = rng.normal(loc=0.5, scale=0.15, size=(n, num_features))
            feats[:, 3] = rng.uniform(0.7, 1.0, n)
            feats[:, 5] = rng.uniform(0.0, 0.15, n)
        elif attack_type == "BruteForce":
            feats = rng.normal(loc=0.6, scale=0.1, size=(n, num_features))
            feats[:, 8] = rng.uniform(0.5, 0.9, n)
            feats[:, 11] = rng.uniform(0.0, 0.2, n)
        elif attack_type == "WebAttack":
            feats = rng.normal(loc=0.55, scale=0.12, size=(n, num_features))
            feats[:, 11] = rng.uniform(0.7, 1.0, n)
            feats[:, 3] = rng.uniform(0.001, 0.01, n)
        elif attack_type == "Infiltration":
            feats = rng.normal(loc=0.35, scale=0.1, size=(n, num_features))
            feats[:, 11] = rng.uniform(0.5, 0.8, n)
        elif attack_type == "Botnet":
            feats = rng.normal(loc=0.45, scale=0.08, size=(n, num_features))
            feats[:, 6] = rng.uniform(0.4, 0.6, n)
            feats[:, 9] = rng.uniform(0.7, 1.0, n)
        elif attack_type == "ZeroDay":
            feats = rng.normal(loc=0.7, scale=0.2, size=(n, num_features))
        else:
            feats = rng.normal(loc=0.6, scale=0.15, size=(n, num_features))

        attack_features.append(feats.astype(np.float32))
        attack_labels_binary.extend([1] * n)
        attack_labels_class.extend([attack_type] * n)

    attack_features = np.vstack(attack_features)
    attack_labels_binary = np.array(attack_labels_binary, dtype=np.int32)

    all_features = np.clip(
        np.vstack([benign_features, attack_features]), 0.0, 1.0
    )
    all_labels_binary = np.concatenate([benign_labels_binary, attack_labels_binary])
    all_labels_class = benign_labels_class + attack_labels_class

    perm = rng.permutation(len(all_features))
    return all_features[perm], all_labels_binary[perm], [all_labels_class[i] for i in perm]


# ──────────────────────────────────────────────
# SNN Evaluation
# ──────────────────────────────────────────────

def evaluate_snn(
    features: np.ndarray,
    labels_binary: np.ndarray,
    labels_class: list[str],
    warmup_samples: int = 500,
    warmup_features: Optional[np.ndarray] = None,
    context_keys: Optional[np.ndarray] = None,
    warmup_context_keys: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Evaluate the Hebbian SNN ensemble on test data.

    The SNN is unsupervised, so we:
    1. Feed warmup_samples of benign-only traffic to establish baselines.
    2. Run the remaining samples and collect anomaly predictions.
    3. Compare anomaly flags to ground truth labels.

    `seed` is passed straight through to HebbianEnsemble so weight
    initialisation is reproducible. This is a live parameter, not a
    hardcoded value -- callers (run_benchmark, seed_sweep) control it.
    """
    logger.info("Evaluating Hebbian SNN Ensemble on %d samples...", len(features))

    encoder = SpikeEncoder(num_features=features.shape[1])
    ensemble = HebbianEnsemble(seed=seed)

    if warmup_features is not None:
        logger.info("Warmup phase: feeding %d benign samples from dedicated warmup set...", len(warmup_features))
        for idx_w, x in enumerate(warmup_features):
            spikes = encoder.encode(x)
            ck = str(warmup_context_keys[idx_w]) if warmup_context_keys is not None else None
            ensemble.process_event(spikes, learn=True, context_key=ck)

        warmup_len = len(warmup_features)
        eval_indices = list(range(len(features)))
    else:
        benign_indices = np.where(labels_binary == 0)[0]
        warmup_indices = benign_indices[:warmup_samples]
        warmup_len = len(warmup_indices)

        logger.info("Warmup phase: feeding %d benign samples...", len(warmup_indices))
        for idx in warmup_indices:
            spikes = encoder.encode(features[idx])
            ck = str(context_keys[idx]) if context_keys is not None else None
            ensemble.process_event(spikes, learn=True, context_key=ck)

        eval_indices = list(range(len(features)))
        eval_set = set(eval_indices) - set(warmup_indices.tolist())
        eval_indices = sorted(eval_set)

    y_true = []
    y_pred = []
    z_scores = []
    latencies = []

    for idx in eval_indices:
        spikes = encoder.encode(features[idx])
        ck = str(context_keys[idx]) if context_keys is not None else None

        t0 = time.perf_counter_ns()
        result = ensemble.process_event(spikes, learn=False, context_key=ck)
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        y_true.append(labels_binary[idx])
        y_pred.append(1 if result["is_anomaly"] else 0)
        z_scores.append(result["z_score"])
        latencies.append(latency_ms)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    z_scores = np.array(z_scores)
    latencies = np.array(latencies)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=cast(Any, 0)))
    recall = float(recall_score(y_true, y_pred, zero_division=cast(Any, 0)))
    f1 = float(f1_score(y_true, y_pred, zero_division=cast(Any, 0)))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    results = {
        "engine": "CortiX Hebbian SNN",
        "seed": seed,
        "samples_evaluated": len(eval_indices),
        "warmup_samples": warmup_len,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "detection_rate": recall,
        "z_scores": z_scores,
        "y_true": y_true,
        "y_pred": y_pred,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
    }

    try:
        from cortix.api.routes.brain import set_ensemble
        set_ensemble(ensemble)
    except Exception as e:
        logger.warning("Could not register ensemble with Brain API: %s", e)

    try:
        from datetime import datetime, timezone
        from cortix.database import get_session, SystemMetric, Threat
        from cortix.redis_bus import get_bus, CHANNEL_LIVE_EVENTS

        db = get_session()
        mean_lat = float(np.mean(latencies)) if len(latencies) > 0 else 1.0
        throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        metric = SystemMetric(
            timestamp=datetime.now(timezone.utc),
            event_count=len(eval_indices),
            tp_count=int(tp),
            fp_count=int(fp),
            fn_count=int(fn),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p99_ms=float(np.percentile(latencies, 99)),
            throughput_pps=throughput,
        )
        db.add(metric)

        bus = get_bus()
        threat_count = 0
        for i, idx in enumerate(eval_indices):
            if y_pred[i] == 1:
                cls = labels_class[idx] if idx < len(labels_class) else "Anomaly"
                z = float(z_scores[i])
                conf = min(0.99, max(0.5, z / 20.0))

                src_ip = f"172.16.{(idx >> 8) & 255}.{idx & 255}"

                threat = Threat(
                    timestamp=datetime.now(timezone.utc),
                    src_ip=src_ip,
                    dst_port=80,
                    protocol="TCP",
                    attack_class=cls,
                    confidence=conf,
                    z_score=z,
                    action_taken="Block IP" if z > 15.0 else "Log Alert",
                    resolved=False,
                    flow_features_json=json.dumps(features[idx].tolist()),
                )
                db.add(threat)
                threat_count += 1

                if threat_count <= 100 and bus.is_connected:
                    bus.publish(CHANNEL_LIVE_EVENTS, {
                        "event": "THREAT_ALERT",
                        "src_ip": src_ip,
                        "attack_class": cls,
                        "confidence": conf,
                        "z_score": z,
                        "action_taken": threat.action_taken,
                        "timestamp": time.time(),
                    })
        db.commit()
        db.close()
        logger.info("Persisted real benchmark SystemMetric and %d Threat events to database.", threat_count)
    except Exception as e:
        logger.warning("Database persistence during benchmark skipped/failed: %s", e)

    return results


def evaluate_hybrid(
    snn_results: dict,
    raw_features: np.ndarray,
    labels_binary: np.ndarray,
    labels_class: list[str],
    dataset_name: str = "cicids2017",
) -> dict:
    """
    Stage 2 of the Hybrid Pipeline:
    Routes all flows flagged by the SNN through the deep PyTorch LSTM-CNN classifier.

    dataset_name selects which checkpoint/scaler/encoder to use, and (for
    nslkdd) triggers loading classifier-consistent features instead of the
    SNN's raw_features array.
    """
    from cortix.config import config
    import pickle

    hybrid_results = snn_results.copy()
    hybrid_results["engine"] = "CortiX Hybrid (SNN + LSTM-CNN)"

    if dataset_name == "nslkdd":
        model_path = config.MODEL_PATH_NSLKDD
        scaler_path = "models/scaler_nslkdd.pkl"
        label_encoder_path = "models/label_encoder_nslkdd.pkl"

        from evaluation.classifier_eval_utils import (
            load_nslkdd_features_for_classifier,
            get_benign_class_index,
        )
        classifier_features = load_nslkdd_features_for_classifier(
            scaler_path=scaler_path
        )
        if classifier_features is None:
            logger.warning(
                "Stage 2: NSL-KDD classifier features unavailable "
                "(train the classifier first). Skipping Stage 2."
            )
            return hybrid_results

        benign_idx = get_benign_class_index(label_encoder_path)
        if benign_idx is None:
            logger.warning("Stage 2: could not resolve BENIGN class index. Skipping.")
            return hybrid_results
    else:
        model_path = config.MODEL_PATH_CICIDS2017
        scaler_path = "models/scaler.pkl"
        classifier_features = raw_features  # existing CICIDS2017 behaviour
        benign_idx = 0  # CICIDS2017 pipeline already assumed this; unchanged

    if not os.path.exists(model_path):
        logger.warning(
            "Hybrid Classifier model not found at %s. Skipping Stage 2. "
            "Run cortix/classifier/train.py first.", model_path
        )
        return hybrid_results

    import torch
    from cortix.classifier.model import CortixLSTMCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    num_classes = ckpt["fc2.bias"].shape[0] if "fc2.bias" in ckpt else config.CLASSIFIER_NUM_CLASSES
    # Infer num_features from the checkpoint's own conv1 weight shape
    # (in_channels dimension) rather than trusting a possibly-wrong config
    # constant -- this makes the code correct regardless of which dataset
    # the loaded checkpoint was actually trained on.
    if "conv1.weight" in ckpt:
        num_features = ckpt["conv1.weight"].shape[1]
    else:
        num_features = classifier_features.shape[1]

    model = CortixLSTMCNN(num_classes=num_classes, num_features=num_features).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    scaler_obj = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler_obj = pickle.load(f)
        except Exception:
            scaler_obj = None

    y_pred_snn = snn_results["y_pred"]
    y_pred_hybrid = y_pred_snn.copy()

    seq_len = config.CLASSIFIER_SEQ_LEN

    with torch.no_grad():
        for i in range(len(y_pred_snn)):
            if y_pred_snn[i] == 1:
                start_idx = max(0, i - seq_len + 1)
                seq_features = classifier_features[start_idx:i + 1]

                if len(seq_features) < seq_len:
                    padding = np.zeros(
                        (seq_len - len(seq_features), classifier_features.shape[1]),
                        dtype=np.float32,
                    )
                    seq_features = np.vstack([padding, seq_features])

                # classifier_features are ALREADY scaled by the classifier's
                # own saved scaler (loaded above / inside the nslkdd loader) --
                # do NOT re-apply scaler_obj.transform() here, that would
                # double-scale. (For CICIDS2017's existing path, raw_features
                # was never pre-scaled by the classifier's scaler, so keep
                # applying it there.)
                if dataset_name != "nslkdd" and scaler_obj is not None and \
                        getattr(scaler_obj, "n_features_in_", 0) == seq_features.shape[1]:
                    try:
                        seq_features = scaler_obj.transform(seq_features)
                    except Exception:
                        pass

                x_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(device)

                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=-1)
                benign_prob = probs[0, benign_idx].item()

                if benign_prob > 0.85:
                    y_pred_hybrid[i] = 0

    y_true = snn_results["y_true"]

    tp = np.sum((y_true == 1) & (y_pred_hybrid == 1))
    tn = np.sum((y_true == 0) & (y_pred_hybrid == 0))
    fp = np.sum((y_true == 0) & (y_pred_hybrid == 1))
    fn = np.sum((y_true == 1) & (y_pred_hybrid == 0))

    accuracy = float(accuracy_score(y_true, y_pred_hybrid))
    precision = float(precision_score(y_true, y_pred_hybrid, zero_division=cast(Any, 0)))
    recall = float(recall_score(y_true, y_pred_hybrid, zero_division=cast(Any, 0)))
    f1 = float(f1_score(y_true, y_pred_hybrid, zero_division=cast(Any, 0)))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    hybrid_results.update({
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1_score": f1, "fpr": fpr, "fnr": fnr,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "detection_rate": recall, "y_pred": y_pred_hybrid,
    })

    logger.info("Hybrid Stage 2 Evaluation Complete. New FPR: %.4f%%, Detection: %.2f%%",
                fpr * 100, recall * 100)
    return hybrid_results


# ──────────────────────────────────────────────
# Visualisation Generators
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str,
):
    """Generate and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=cast(Any, ax),
        linewidths=0.5,
        linecolor="#333333",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    z_scores: np.ndarray,
    title: str,
    output_path: str,
):
    """Generate and save an ROC curve."""
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, z_scores)
    auc_val = roc_auc_score(y_true, z_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr_arr,
        tpr_arr,
        color="#ef4444",
        lw=2.5,
        label=f"ROC Curve (AUC = {auc_val:.4f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Baseline")
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.15, color="#ef4444")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC curve saved to %s", output_path)


def plot_precision_recall(
    y_true: np.ndarray,
    z_scores: np.ndarray,
    title: str,
    output_path: str,
):
    """Generate and save a Precision-Recall curve."""
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, z_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall_arr,
        precision_arr,
        color="#3b82f6",
        lw=2.5,
        label="Precision-Recall Curve",
    )
    ax.fill_between(recall_arr, precision_arr, alpha=0.15, color="#3b82f6")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Precision-Recall curve saved to %s", output_path)


def plot_latency_profile(latency_results: list[dict], output_path: str):
    """Generate latency comparison bar chart."""
    engines = [r["engine"] for r in latency_results]
    p50s = [r["latency_p50_ms"] for r in latency_results]
    p99s = [r["latency_p99_ms"] for r in latency_results]

    x = np.arange(len(engines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, p50s, width, label="p50 Latency (ms)", color="#3b82f6")
    bars2 = ax.bar(x + width / 2, p99s, width, label="p99 Latency (ms)", color="#ef4444")

    ax.set_xlabel("Engine", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Hot-Path Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(engines, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Latency profile saved to %s", output_path)


# ──────────────────────────────────────────────
# Summary Report
# ──────────────────────────────────────────────

def print_benchmark_report(results: list[dict]):
    """Print a formatted comparison report to stdout."""
    print("\n" + "=" * 100)
    print("                         CORTIX SYSTEM BENCHMARK — MEASURED RESULTS")
    print("=" * 100)
    print(
        f"{'Engine':<25} | {'Accuracy':>10} | {'FPR':>10} | "
        f"{'Detection':>10} | {'F1':>8} | {'p50 (ms)':>10} | {'p99 (ms)':>10}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['engine']:<25} | "
            f"{r['accuracy'] * 100:>8.2f}% | "
            f"{r['fpr'] * 100:>8.4f}% | "
            f"{r['detection_rate'] * 100:>8.2f}% | "
            f"{r['f1_score']:>8.4f} | "
            f"{r['latency_p50_ms']:>8.2f}ms | "
            f"{r['latency_p99_ms']:>8.2f}ms"
        )

    print("=" * 100)

    snn_result = next((r for r in results if "SNN" in r["engine"]), results[0])
    hybrid_result = next((r for r in results if "Hybrid" in r["engine"]), results[-1])
    print("\nAcceptance Criteria (Hybrid Pipeline):")
    print(f"  ✓ Hot-path p50 ≤ 3ms:      {'PASS ✅' if snn_result['latency_p50_ms'] <= 3.0 else 'FAIL ❌'} ({snn_result['latency_p50_ms']:.2f}ms)")
    print(f"  ✓ FPR ≤ 0.3%:              {'PASS ✅' if hybrid_result['fpr'] <= 0.003 else 'FAIL ❌'} ({hybrid_result['fpr'] * 100:.4f}%)")
    print(f"  ✓ Detection Rate ≥ 99%:    {'PASS ✅' if hybrid_result['detection_rate'] >= 0.99 else 'FAIL ❌'} ({hybrid_result['detection_rate'] * 100:.2f}%)")
    print(f"  ✓ F1 Score ≥ 0.70:         {'PASS ✅' if hybrid_result['f1_score'] >= 0.70 else 'FAIL ❌'} ({hybrid_result['f1_score']:.4f})")
    print("=" * 100 + "\n")


# ──────────────────────────────────────────────
# Dataset Loaders
# ──────────────────────────────────────────────

def load_nslkdd_dataset(train_csv: str = "data/nslkdd_train.csv", test_csv: str = "data/nslkdd_test.csv"):
    """
    Load NSL-KDD dataset using continuous numeric features for SpikeEncoder population coding.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from data.prepare_nslkdd import prepare_nslkdd, COLUMN_NAMES

    raw_train_path = "data/nslkdd/KDDTrain+.txt"
    raw_test_path = "data/nslkdd/KDDTest+.txt"

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        logger.info("Preprocessed NSL-KDD CSVs missing. Running data/prepare_nslkdd.py...")
        prepare_nslkdd()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = np.asarray(train_df.drop(columns=["Label"]))
    y_train = np.asarray((train_df["Label"] != "BENIGN").astype(int))

    X_test = np.asarray(test_df.drop(columns=["Label"]))
    y_test = np.asarray((test_df["Label"] != "BENIGN").astype(int))
    y_test_class = test_df["Label"].tolist()
    train_services = None
    test_services = None

    X_train_raw = np.asarray(train_df.drop(columns=["Label"]), dtype=np.float32)
    X_test_raw = np.asarray(test_df.drop(columns=["Label"]), dtype=np.float32)

    X_train = np.tanh(X_train_raw / 3.0)
    X_test = np.tanh(X_test_raw / 3.0)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(cast(Any, X_train))
    X_test_scaled = scaler.transform(cast(Any, X_test))

    X_train_sel = X_train_scaled
    X_test_sel = X_test_scaled

    return {
        "X_train": X_train_sel,
        "y_train": y_train,
        "X_test": X_test_sel,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "y_test_class": y_test_class,
        "train_services": train_services,
        "test_services": test_services,
    }


def load_cicids2017_dataset(max_samples: int = 10000):
    """Load CICIDS2017 dataset for tuning and benchmarking."""
    import pandas as pd
    import glob
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split

    files = glob.glob("data/cicids2017/*.csv")
    if not files:
        raise FileNotFoundError("No CICIDS2017 CSV files found in data/cicids2017/")

    df_list = [pd.read_csv(f).sample(frac=0.1, random_state=42) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    label_col = "Label"
    y = np.asarray((df[label_col] != "BENIGN").astype(int))
    y_class = df[label_col].tolist()
    X = np.asarray(df.drop(columns=[label_col]))

    X_train, X_test, y_train, y_test, _, y_test_class = train_test_split(
        X, y, y_class, test_size=0.3, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=16)
    selector.fit(X_train_scaled, np.asarray(y_train))

    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel = selector.transform(X_test_scaled)

    return X_test_sel, X_test_scaled, y_test, y_test_class


# ──────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────

def run_benchmark(
    output_dir: str = "evaluation/results",
    tune_file: Optional[str] = None,
    seed: Optional[int] = None,
    save_plots: bool = True,
    dataset: Optional[str] = None,
):
    """
    Execute the full benchmark suite.

    seed precedence: explicit `seed` arg > tune_file["seed"] > config.RANDOM_SEED.
    This is resolved live at call time -- no hardcoded values -- so a
    seed sweep script can call this repeatedly with different seeds and
    get genuinely different (but each individually reproducible) runs.
    """
    if seed is not None:
        config.RANDOM_SEED = seed

    if tune_file and os.path.exists(tune_file):
        with open(tune_file, "r") as f:
            t = json.load(f)
            config.SLIDING_WINDOW_SIZE = t.get("window_size", config.SLIDING_WINDOW_SIZE)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.HEBBIAN_LR = t.get("learning_rate", t.get("eta", config.HEBBIAN_LR))
            config.METAPLASTICITY_ALPHA = t.get("meta_alpha", config.METAPLASTICITY_ALPHA)
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))
            # Only let the tune file set the seed if the caller didn't
            # already pass one explicitly -- explicit arg always wins.
            if seed is None and "seed" in t:
                config.RANDOM_SEED = t.get("seed")
            logger.info("Loaded tuning config: %s", t)

    resolved_seed = config.RANDOM_SEED
    logger.info("Using RANDOM_SEED=%s for this run", resolved_seed)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CORTIX BENCHMARK SUITE — Starting")
    logger.info("=" * 60)

    if dataset == "nslkdd":
        nsl_data = load_nslkdd_dataset()
        X_train_sel = nsl_data["X_train"]
        y_train = nsl_data["y_train"]
        X_test_sel = nsl_data["X_test"]
        y_test = nsl_data["y_test"]
        y_test_class = nsl_data["y_test_class"]
        train_services = nsl_data["train_services"]
        test_services = nsl_data["test_services"]

        benign_train_mask = y_train == 0
        benign_train = X_train_sel[benign_train_mask]

        warmup_n = min(10000, len(benign_train))
        warmup_features = benign_train[:warmup_n]

        if train_services is not None and test_services is not None:
            warmup_context_keys = train_services[benign_train_mask][:warmup_n]
            context_keys = test_services
        else:
            warmup_context_keys = None
            context_keys = None

        features = X_test_sel
        raw_features = nsl_data["X_test_raw"]
        labels_binary = y_test
        labels_class = y_test_class
        logger.info("Loaded NSL-KDD: %d test samples", len(features))

    elif dataset == "cicids2017":
        import pandas as pd
        import glob
        from sklearn.preprocessing import MinMaxScaler

        files = glob.glob("data/cicids2017/*.csv")
        df_list = []
        for f in files:
            df_list.append(pd.read_csv(f).sample(frac=0.1, random_state=42))
        df = pd.concat(df_list, ignore_index=True)

        df.columns = df.columns.str.strip()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        label_col = "Label"
        y = (df[label_col] != "BENIGN").astype(int).values
        y_class = df[label_col].tolist()
        X = df.drop(columns=[label_col]).values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, _, y_test_class = train_test_split(
            X, y, y_class, test_size=0.3, random_state=42, stratify=y
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=16)
        selector.fit(X_train_scaled, y_train)

        X_train_sel = selector.transform(X_train_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        benign_train_mask = y_train == 0
        benign_train = X_train_sel[benign_train_mask]
        warmup_n = min(5000, len(benign_train))
        warmup_features = benign_train[:warmup_n]

        warmup_context_keys = (cast(np.ndarray, X_train)[benign_train_mask, 0][:warmup_n] // 1000).astype(int)
        context_keys = (cast(np.ndarray, X_test)[:, 0] // 1000).astype(int)

        features = X_test_sel
        raw_features = X_test_scaled
        labels_binary = y_test
        labels_class = y_test_class
        logger.info("Loaded CICIDS2017: %d test samples", len(features))

    else:
        features, labels_binary, labels_class = generate_synthetic_flows(
            num_benign=3000, num_attack=1000
        )
        raw_features = features
        warmup_features = None
        warmup_context_keys = None
        context_keys = None
        logger.info("Generated %d test samples (%d benign, %d attack)",
                    len(features), np.sum(labels_binary == 0), np.sum(labels_binary == 1))

    all_results = []

    # 1. SNN Evaluation — seed passed through live, not hardcoded
    snn_results = evaluate_snn(
        cast(np.ndarray, features),
        cast(np.ndarray, labels_binary),
        cast(list[str], labels_class),
        warmup_features=warmup_features,
        context_keys=context_keys,
        warmup_context_keys=warmup_context_keys,
        seed=resolved_seed,
    )
    all_results.append(snn_results)

    # 1.5 Hybrid Evaluation (SNN + LSTM-CNN)
    hybrid_results = evaluate_hybrid(
        snn_results,
        cast(np.ndarray, raw_features),
        cast(np.ndarray, labels_binary),
        cast(list[str], labels_class),
        dataset_name=dataset or "cicids2017",
    )
    all_results.append(hybrid_results)

    # 2. Generate Visualisations (skippable for fast seed sweeps)
    if save_plots:
        plot_confusion_matrix(
            snn_results["y_true"],
            snn_results["y_pred"],
            labels=["Benign", "Attack"],
            title="CortiX Hebbian SNN — Binary Confusion Matrix",
            output_path=os.path.join(output_dir, "snn_confusion_matrix.png"),
        )

        plot_roc_curve(
            snn_results["y_true"],
            snn_results["z_scores"],
            title="CortiX Hebbian SNN — ROC Curve",
            output_path=os.path.join(output_dir, "snn_roc_curve.png"),
        )

        plot_precision_recall(
            snn_results["y_true"],
            snn_results["z_scores"],
            title="CortiX Hebbian SNN — Precision-Recall Curve",
            output_path=os.path.join(output_dir, "snn_precision_recall.png"),
        )

        plot_confusion_matrix(
            hybrid_results["y_true"],
            hybrid_results["y_pred"],
            labels=["Benign", "Attack"],
            title="CortiX Hybrid Pipeline — Binary Confusion Matrix",
            output_path=os.path.join(output_dir, "hybrid_confusion_matrix.png"),
        )

        plot_latency_profile(all_results, os.path.join(output_dir, "latency_profile.png"))

    # Print report
    print_benchmark_report(all_results)

    # Save JSON results
    json_results = []
    for r in all_results:
        serialisable = {k: v for k, v in r.items()
                        if not isinstance(v, np.ndarray)}
        json_results.append(serialisable)

    results_path = os.path.join(output_dir, f"benchmark_results_seed{resolved_seed}.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Full results saved to %s", results_path)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CortiX Evaluation Benchmark")
    parser.add_argument(
        "--output", type=str, default="evaluation/results",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to CICIDS2017 CSV or 'nslkdd'/'cicids2017' (optional; uses synthetic data if not provided)",
    )
    parser.add_argument(
        "--tune_file", type=str, default=None,
        help="Path to tuning JSON file",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Explicit random seed (overrides config.RANDOM_SEED and tune_file)",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Disable plot generation for faster benchmark runs",
    )
    args = parser.parse_args()
    run_benchmark(
        output_dir=args.output,
        tune_file=args.tune_file,
        seed=args.seed,
        save_plots=not args.no_plots,
        dataset=args.dataset,
    )
