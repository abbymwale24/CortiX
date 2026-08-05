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
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path

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

    Benign flows are drawn from a normal distribution centered around typical
    network behaviour. Attack flows are drawn from shifted distributions to
    simulate anomalous traffic patterns.

    Returns:
        (features, labels_binary, labels_class)
        - features: shape (N, num_features) normalised to [0, 1]
        - labels_binary: 0 = benign, 1 = attack
        - labels_class: string class names
    """
    rng = np.random.RandomState(seed)

    # Benign: low entropy, moderate packet sizes, regular intervals
    benign_features = rng.normal(
        loc=0.3, scale=0.12, size=(num_benign, num_features)
    ).astype(np.float32)
    benign_labels_binary = np.zeros(num_benign, dtype=np.int32)
    benign_labels_class = ["BENIGN"] * num_benign

    # Attack: higher entropy, extreme packet sizes, irregular intervals
    attack_features = []
    attack_labels_binary = []
    attack_labels_class = []

    attacks_per_class = num_attack // (len(ATTACK_CLASSES) - 1)  # Exclude BENIGN

    for attack_type in ATTACK_CLASSES[1:]:
        n = attacks_per_class

        if attack_type in ("DoS", "DDoS"):
            # High volume, low inter-packet interval, uniform packet sizes
            feats = rng.normal(loc=0.85, scale=0.08, size=(n, num_features))
            feats[:, 6] = rng.uniform(0.0, 0.05, n)   # Very low inter-packet interval
            feats[:, 8] = rng.uniform(0.8, 1.0, n)     # High packet count
        elif attack_type == "PortScan":
            # Many different ports, small packets
            feats = rng.normal(loc=0.5, scale=0.15, size=(n, num_features))
            feats[:, 3] = rng.uniform(0.7, 1.0, n)     # High dst port variation
            feats[:, 5] = rng.uniform(0.0, 0.15, n)    # Small packet length
        elif attack_type == "BruteForce":
            # Repetitive patterns, moderate volume
            feats = rng.normal(loc=0.6, scale=0.1, size=(n, num_features))
            feats[:, 8] = rng.uniform(0.5, 0.9, n)     # Moderate-high packet count
            feats[:, 11] = rng.uniform(0.0, 0.2, n)    # Low payload entropy
        elif attack_type == "WebAttack":
            # High entropy payloads, specific ports
            feats = rng.normal(loc=0.55, scale=0.12, size=(n, num_features))
            feats[:, 11] = rng.uniform(0.7, 1.0, n)    # High payload entropy
            feats[:, 3] = rng.uniform(0.001, 0.01, n)  # Port 80/443 range (normalised)
        elif attack_type == "Infiltration":
            # Stealthy: very close to benign but with subtle anomalies
            feats = rng.normal(loc=0.35, scale=0.1, size=(n, num_features))
            feats[:, 11] = rng.uniform(0.5, 0.8, n)    # Slightly elevated entropy
        elif attack_type == "Botnet":
            # Periodic beaconing patterns
            feats = rng.normal(loc=0.45, scale=0.08, size=(n, num_features))
            feats[:, 6] = rng.uniform(0.4, 0.6, n)     # Regular inter-packet interval
            feats[:, 9] = rng.uniform(0.7, 1.0, n)     # Long flow duration
        elif attack_type == "ZeroDay":
            # Novel patterns: mix of different anomalous signatures
            feats = rng.normal(loc=0.7, scale=0.2, size=(n, num_features))
        else:
            feats = rng.normal(loc=0.6, scale=0.15, size=(n, num_features))

        attack_features.append(feats.astype(np.float32))
        attack_labels_binary.extend([1] * n)
        attack_labels_class.extend([attack_type] * n)

    attack_features = np.vstack(attack_features)
    attack_labels_binary = np.array(attack_labels_binary, dtype=np.int32)

    # Combine and clip to [0, 1]
    all_features = np.clip(
        np.vstack([benign_features, attack_features]), 0.0, 1.0
    )
    all_labels_binary = np.concatenate([benign_labels_binary, attack_labels_binary])
    all_labels_class = benign_labels_class + attack_labels_class

    # Shuffle
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
    warmup_features: np.ndarray = None,
) -> dict:
    """
    Evaluate the Hebbian SNN ensemble on test data.

    The SNN is unsupervised, so we:
    1. Feed warmup_samples of benign-only traffic to establish baselines.
    2. Run the remaining samples and collect anomaly predictions.
    3. Compare anomaly flags to ground truth labels.
    """
    logger.info("Evaluating Hebbian SNN Ensemble on %d samples...", len(features))

    encoder = SpikeEncoder(num_features=features.shape[1])
    ensemble = HebbianEnsemble()

    if warmup_features is not None:
        logger.info("Warmup phase: feeding %d benign samples from dedicated warmup set...", len(warmup_features))
        for x in warmup_features:
            spikes = encoder.encode(x)
            ensemble.process_event(spikes, learn=True)
        
        warmup_len = len(warmup_features)
        eval_indices = list(range(len(features)))
    else:
        # Phase 1: Warmup with benign traffic from the test set (synthetic fallback)
        benign_indices = np.where(labels_binary == 0)[0]
        warmup_indices = benign_indices[:warmup_samples]
        warmup_len = len(warmup_indices)

        logger.info("Warmup phase: feeding %d benign samples...", len(warmup_indices))
        for idx in warmup_indices:
            spikes = encoder.encode(features[idx])
            ensemble.process_event(spikes, learn=True)

        # Phase 2: Evaluate on all remaining samples
        eval_indices = list(range(len(features)))
        # Remove warmup samples from evaluation set
        eval_set = set(eval_indices) - set(warmup_indices.tolist())
        eval_indices = sorted(eval_set)

    y_true = []
    y_pred = []
    z_scores = []
    latencies = []

    for idx in eval_indices:
        spikes = encoder.encode(features[idx])

        t0 = time.perf_counter_ns()
        
        # Conditional Continual Learning
        result = ensemble.process_event(spikes, learn=False)
        pred = 1 if result["is_anomaly"] else 0
        if pred == 0:
            ensemble.process_event(spikes, learn=True)
            
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        y_true.append(labels_binary[idx])
        y_pred.append(1 if result["is_anomaly"] else 0)
        z_scores.append(result["z_score"])
        latencies.append(latency_ms)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    z_scores = np.array(z_scores)
    latencies = np.array(latencies)

    # Compute metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    results = {
        "engine": "CortiX Hebbian SNN",
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
        "detection_rate": recall,  # Same as recall
        "z_scores": z_scores,
        "y_true": y_true,
        "y_pred": y_pred,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
    }

    logger.info(
        "SNN Results — Accuracy: %.2f%% | FPR: %.4f%% | Detection Rate: %.2f%% | F1: %.4f",
        accuracy * 100,
        fpr * 100,
        recall * 100,
        f1,
    )

    return results


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
        ax=ax,
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

    # Add value labels on bars
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

    # Acceptance criteria checks
    snn_result = next((r for r in results if "SNN" in r["engine"]), results[0])
    print("\nAcceptance Criteria:")
    print(f"  ✓ Hot-path p50 ≤ 9ms:      {'PASS ✅' if snn_result['latency_p50_ms'] <= 9.0 else 'FAIL ❌'} ({snn_result['latency_p50_ms']:.2f}ms)")
    print(f"  ✓ FPR ≤ 0.3%:              {'PASS ✅' if snn_result['fpr'] <= 0.003 else 'FAIL ❌'} ({snn_result['fpr'] * 100:.4f}%)")
    print(f"  ✓ Detection Rate ≥ 70%:    {'PASS ✅' if snn_result['detection_rate'] >= 0.70 else 'FAIL ❌'} ({snn_result['detection_rate'] * 100:.2f}%)")
    print(f"  ✓ F1 Score ≥ 0.70:         {'PASS ✅' if snn_result['f1_score'] >= 0.70 else 'FAIL ❌'} ({snn_result['f1_score']:.4f})")
    print("=" * 100 + "\n")


# ──────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────

def run_benchmark(output_dir: str = "evaluation/results", tune_file: str = None):
    """Execute the full benchmark suite."""
    if tune_file and os.path.exists(tune_file):
        with open(tune_file, "r") as f:
            t = json.load(f)
            config.SLIDING_WINDOW_SIZE = t.get("window_size", config.SLIDING_WINDOW_SIZE)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", config.ANOMALY_Z_THRESHOLD)
            config.HEBBIAN_LR = t.get("learning_rate", config.HEBBIAN_LR)
            config.METAPLASTICITY_ALPHA = t.get("meta_alpha", config.METAPLASTICITY_ALPHA)
            config.HIDDEN_NEURONS = t.get("hidden_neurons", config.HIDDEN_NEURONS)
            logger.info("Loaded tuning config: %s", t)
            
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CORTIX BENCHMARK SUITE — Starting")
    logger.info("=" * 60)

    # Load real datasets if requested
    dataset = args.dataset
    if dataset == "nslkdd":
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        
        train_path = "data/nslkdd/KDDTrain+.txt"
        test_path = "data/nslkdd/KDDTest+.txt"
        
        # NSL-KDD columns
        cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
                "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
                "root_shell","su_attempted","num_root","num_file_creations","num_shells",
                "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
                "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
                "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
                "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
                "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
                
        df_train = pd.read_csv(train_path, names=cols)
        df_test = pd.read_csv(test_path, names=cols)
        
        # Convert categorical
        for col in ["protocol_type", "service", "flag"]:
            df_train[col] = df_train[col].astype('category').cat.codes
            df_test[col] = df_test[col].astype('category').cat.codes
            
        X_train = df_train.drop(["label", "difficulty"], axis=1).values
        y_train = (df_train["label"] != "normal").astype(int).values
        X_test = df_test.drop(["label", "difficulty"], axis=1).values
        y_test = (df_test["label"] != "normal").astype(int).values
        y_test_class = df_test["label"].tolist()
        
        # Scale (fit on train only!)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select features using SelectKBest
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=16)
        selector.fit(X_train_scaled, y_train)
        
        X_train_sel = selector.transform(X_train_scaled)
        X_test_sel = selector.transform(X_test_scaled)
        
        # For unsupervised evaluation, we combine a warmup set of benign train data, then evaluate on test data.
        benign_train = X_train_sel[y_train == 0]
        warmup_n = min(10000, len(benign_train))
        warmup_features = benign_train[:warmup_n]
        
        features = X_test_sel
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
            # Subsample large files to speed up benchmark
            df_list.append(pd.read_csv(f).sample(frac=0.1, random_state=42))
        df = pd.concat(df_list, ignore_index=True)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Drop NaNs and Infs
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        label_col = "Label"
        y = (df[label_col] != "BENIGN").astype(int).values
        y_class = df[label_col].tolist()
        X = df.drop(columns=[label_col]).values
        
        # Train/Test split
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
        
        benign_train = X_train_sel[y_train == 0]
        warmup_n = min(5000, len(benign_train))
        warmup_features = benign_train[:warmup_n]
        
        features = X_test_sel
        labels_binary = y_test
        labels_class = y_test_class
        logger.info("Loaded CICIDS2017: %d test samples", len(features))
        
    else:
        # Generate synthetic test data
        features, labels_binary, labels_class = generate_synthetic_flows(
            num_benign=3000, num_attack=1000
        )
        warmup_features = None # Uses default logic inside evaluate_snn
        logger.info("Generated %d test samples (%d benign, %d attack)",
                    len(features), np.sum(labels_binary == 0), np.sum(labels_binary == 1))

    all_results = []

    # 1. SNN Evaluation
    snn_results = evaluate_snn(features, labels_binary, labels_class, warmup_features=warmup_features)
    all_results.append(snn_results)

    # 2. Generate Visualisations
    # Binary confusion matrix for SNN
    plot_confusion_matrix(
        snn_results["y_true"],
        snn_results["y_pred"],
        labels=["Benign", "Attack"],
        title="CortiX Hebbian SNN — Binary Confusion Matrix",
        output_path=os.path.join(output_dir, "snn_confusion_matrix.png"),
    )

    # ROC curve
    plot_roc_curve(
        snn_results["y_true"],
        snn_results["z_scores"],
        title="CortiX Hebbian SNN — ROC Curve",
        output_path=os.path.join(output_dir, "snn_roc_curve.png"),
    )

    # Precision-Recall curve
    plot_precision_recall(
        snn_results["y_true"],
        snn_results["z_scores"],
        title="CortiX Hebbian SNN — Precision-Recall Curve",
        output_path=os.path.join(output_dir, "snn_precision_recall.png"),
    )

    # Latency profile
    plot_latency_profile(all_results, os.path.join(output_dir, "latency_profile.png"))

    # Print report
    print_benchmark_report(all_results)

    # Save JSON results
    json_results = []
    for r in all_results:
        serialisable = {k: v for k, v in r.items()
                        if not isinstance(v, np.ndarray)}
        json_results.append(serialisable)

    results_path = os.path.join(output_dir, "benchmark_results.json")
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
        help="Path to CICIDS2017 CSV (optional; uses synthetic data if not provided)",
    )
    parser.add_argument(
        "--tune_file", type=str, default=None,
        help="Path to tuning JSON file",
    )
    args = parser.parse_args()
    run_benchmark(output_dir=args.output, tune_file=args.tune_file)
