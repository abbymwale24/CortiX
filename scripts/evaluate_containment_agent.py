"""
CortiX — Fast Held-Out Evaluation of Trained RL Containment Agent

Evaluates the trained Double-Dueling DQN agent (models/dqn_containment.zip)
on a completely held-out test set built from KDDTest+ flagged events.
Calculates per-class decision accuracy, precision, and action breakdown.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch

from cortix.config import config
from cortix.containment.agent import ContainmentAgent
from cortix.classifier.model import CortixLSTMCNN
from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.ensemble import HebbianEnsemble
from evaluation.benchmark import load_nslkdd_dataset, evaluate_snn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.scripts.evaluate_containment_agent")

REAL_CLASSES = ["BENIGN", "BruteForce", "DoS", "Infiltration", "PortScan", "WebAttack"]

TARGET_POLICY = {
    "BENIGN": [0],                        # ALLOW
    "DoS": [4, 2, 3],                     # HARD_BLOCK, TEMP_BLOCK, QUARANTINE
    "BruteForce": [4, 2, 3, 1],           # HARD_BLOCK, TEMP_BLOCK, QUARANTINE, RATE_LIMIT
    "PortScan": [2, 3, 4],                # TEMP_BLOCK, QUARANTINE, HARD_BLOCK
    "Infiltration": [5, 1, 2, 3, 4],      # HONEYPOT_REDIRECT, RATE_LIMIT, BLOCK
    "WebAttack": [5, 1, 2, 3, 4],         # HONEYPOT_REDIRECT, RATE_LIMIT, BLOCK
}


def build_heldout_test_dataset(output_path: str = "data/real_containment_events_kddtest.pkl") -> list[dict]:
    """Build a held-out event dataset from KDDTest+ without slow DB transactions."""
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            events = pickle.load(f)
            if len(events) > 0:
                logger.info("Loading cached held-out test events from %s (%d events)...", output_path, len(events))
                return events

    # Load tuning parameters if available
    tune_file = "evaluation/tuned_params_nslkdd.json"
    if os.path.exists(tune_file):
        with open(tune_file, "r") as f:
            t = json.load(f)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.ANOMALY_MODE = t.get("anomaly_mode", config.ANOMALY_MODE)

    logger.info("Building held-out event dataset from KDDTest+...")
    nsl_data = load_nslkdd_dataset()
    X_train_sel = nsl_data["X_train"]
    y_train_binary = nsl_data["y_train"]
    X_test_sel = nsl_data["X_test"]
    y_test_binary = nsl_data["y_test"]
    
    benign_mask = y_train_binary == 0
    warmup_features = X_train_sel[benign_mask][:10000]

    logger.info("Initializing SNN ensemble with warmup (10,000 benign samples)...")
    encoder = SpikeEncoder(num_features=X_test_sel.shape[1], num_neurons=X_test_sel.shape[1] * 8)
    ensemble = HebbianEnsemble(n_input=X_test_sel.shape[1] * 8, seed=42)

    for i in range(len(warmup_features)):
        spikes = encoder.encode(warmup_features[i])
        ensemble.process_event(spikes)

    logger.info("Running Stage 1 SNN evaluation over KDDTest+ (%d samples)...", len(X_test_sel))
    z_scores_all = []
    is_anomaly_all = []
    for i in range(len(X_test_sel)):
        spikes = encoder.encode(X_test_sel[i])
        res = ensemble.process_event(spikes, learn=False)
        z_scores_all.append(res["z_score"])
        
        if config.ANOMALY_MODE == "lower":
            is_anom = res["z_score"] <= config.ANOMALY_Z_THRESHOLD
        else:
            is_anom = res["z_score"] >= config.ANOMALY_Z_THRESHOLD
        is_anomaly_all.append(is_anom)

    z_scores_all = np.array(z_scores_all, dtype=np.float32)
    is_anomaly_all = np.array(is_anomaly_all, dtype=bool)

    flagged_indices = np.where(is_anomaly_all)[0]
    z_scores_flagged = z_scores_all[flagged_indices]
    logger.info("KDDTest+ Total: %d | SNN Flagged Anomalies: %d", len(y_test_binary), len(flagged_indices))

    # Load classifier scaled test features & labels
    with open("models/scaler_nslkdd.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/label_encoder_nslkdd.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    classes = list(label_encoder.classes_)

    df_test = pd.read_csv("data/nslkdd_test.csv")
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df_test.select_dtypes(include="number").columns
    for col in numeric_cols:
        df_test[col] = df_test[col].fillna(df_test[col].median() if not pd.isna(df_test[col].median()) else 0.0)
    object_cols = df_test.select_dtypes(include="object").columns
    for col in object_cols:
        df_test[col] = df_test[col].fillna("Unknown")

    feature_cols = [c for c in df_test.columns if c != "Label"]
    X_raw_test = df_test[feature_cols].to_numpy(dtype=np.float32)
    X_test_scaled = scaler.transform(X_raw_test).astype(np.float32)
    y_test_encoded = label_encoder.transform(df_test["Label"].astype(str).tolist())

    # Load Stage 2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("models/lstm_cnn_nslkdd_finetuned.pt", map_location=device, weights_only=True)
    model = CortixLSTMCNN(num_classes=len(classes), num_features=X_test_scaled.shape[1], seq_len=1).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # Build event dicts
    X_flagged = X_test_scaled[flagged_indices]
    y_flagged_encoded = y_test_encoded[flagged_indices]

    events = []
    batch_size = 256
    with torch.no_grad():
        for b_start in range(0, len(flagged_indices), batch_size):
            b_end = min(b_start + batch_size, len(flagged_indices))
            x_batch = torch.tensor(X_flagged[b_start:b_end], dtype=torch.float32).unsqueeze(1).to(device)
            logits = model(x_batch)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()

            for i_sub, i_global in enumerate(range(b_start, b_end)):
                probs = probs_batch[i_sub]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                true_idx = int(y_flagged_encoded[i_global])
                true_class = classes[true_idx] if true_idx < len(classes) else "BENIGN"
                pred_class = classes[pred_idx] if pred_idx < len(classes) else "BENIGN"

                vol = float(np.clip(np.abs(X_flagged[i_global, 0]) / 3.0, 0.05, 0.95))
                dt = float(np.clip(0.1 + 0.05 * np.abs(X_flagged[i_global, 1]), 0.01, 1.0))
                is_att = bool(true_class != "BENIGN")

                events.append({
                    "is_attack": is_att,
                    "class": true_class,
                    "pred_class": pred_class,
                    "confidence": conf,
                    "z_score": float(z_scores_flagged[i_global]),
                    "reputation": float(0.8 if is_att else 0.1),
                    "volume": vol,
                    "time_since_last": dt,
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(events, f)
    logger.info("Saved %d held-out test events to %s", len(events), output_path)
    return events


def evaluate_trained_agent():
    events = build_heldout_test_dataset()
    logger.info("Evaluating trained RL Containment Agent on %d held-out test events...", len(events))

    agent_path = "models/dqn_containment.zip"
    if not os.path.exists(agent_path):
        logger.error("Trained agent weights not found at %s", agent_path)
        return

    agent = ContainmentAgent(agent_path=agent_path)

    # Class performance stats
    action_names = {0: "ALLOW", 1: "RATE_LIMIT", 2: "TEMP_BLOCK", 3: "QUARANTINE", 4: "HARD_BLOCK", 5: "HONEYPOT"}
    
    class_stats = {
        cls: {"total": 0, "correct": 0, "action_counts": {a: 0 for a in range(6)}}
        for cls in REAL_CLASSES
    }

    for ev in events:
        true_cls = ev["class"]
        if true_cls not in class_stats:
            continue

        action = agent.select_action(
            z_score=ev["z_score"],
            confidence=ev["confidence"],
            predicted_class=ev["pred_class"],
            rolling_fpr=0.01,
            reputation=ev["reputation"],
            volume_percentile=ev["volume"],
            time_since_last_alert=ev["time_since_last"],
        )

        class_stats[true_cls]["total"] += 1
        class_stats[true_cls]["action_counts"][action] += 1

        allowed_actions = TARGET_POLICY.get(true_cls, [0])
        if action in allowed_actions:
            class_stats[true_cls]["correct"] += 1

    print("\n" + "=" * 90)
    print("      CORTIX RL CONTAINMENT AGENT — HELD-OUT EVALUATION REPORT (KDDTest+)")
    print("=" * 90)
    print(f"{'Class Name':<15} | {'Total':>7} | {'Correct':>8} | {'Accuracy':>10} | {'Primary Actions Chosen'}")
    print("-" * 90)

    for cls in REAL_CLASSES:
        st = class_stats[cls]
        tot = st["total"]
        corr = st["correct"]
        acc = (corr / tot * 100.0) if tot > 0 else 0.0
        
        counts = st["action_counts"]
        top_actions = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        act_str = ", ".join([f"{action_names[a]}:{c}" for a, c in top_actions if c > 0][:3])
        
        print(f"{cls:<15} | {tot:>7} | {corr:>8} | {acc:>9.2f}% | {act_str}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    evaluate_trained_agent()
