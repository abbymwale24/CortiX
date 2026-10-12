"""
CortiX — Real Containment Events Dataset Builder

Constructs a real events_dataset from KDDTrain+ using:
1. Stage 1 SNN z-scores for SNN flagged events (82,090 samples)
2. Stage 2 Fine-Tuned LSTM-CNN predictions & confidence scores
3. Ground-truth labels mapped to the 6 real classes:
   ['BENIGN', 'BruteForce', 'DoS', 'Infiltration', 'PortScan', 'WebAttack']
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch

from cortix.config import config
from cortix.classifier.model import CortixLSTMCNN
from cortix.classifier.finetune_on_snn import load_train_features_for_classifier
from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.ensemble import HebbianEnsemble
from evaluation.benchmark import load_nslkdd_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.scripts.build_real_events_dataset")

REAL_CLASSES = ["BENIGN", "BruteForce", "DoS", "Infiltration", "PortScan", "WebAttack"]


def build_real_events_dataset(output_path: str = "data/real_containment_events_kddtrain.pkl") -> list[dict]:
    # 1. Apply tuning config if available
    tune_file = "evaluation/tuned_params_nslkdd.json"
    if os.path.exists(tune_file):
        with open(tune_file, "r") as f:
            t = json.load(f)
            config.SLIDING_WINDOW_SIZE = t.get("window_size", config.SLIDING_WINDOW_SIZE)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.HEBBIAN_LR = t.get("learning_rate", t.get("eta", config.HEBBIAN_LR))
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))
            config.ANOMALY_MODE = t.get("anomaly_mode", config.ANOMALY_MODE)
            if "adaptive_threshold" in t:
                config.ADAPTIVE_THRESHOLD = t["adaptive_threshold"]
            logger.info("Loaded tuning config: %s", t)

    # 2. Load KDDTrain+ SNN features & binary/class labels
    logger.info("Loading KDDTrain+ features...")
    nsl_data = load_nslkdd_dataset()
    X_train_sel = nsl_data["X_train"]
    y_train_binary = nsl_data["y_train"]
    
    benign_mask = y_train_binary == 0
    warmup_features = X_train_sel[benign_mask][:10000]

    # Load SNN flagged indices
    cache_file = "models/snn_train_flags_cache.npy"
    if os.path.exists(cache_file):
        flagged_indices = np.load(cache_file)
        logger.info("Loaded %d SNN flagged indices from %s.", len(flagged_indices), cache_file)
    else:
        logger.error("SNN flags cache missing at %s", cache_file)
        return []

    # 3. Fast SNN evaluation over warmup + flagged samples to extract z-scores
    logger.info("Initializing SNN ensemble with warmup (10,000 benign samples)...")
    encoder = SpikeEncoder(num_features=X_train_sel.shape[1], num_neurons=X_train_sel.shape[1] * 8)
    ensemble = HebbianEnsemble(n_input=X_train_sel.shape[1] * 8, seed=42)
    
    for i in range(len(warmup_features)):
        spikes = encoder.encode(warmup_features[i])
        ensemble.process_event(spikes)
    
    # Sample 10,000 real flagged events for optimal RL training efficiency
    if len(flagged_indices) > 10000:
        rng = np.random.default_rng(42)
        flagged_indices = rng.choice(flagged_indices, size=10000, replace=False)
        flagged_indices.sort()

    logger.info("Evaluating SNN over %d real flagged events to extract z-scores...", len(flagged_indices))
    z_scores_flagged = []
    for idx in flagged_indices:
        spikes = encoder.encode(X_train_sel[idx])
        res = ensemble.process_event(spikes)
        z_scores_flagged.append(res["z_score"])
    z_scores_flagged = np.array(z_scores_flagged, dtype=np.float32)

    # 4. Load classifier-scaled features & real 6-class labels for KDDTrain+
    X_clf_scaled, y_clf_encoded = load_train_features_for_classifier()
    
    with open("models/label_encoder_nslkdd.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    classes = list(label_encoder.classes_)
    logger.info("LabelEncoder classes: %s", classes)

    # 5. Load Stage 2 Fine-Tuned Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetuned_path = "models/lstm_cnn_nslkdd_finetuned.pt"
    base_model_path = "models/lstm_cnn_nslkdd.pt"
    model_path = finetuned_path if os.path.exists(finetuned_path) else base_model_path
    
    logger.info("Loading Stage 2 Classifier model checkpoint: %s", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = len(classes)
    num_features = X_clf_scaled.shape[1]

    model = CortixLSTMCNN(num_classes=num_classes, num_features=num_features, seq_len=1).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # 6. Build real event dicts for flagged samples
    X_flagged = X_clf_scaled[flagged_indices]
    y_flagged_encoded = y_clf_encoded[flagged_indices]

    events_dataset = []
    logger.info("Evaluating Stage 2 classifier predictions on %d flagged events...", len(flagged_indices))
    
    # Process in batches of 256 for fast PyTorch evaluation
    batch_size = 256
    num_samples = len(flagged_indices)
    
    with torch.no_grad():
        for b_start in range(0, num_samples, batch_size):
            b_end = min(b_start + batch_size, num_samples)
            x_batch = torch.tensor(X_flagged[b_start:b_end], dtype=torch.float32).unsqueeze(1).to(device)
            logits = model(x_batch)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            
            for i_sub, i_global in enumerate(range(b_start, b_end)):
                probs = probs_batch[i_sub]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                true_idx = int(y_flagged_encoded[i_global])
                true_class_name = classes[true_idx] if true_idx < len(classes) else "BENIGN"
                pred_class_name = classes[pred_idx] if pred_idx < len(classes) else "BENIGN"

                z_val = float(z_scores_flagged[i_global])
                is_attack = bool(true_class_name != "BENIGN")

                vol = float(np.clip(np.abs(X_flagged[i_global, 0]) / 3.0, 0.05, 0.95))
                dt = float(np.clip(0.1 + 0.05 * np.abs(X_flagged[i_global, 1]), 0.01, 1.0))
                rep = float(0.8 if is_attack else 0.1)

                events_dataset.append({
                    "is_attack": is_attack,
                    "class": true_class_name,
                    "pred_class": pred_class_name,
                    "confidence": conf,
                    "z_score": z_val,
                    "reputation": rep,
                    "volume": vol,
                    "time_since_last": dt,
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(events_dataset, f)

    logger.info("Successfully built and saved %d real containment events to %s", len(events_dataset), output_path)
    return events_dataset


if __name__ == "__main__":
    ds = build_real_events_dataset()
    print(f"Successfully generated {len(ds)} real containment events.")
