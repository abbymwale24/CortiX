"""
CortiX — Stage 2 Fine-Tuning on SNN Anomalies

This script runs the Stage 1 SNN on the Training Set (KDDTrain+) to collect the
population of events that the SNN flags as anomalous (both True Positives and False Positives).
It then fine-tunes the Stage 2 LSTM-CNN classifier exclusively on this population, 
so that Stage 2 specializes in overturning SNN mistakes.

The Test Set (KDDTest+) is strictly excluded to prevent data leakage.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from typing import cast, Any

from cortix.config import config
from cortix.classifier.model import CortixLSTMCNN
from cortix.classifier.dataset import SequenceFlowDataset, clean_dataframe
from evaluation.benchmark import evaluate_snn, load_nslkdd_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.classifier.finetune_on_snn")

def load_train_features_for_classifier(train_csv="data/nslkdd_train.csv", scaler_path="models/scaler_nslkdd.pkl", encoder_path="models/label_encoder_nslkdd.pkl"):
    """Load the exact standard-scaled features and label encoding for the training set."""
    df = pd.read_csv(train_csv)
    
    # We must NOT use clean_dataframe() because it drops duplicates, which misaligns 
    # our indices with evaluate_snn(). We do the cleaning manually here instead.
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        col_median = df[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        df[col] = df[col].fillna(col_median)
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    feature_cols = [c for c in df.columns if c != "Label"]
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    
    labels = df["Label"].astype(str).tolist()
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
        
    y_encoded = label_encoder.transform(labels)
    return X_scaled, y_encoded

def main():
    # 1. Apply any tuning config overrides
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

    # 2. Load the KDDTrain+ dataset for the SNN (MinMax scaled / tanh)
    logger.info("Loading KDDTrain+ for SNN processing...")
    nsl_data = load_nslkdd_dataset()
    X_train_sel = nsl_data["X_train"]
    y_train = nsl_data["y_train"]
    
    # 10k benign warmup from train
    benign_train_mask = y_train == 0
    warmup_features = X_train_sel[benign_train_mask][:10000]

    y_dummy_class = ["BENIGN" if y == 0 else "ATTACK" for y in y_train]
    
    # Cache the SNN results to disk so we don't have to wait 5 hours if the script crashes later
    cache_file = "models/snn_train_flags_cache.npy"
    if os.path.exists(cache_file):
        logger.info("Found cached SNN flags at %s. Loading...", cache_file)
        flagged_indices = np.load(cache_file)
    else:
        logger.info("Running SNN over KDDTrain+ to find the flagged events...")
        snn_results = evaluate_snn(
            features=cast(Any, X_train_sel),
            labels_binary=cast(Any, y_train),
            labels_class=y_dummy_class,
            warmup_features=cast(Any, warmup_features),
            seed=42
        )

        if config.ANOMALY_MODE == "lower":
            y_pred_snn = (snn_results["z_scores"] <= config.ANOMALY_Z_THRESHOLD).astype(int)
        else:
            y_pred_snn = (snn_results["z_scores"] >= config.ANOMALY_Z_THRESHOLD).astype(int)
        
        flagged_indices = np.where(y_pred_snn == 1)[0]
        np.save(cache_file, flagged_indices)
        logger.info("Saved %d SNN flags to cache.", len(flagged_indices))

    logger.info("SNN flagged %d events out of %d in KDDTrain+.", 
                len(flagged_indices), len(y_train))
    
    if len(flagged_indices) < 100:
        logger.error("Too few flagged events to finetune. Exiting.")
        return

    # 3. Extract the classifier-ready standard-scaled features for these specific indices
    logger.info("Loading standard-scaled features for KDDTrain+...")
    X_clf_scaled, y_clf_encoded = load_train_features_for_classifier()
    
    X_flagged = X_clf_scaled[flagged_indices]
    y_flagged = y_clf_encoded[flagged_indices]

    logger.info("Finetuning population size: %d", len(X_flagged))

    # 4. Train/Val/Test Split (70/15/15)
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_flagged, y_flagged, test_size=0.15, random_state=42, stratify=y_flagged
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.15/0.85, random_state=42, stratify=y_train_val
        )
    except ValueError:
        logger.warning("Stratified split failed (likely due to rare classes). Falling back to random split.")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_flagged, y_flagged, test_size=0.15, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.15/0.85, random_state=42
        )

    seq_len = 1
    batch_size = 128
    
    train_dataset = SequenceFlowDataset(X_train, y_train, seq_len=seq_len)
    val_dataset = SequenceFlowDataset(X_val, y_val, seq_len=seq_len)
    test_dataset = SequenceFlowDataset(X_test, y_test, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Train set: %d | Val set: %d | Internal Test set: %d", len(train_dataset), len(val_dataset), len(test_dataset))

    # 5. Model Loading & Fine-Tuning Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "models/lstm_cnn_nslkdd.pt"
    if not os.path.exists(model_path):
        logger.error("Could not find base model checkpoint at %s", model_path)
        return

    logger.info("Loading base model checkpoint: %s", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    num_classes = ckpt["fc2.bias"].shape[0] if "fc2.bias" in ckpt else config.CLASSIFIER_NUM_CLASSES
    num_features = ckpt["conv1.weight"].shape[1] if "conv1.weight" in ckpt else X_flagged.shape[1]
    
    model = CortixLSTMCNN(num_classes=num_classes, num_features=num_features, seq_len=seq_len).to(device)
    model.load_state_dict(ckpt)
    
    class_counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for val in labels:
            class_counts[val] += 1
            
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts + 1e-8)
    class_weights = torch.clamp(class_weights, max=100.0).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    lr = 1e-5
    epochs = 20
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    save_path = "models/lstm_cnn_nslkdd_finetuned.pt"
    patience = 5
    patience_counter = 0

    logger.info("Starting Fine-Tuning for %d epochs...", epochs)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= (total + 1e-8)
        train_acc = 100.0 * correct / (total + 1e-8)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits = model(data)
                loss = criterion(logits, targets)

                val_loss += loss.item() * data.size(0)
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= (val_total + 1e-8)
        val_acc = 100.0 * val_correct / (val_total + 1e-8)
        scheduler.step()

        logger.info(
            "Epoch [%2d/%2d] | Train Loss: %.4f, Acc: %5.2f%% | Val Loss: %.4f, Acc: %5.2f%%",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info("  -> Saved new best checkpoint to %s", save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs.", epoch)
                break
                
    logger.info("Evaluating finetuned model on the internal split test set...")
    ckpt_best = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_best)
    model.eval()
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            _, predicted = logits.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    test_acc = 100.0 * test_correct / (test_total + 1e-8)
    logger.info("Internal Test Set Accuracy: %.2f%%", test_acc)
    logger.info("Fine-tuning complete. You can now run hybrid_threshold_diagnostic.py with the finetuned model!")

if __name__ == "__main__":
    main()
