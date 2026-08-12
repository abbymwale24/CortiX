"""
CortiX Module 3 — Classifier Dataset Loader & Preprocessing

Preprocesses raw tabular flow CSV data from CICIDS2017/UNSW-NB15, 
scales features, handles missing/infinite values, balances minority classes, 
and generates sequential temporal sliding windows of shape (seq_len=10, features=40).
"""

import os
import pickle
import logging
from typing import Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from cortix.config import config

logger = logging.getLogger("cortix.classifier.dataset")

# Canonical 40 Features to select from standard datasets
SELECTED_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length"
]

ATTACK_CLASSES = [
    "BENIGN", "DoS", "DDoS", "PortScan", "BruteForce",
    "WebAttack", "Infiltration", "Botnet", "ZeroDay"
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean infinite and missing values in tabular dataset."""
    df = df.copy()
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop completely duplicate rows
    df = df.drop_duplicates()

    # Clean infinity and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        col_median = df[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        df[col] = df[col].fillna(col_median)

    # Handle object columns
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    return df


class SequenceFlowDataset(Dataset):
    """
    Sequence dataset for PyTorch. 
    Groups rows into temporal overlapping sequences of length seq_len.
    """
    def __init__(self, X: Any, y: Any, seq_len: int = 10):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        
        # We need to form contiguous sliding sequences
        self.valid_indices = []
        for i in range(len(X) - seq_len + 1):
            # Assumes subsequent rows represent a temporal sequence
            self.valid_indices.append(i)
            
    def __len__(self) -> int:
        return len(self.valid_indices)
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.valid_indices[index]
        end = start + self.seq_len
        
        # Sequence input of shape (seq_len, features)
        x_seq = self.X[start:end]
        
        # Label of the LAST element in the sequence
        y_label = self.y[end - 1]
        
        return x_seq, y_label


def prepare_datasets(
    csv_path: str,
    seq_len: int = 10,
    batch_size: int = 256,
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_save_path: str | None = None,
    label_encoder_save_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder, StandardScaler]:
    """
    Load CSV, clean features, standardise, form sequence datasets, and return loaders.
    """
    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    # Extract target label
    if "Label" not in df.columns:
        raise ValueError("Dataset CSV must contain a 'Label' column")
        
    labels = df["Label"].astype(str).tolist()
    
    # Check if dataset matches standard CICIDS2017 SELECTED_FEATURES
    matching_selected = sum(1 for f in SELECTED_FEATURES if f in df.columns)
    if matching_selected >= len(SELECTED_FEATURES) // 2:
        existing_features = [f for f in SELECTED_FEATURES if f in df.columns]
        missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
        if missing_features:
            logger.warning("Missing %d features from target list. Padding with zeroes.", len(missing_features))
        X_df = df[existing_features].copy()
        for col in missing_features:
            X_df[col] = 0.0
        X_df = X_df[SELECTED_FEATURES]
    else:
        # Dataset-native features (e.g. NSL-KDD with 119 features)
        feature_cols = [c for c in df.columns if c != "Label"]
        logger.info("Using %d dataset-native features.", len(feature_cols))
        X_df = df[feature_cols].copy()
        
    X_arr = X_df.to_numpy(dtype=np.float32)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_encoded = np.asarray(label_encoder.transform(labels))

    # Split dataset stratifying classes
    # First split off train
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_arr, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    # Then split val from train
    val_adjust = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjust, random_state=42, stratify=y_train_val
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scale state for production inference
    os.makedirs("models", exist_ok=True)
    scaler_out = scaler_save_path or ("models/scaler_nslkdd.pkl" if "nslkdd" in csv_path.lower() else "models/scaler.pkl")
    encoder_out = label_encoder_save_path or ("models/label_encoder_nslkdd.pkl" if "nslkdd" in csv_path.lower() else "models/label_encoder.pkl")
    with open(scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_out, "wb") as f:
        pickle.dump(label_encoder, f)
    if "nslkdd" not in csv_path.lower() and not os.path.exists("models/scaler.pkl"):
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # Create sequential datasets
    train_dataset = SequenceFlowDataset(X_train_scaled, y_train, seq_len)
    val_dataset = SequenceFlowDataset(X_val_scaled, y_val, seq_len)
    test_dataset = SequenceFlowDataset(X_test_scaled, y_test, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        "Dataset loaded. Train shape: %s, Val shape: %s, Test shape: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    return train_loader, val_loader, test_loader, label_encoder, scaler
