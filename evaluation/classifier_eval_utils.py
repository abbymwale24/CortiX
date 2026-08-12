"""
CortiX — Classifier Evaluation Utilities

Provides dataset-native feature preparation for Stage 2 (LSTM-CNN) inference,
ensuring the exact training-time preprocessing pipeline (clean_dataframe + 
dataset-native feature extraction + saved StandardScaler transform) is used, 
independent of the SNN's tanh/MinMax feature squashing.
"""

import os
import pickle
import logging
from typing import Optional

import numpy as np
import pandas as pd

from cortix.classifier.dataset import clean_dataframe, SELECTED_FEATURES

logger = logging.getLogger("cortix.evaluation.classifier_eval_utils")


def load_nslkdd_features_for_classifier(
    test_csv: str = "data/nslkdd_test.csv",
    scaler_path: str = "models/scaler_nslkdd.pkl",
) -> Optional[np.ndarray]:
    """
    Load NSL-KDD test set and apply the exact saved StandardScaler from training.
    
    Returns:
        np.ndarray of shape (N_samples, 119) with float32 values, or None if files missing.
    """
    if not os.path.exists(test_csv):
        from data.prepare_nslkdd import prepare_nslkdd
        logger.info("NSL-KDD test CSV missing at %s, running data/prepare_nslkdd.py...", test_csv)
        prepare_nslkdd()

    if not os.path.exists(test_csv):
        logger.error("NSL-KDD test CSV not found at %s", test_csv)
        return None

    if not os.path.exists(scaler_path):
        logger.warning(
            "Saved scaler not found at %s. Please train the classifier first "
            "using 'python -m cortix.classifier.train --dataset data/nslkdd_train.csv'.",
            scaler_path,
        )
        return None

    df = pd.read_csv(test_csv)
    df = clean_dataframe(df)

    # Use dataset-native features (all columns except Label)
    feature_cols = [c for c in df.columns if c != "Label"]
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        logger.info(
            "Loaded NSL-KDD classifier features: shape=%s, scaled with %s",
            X_scaled.shape, scaler_path
        )
        return X_scaled
    except Exception as e:
        logger.error("Failed to load or apply scaler from %s: %s", scaler_path, e)
        return None


def get_benign_class_index(label_encoder_path: str = "models/label_encoder_nslkdd.pkl") -> Optional[int]:
    """
    Look up the exact class index for 'BENIGN' in the saved LabelEncoder.
    
    Returns:
        Integer class index, or None if encoder missing / BENIGN not present.
    """
    if not os.path.exists(label_encoder_path):
        logger.warning("Label encoder not found at %s", label_encoder_path)
        return None

    try:
        with open(label_encoder_path, "rb") as f:
            encoder = pickle.load(f)
        classes = list(encoder.classes_)
        if "BENIGN" in classes:
            idx = classes.index("BENIGN")
            logger.info("Resolved BENIGN class index: %d (from classes=%s)", idx, classes)
            return idx
        else:
            logger.warning("'BENIGN' not found in encoder classes: %s", classes)
            return None
    except Exception as e:
        logger.error("Failed to load label encoder from %s: %s", label_encoder_path, e)
        return None
