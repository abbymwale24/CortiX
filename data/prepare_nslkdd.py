"""
CortiX — NSL-KDD Dataset Preparation Pipeline

Loads the raw NSL-KDD .txt files, applies column names, maps attack labels
to CortiX's 9-class taxonomy, one-hot encodes categoricals, normalises
numeric features, and outputs train/test CSVs ready for evaluation.

Usage:
    python data/prepare_nslkdd.py

Output:
    data/nslkdd_train.csv
    data/nslkdd_test.csv
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.data.nslkdd")

# ──────────────────────────────────────────────
# NSL-KDD column names (41 features + label + difficulty)
# ──────────────────────────────────────────────
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# ──────────────────────────────────────────────
# NSL-KDD Label → CortiX 9-Class Mapping
# ──────────────────────────────────────────────
LABEL_MAP = {
    # Benign
    "normal": "BENIGN",

    # DoS attacks
    "neptune": "DoS",
    "smurf": "DoS",
    "land": "DoS",
    "pod": "DoS",
    "teardrop": "DoS",
    "back": "DoS",
    "apache2": "DoS",
    "mailbomb": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",

    # Port Scanning / Probing
    "portsweep": "PortScan",
    "ipsweep": "PortScan",
    "nmap": "PortScan",
    "satan": "PortScan",
    "mscan": "PortScan",
    "saint": "PortScan",

    # Brute Force / Remote-to-Local (R2L)
    "guess_passwd": "BruteForce",
    "ftp_write": "BruteForce",
    "imap": "BruteForce",
    "phf": "BruteForce",
    "multihop": "BruteForce",
    "warezmaster": "BruteForce",
    "sendmail": "BruteForce",
    "named": "BruteForce",
    "snmpgetattack": "BruteForce",
    "snmpguess": "BruteForce",
    "xlock": "BruteForce",
    "xsnoop": "BruteForce",
    "worm": "BruteForce",

    # Infiltration / User-to-Root (U2R)
    "spy": "Infiltration",
    "warezclient": "Infiltration",
    "httptunnel": "Infiltration",

    # Web / Privilege Escalation attacks
    "buffer_overflow": "WebAttack",
    "rootkit": "WebAttack",
    "loadmodule": "WebAttack",
    "perl": "WebAttack",
    "sqlattack": "WebAttack",
    "xterm": "WebAttack",
    "ps": "WebAttack",

    # Botnet (none in original KDD, placeholder)
    # "botnet_sample": "Botnet",
}

# Categorical columns that need one-hot encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_nslkdd(filepath: str) -> pd.DataFrame:
    """Load a raw NSL-KDD .txt file with proper column names."""
    logger.info("Loading: %s", filepath)
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    logger.info("  → %d rows, %d columns", len(df), len(df.columns))
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map fine-grained NSL-KDD labels to CortiX 9-class taxonomy."""
    # Strip trailing dots from labels (some versions have "normal." etc.)
    df["label"] = df["label"].str.strip().str.rstrip(".")

    original_labels = df["label"].unique()
    logger.info("Original label count: %d unique labels", len(original_labels))

    df["Label"] = df["label"].map(LABEL_MAP)

    # Unmapped attacks → ZeroDay (novel/unknown attacks)
    unmapped = df["Label"].isna()
    if unmapped.any():
        unmapped_labels = df.loc[unmapped, "label"].unique()
        logger.warning(
            "Unmapped labels (→ ZeroDay): %s (%d rows)",
            unmapped_labels.tolist(),
            unmapped.sum(),
        )
        df.loc[unmapped, "Label"] = "ZeroDay"

    # Report class distribution
    logger.info("\nClass distribution:")
    for label, count in df["Label"].value_counts().items():
        pct = count / len(df) * 100
        logger.info("  %-15s %8d  (%.1f%%)", label, count, pct)

    return df


def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    One-hot encode categoricals and normalise numerics.
    Fit on train, transform both train and test.
    """
    # Drop original label and difficulty columns from features
    label_train = df_train["Label"].copy()
    label_test = df_test["Label"].copy()

    feature_cols = [c for c in COLUMN_NAMES if c not in ("label", "difficulty")]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    # One-hot encode categoricals (fit on train categories)
    train_cats = df_train[CATEGORICAL_COLS]
    test_cats = df_test[CATEGORICAL_COLS]

    # Combine for consistent encoding, then split
    combined_cats = pd.concat([train_cats, test_cats], axis=0)
    combined_encoded = pd.get_dummies(combined_cats, columns=CATEGORICAL_COLS, dtype=np.float32)

    train_encoded = combined_encoded.iloc[:len(df_train)].reset_index(drop=True)
    test_encoded = combined_encoded.iloc[len(df_train):].reset_index(drop=True)

    # Numeric features
    train_numeric = df_train[numeric_cols].astype(np.float64).reset_index(drop=True)
    test_numeric = df_test[numeric_cols].astype(np.float64).reset_index(drop=True)

    # Replace inf/nan
    train_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_numeric.fillna(0, inplace=True)
    test_numeric.fillna(0, inplace=True)

    # Standardise using train statistics
    train_mean = train_numeric.mean()
    train_std = train_numeric.std().replace(0, 1)

    train_numeric = (train_numeric - train_mean) / train_std
    test_numeric = (test_numeric - train_mean) / train_std

    # Combine numeric + encoded categoricals + label
    train_final = pd.concat([train_numeric, train_encoded, label_train.reset_index(drop=True)], axis=1)
    test_final = pd.concat([test_numeric, test_encoded, label_test.reset_index(drop=True)], axis=1)

    logger.info("Final feature dimensions: train=%s, test=%s",
                train_final.shape, test_final.shape)

    return train_final, test_final


def prepare_nslkdd(
    input_dir: str = "data/nslkdd",
    output_train: str = "data/nslkdd_train.csv",
    output_test: str = "data/nslkdd_test.csv",
):
    """Full preparation pipeline for NSL-KDD."""

    train_path = os.path.join(input_dir, "KDDTrain+.txt")
    test_path = os.path.join(input_dir, "KDDTest+.txt")

    if not os.path.exists(train_path):
        logger.error("Train file not found: %s", train_path)
        logger.info("Run 'bash data/download_nslkdd.sh' first.")
        sys.exit(1)

    if not os.path.exists(test_path):
        logger.error("Test file not found: %s", test_path)
        sys.exit(1)

    # Load
    df_train = load_nslkdd(train_path)
    df_test = load_nslkdd(test_path)

    # Map labels
    logger.info("\n── Train set labels ──")
    df_train = map_labels(df_train)
    logger.info("\n── Test set labels ──")
    df_test = map_labels(df_test)

    # Preprocess features
    train_final, test_final = preprocess_features(df_train, df_test)

    # Save
    train_final.to_csv(output_train, index=False)
    test_final.to_csv(output_test, index=False)

    logger.info("\n✓ Train saved to: %s (%d rows)", output_train, len(train_final))
    logger.info("✓ Test saved to: %s (%d rows)", output_test, len(test_final))

    return train_final, test_final


if __name__ == "__main__":
    prepare_nslkdd()
