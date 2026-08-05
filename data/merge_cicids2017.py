"""
CortiX — CICIDS2017 Dataset Merger & Label Mapper

Merges all CICIDS2017 day CSVs into a single file and maps
the raw CIC labels to CortiX's 9-class taxonomy.

Usage:
    python data/merge_cicids2017.py

Output:
    data/cicids2017_merged.csv
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.data.merge")

# ──────────────────────────────────────────────
# CICIDS2017 Label → CortiX 9-class Mapping
# ──────────────────────────────────────────────
# The raw CICIDS2017 dataset has ~15 fine-grained labels.
# We consolidate them into CortiX's 9-class taxonomy:

LABEL_MAP = {
    # Benign
    "BENIGN": "BENIGN",

    # DoS attacks
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",

    # DDoS
    "DDoS": "DDoS",

    # Port Scanning
    "PortScan": "PortScan",

    # Brute Force
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",

    # Web Attacks
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack  Brute Force": "WebAttack",
    "Web Attack  XSS": "WebAttack",
    "Web Attack  Sql Injection": "WebAttack",

    # Infiltration
    "Infiltration": "Infiltration",

    # Botnet
    "Bot": "Botnet",
}


def merge_dataset(input_dir: str = "data/cicids2017", output_path: str = "data/cicids2017_merged.csv"):
    """
    Merge all CICIDS2017 CSV files and apply label mapping.

    Args:
        input_dir: Directory containing the raw CSV files.
        output_path: Path to write the merged CSV.
    """
    if not os.path.exists(input_dir):
        logger.error("Input directory does not exist: %s", input_dir)
        logger.info("Run 'bash data/download_cicids2017.sh' first to download the dataset.")
        sys.exit(1)

    csv_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ])

    if not csv_files:
        logger.error("No CSV files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d CSV files to merge", len(csv_files))

    frames = []
    for csv_file in csv_files:
        path = os.path.join(input_dir, csv_file)
        logger.info("  Loading: %s", csv_file)
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            frames.append(df)
            logger.info("    → %d rows, %d columns", len(df), len(df.columns))
        except Exception as exc:
            logger.error("    ✗ Failed to load %s: %s", csv_file, exc)

    if not frames:
        logger.error("No dataframes loaded. Aborting.")
        sys.exit(1)

    # Merge all frames
    merged = pd.concat(frames, ignore_index=True)
    logger.info("Merged dataset: %d total rows", len(merged))

    # Strip column names again after merge
    merged.columns = merged.columns.str.strip()

    # Apply label mapping
    if "Label" not in merged.columns:
        logger.error("'Label' column not found. Available columns: %s", list(merged.columns))
        sys.exit(1)

    merged["Label"] = merged["Label"].str.strip()

    # Map labels
    original_labels = merged["Label"].unique()
    logger.info("Original labels: %s", original_labels)

    merged["Label"] = merged["Label"].map(LABEL_MAP)

    # Handle unmapped labels (treat as ZeroDay for novelty detection study)
    unmapped = merged["Label"].isna()
    if unmapped.any():
        unmapped_labels = merged.loc[unmapped, "Label"].unique()
        logger.warning("Unmapped labels (mapped to ZeroDay): %s", unmapped_labels)
        merged.loc[unmapped, "Label"] = "ZeroDay"

    # Clean data
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.dropna(axis=0, how="all", inplace=True)

    # Drop duplicates
    before = len(merged)
    merged.drop_duplicates(inplace=True)
    logger.info("Dropped %d duplicate rows", before - len(merged))

    # Report class distribution
    logger.info("\nClass distribution after mapping:")
    for label, count in merged["Label"].value_counts().items():
        pct = count / len(merged) * 100
        logger.info("  %-15s %8d  (%.1f%%)", label, count, pct)

    # Save
    merged.to_csv(output_path, index=False)
    logger.info("\nMerged dataset saved to: %s (%d rows)", output_path, len(merged))

    return merged


if __name__ == "__main__":
    merge_dataset()
