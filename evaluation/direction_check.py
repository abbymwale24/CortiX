"""
CortiX — Anomaly Direction Check

The encoder sweep showed AUC=0.34 at the current default config
(npf=8, thalamic_gate=True) -- WORSE than random. Mathematically,
AUC(-z) = 1 - AUC(z) always, so the reverse-direction score has
AUC = 0.6598 on this exact same data. That means the scorer's
one-tailed assumption ("attacks produce HIGHER z") may be backwards
for this encoder, NOT that there's no signal at all.

This script runs the SNN ONCE (default config -- no sweep, cheap)
and reports AUC under three scoring directions:
  - z            (current "upper" mode: higher = more anomalous)
  - -z           (flipped: lower = more anomalous)
  - abs(z)       (bilateral: far from median in EITHER direction = anomalous)

Whichever direction actually wins tells us whether the fix is a
one-line config change (ANOMALY_MODE / scoring sign) rather than an
expensive architecture change.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/direction_check.py \
        --tune_file evaluation/tuned_params_nslkdd.json --seed 42
"""

import os
import json
import argparse
from typing import cast

import numpy as np
from sklearn.metrics import roc_auc_score

from cortix.config import config
from evaluation.benchmark import evaluate_snn, load_nslkdd_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tune_file and os.path.exists(args.tune_file):
        with open(args.tune_file, "r") as f:
            t = json.load(f)
            config.ANOMALY_Z_THRESHOLD = t.get("z_threshold", t.get("threshold", config.ANOMALY_Z_THRESHOLD))
            config.HEBBIAN_LR = t.get("learning_rate", t.get("eta", config.HEBBIAN_LR))
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))

    print("Loading NSL-KDD (full dataset)...")
    nsl_data = load_nslkdd_dataset()
    y_train = nsl_data["y_train"]
    X_train_sel = nsl_data["X_train"]
    benign_train_mask = y_train == 0
    benign_train = X_train_sel[benign_train_mask]
    warmup_n = min(10000, len(benign_train))

    print("Running ONE live SNN pass at default config (this is cheap -- no sweep)...")
    result = evaluate_snn(
        cast(np.ndarray, nsl_data["X_test"]),
        cast(np.ndarray, nsl_data["y_test"]),
        cast(list[str], nsl_data["y_test_class"]),
        warmup_features=benign_train[:warmup_n],
        seed=args.seed,
    )

    y_true = result["y_true"]
    z = result["z_scores"]

    auc_upper = float(roc_auc_score(y_true, z))
    auc_lower = float(roc_auc_score(y_true, -z))
    auc_bilateral = float(roc_auc_score(y_true, np.abs(z)))

    print("\n" + "=" * 80)
    print("ANOMALY DIRECTION CHECK")
    print("=" * 80)
    print(f"AUC using  z          (current 'upper' mode):  {auc_upper:.4f}")
    print(f"AUC using -z          (flipped direction):     {auc_lower:.4f}   (identity: 1 - upper)")
    print(f"AUC using |z|         (bilateral mode):         {auc_bilateral:.4f}")
    print("-" * 80)
    best = max([("upper (z)", auc_upper), ("lower (-z)", auc_lower), ("bilateral (|z|)", auc_bilateral)],
               key=lambda x: x[1])
    print(f"BEST: {best[0]} with AUC={best[1]:.4f}")
    if best[0] != "upper (z)" and best[1] > 0.55:
        print("\n-> The current 'upper' one-tailed assumption is likely WRONG for this")
        print("   encoder config. Switching ANOMALY_MODE (or scoring sign) may recover")
        print("   real separability WITHOUT any architecture/neuron-count changes.")
    elif best[1] < 0.55:
        print("\n-> Even the best direction is weak. This particular encoder config")
        print("   (from the sweep) may genuinely lack strong signal, independent of")
        print("   direction. Worth checking other npf values with this same 3-way test.")
    print("=" * 80)


if __name__ == "__main__":
    main()
