"""
CortiX — Encoder Configuration Sweep

The thalamic top-k gate was added to fight three real problems: dense
Gaussian coding saturates (~most neurons firing per event), one-hot
dummy-zero columns spike on nothing, and both hurt latency + FPR. That
rationale is sound. But the measured result at top_k=2 (976 neurons /
122 features / 8 neurons-per-feature) is AUC=0.34 -- WORSE than random,
meaning benign and attack activation magnitudes are inverted, not just
compressed. The gate went from "too much signal" to "wrong signal."

This script empirically sweeps encoder configurations to find where
separability recovers, while ALSO measuring real sparsity (fraction of
neurons active) for each -- so we get evidence for both halves of the
tradeoff instead of assuming either the sparsity claims or the AUC
collapse without checking.

For each config we report:
  - neurons_per_feature, effective_top_k actually used
  - AUC (SNN z-score vs ground truth) on the NSL-KDD test set
  - mean sparsity (fraction of neurons firing) over a warmup sample,
    to verify the saturation/noise-reduction claims directly

This reuses live evaluate_snn()-equivalent logic but is a slimmed-down
loop (no DB persistence, no plots) purely for fast comparative sweeps.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/encoder_sweep.py \
        --tune_file evaluation/tuned_params_nslkdd.json --seed 42
"""

import os
import json
import argparse
import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from cortix.config import config
from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.ensemble import HebbianEnsemble
from evaluation.benchmark import load_nslkdd_dataset

logging.basicConfig(level=logging.WARNING)  # quiet -- this sweep prints its own summary
logger = logging.getLogger("cortix.evaluation.encoder_sweep")


def run_one_config(
    features: Any,
    labels_binary: Any,
    warmup_features: Any,
    seed: int,
    neurons_per_feature: int,
    thalamic_gate: bool,
) -> dict:
    """Run one live warmup+eval pass with a specific encoder configuration."""
    num_features = features.shape[1]
    num_neurons = num_features * neurons_per_feature

    encoder = SpikeEncoder(
        num_features=num_features,
        num_neurons=num_neurons,
        thalamic_gate=thalamic_gate,
    )
    ensemble = HebbianEnsemble(seed=seed, n_input=encoder.actual_neurons)

    # Warmup (learn=True)
    for x in warmup_features:
        spikes = encoder.encode(x)
        ensemble.process_event(spikes, learn=True)

    # Eval (learn=False) -- also track empirical sparsity
    z_scores = np.zeros(len(features), dtype=np.float32)
    sparsity_samples = []

    for idx in range(len(features)):
        spikes = encoder.encode(features[idx])
        if idx % 50 == 0:  # sample sparsity periodically, cheap
            sparsity_samples.append(float(np.mean(spikes)))
        result = ensemble.process_event(spikes, learn=False)
        z_scores[idx] = result["z_score"]

    auc = float(roc_auc_score(labels_binary, z_scores))
    mean_sparsity = float(np.mean(sparsity_samples)) if sparsity_samples else float("nan")

    return {
        "neurons_per_feature": neurons_per_feature,
        "effective_top_k": encoder.effective_top_k if thalamic_gate else None,
        "thalamic_gate": thalamic_gate,
        "num_neurons": encoder.actual_neurons,
        "auc": auc,
        "mean_fraction_active": mean_sparsity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation/results")
    args = parser.parse_args()

    if args.tune_file and os.path.exists(args.tune_file):
        with open(args.tune_file, "r") as f:
            t = json.load(f)
            config.HEBBIAN_LR = t.get("learning_rate", t.get("eta", config.HEBBIAN_LR))
            config.HIDDEN_NEURONS = t.get("hidden_neurons", t.get("hidden", config.HIDDEN_NEURONS))
            # NOTE: threshold intentionally NOT applied here -- this sweep
            # measures raw AUC, which is threshold-independent by design.

    os.makedirs(args.output, exist_ok=True)

    print("Loading NSL-KDD (full dataset)...")
    nsl_data = load_nslkdd_dataset()
    y_train = nsl_data["y_train"]
    X_train_sel = nsl_data["X_train"]
    benign_train_mask = y_train == 0
    benign_train = X_train_sel[benign_train_mask]
    warmup_n = min(10000, len(benign_train))
    warmup_features = benign_train[:warmup_n]

    features = nsl_data["X_test"]
    labels_binary = nsl_data["y_test"]

    # Sweep: neurons-per-feature values, each tested gated vs ungated.
    # neurons_per_feature=8 + gated is your CURRENT (broken, AUC=0.34) config.
    npf_candidates = [4, 8, 16, 24, 32]

    results = []
    for npf in npf_candidates:
        for gated in (True, False):
            print(f"Running: neurons_per_feature={npf}, thalamic_gate={gated} ...")
            r = run_one_config(
                features, labels_binary, warmup_features,
                seed=args.seed, neurons_per_feature=npf, thalamic_gate=gated,
            )
            results.append(r)
            print(f"  -> AUC={r['auc']:.4f}  top_k={r['effective_top_k']}  "
                  f"mean_active_fraction={r['mean_fraction_active']:.4f}")

    print("\n" + "=" * 100)
    print("ENCODER SWEEP SUMMARY")
    print("=" * 100)
    print(f"{'npf':>5} | {'gate':>6} | {'top_k':>6} | {'AUC':>7} | {'mean_active_frac':>16}")
    print("-" * 100)
    for r in results:
        tk = r["effective_top_k"] if r["effective_top_k"] is not None else "-"
        print(f"{r['neurons_per_feature']:>5} | {str(r['thalamic_gate']):>6} | {str(tk):>6} | "
              f"{r['auc']:>7.4f} | {r['mean_fraction_active']:>16.4f}")

    best = max(results, key=lambda r: r["auc"])
    print("-" * 100)
    print(f"BEST AUC config: neurons_per_feature={best['neurons_per_feature']}, "
          f"thalamic_gate={best['thalamic_gate']}, top_k={best['effective_top_k']}, "
          f"AUC={best['auc']:.4f}, mean_active_fraction={best['mean_fraction_active']:.4f}")
    print("=" * 100 + "\n")

    print("Interpretation:")
    print("  - Compare gated vs ungated AUC at the SAME neurons_per_feature: this isolates")
    print("    whether gating itself (vs. just neuron count) is helping or hurting.")
    print("  - Compare mean_active_fraction across ungated rows: this checks whether the")
    print("    'saturation' claim (dense coding ~70% active) is real at this feature count.")
    print("  - The best AUC row is your candidate new default -- but also weigh its")
    print("    mean_active_fraction against your FPR/latency requirements, not AUC alone.")

    json_path = os.path.join(args.output, "encoder_sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {json_path}")


if __name__ == "__main__":
    main()
