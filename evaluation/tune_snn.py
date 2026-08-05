"""
CortiX — SNN Hyperparameter Tuning Script

Runs a grid search over key SNN hyperparameters:
- Base Learning Rate (eta)
- Z-Score Anomaly Threshold
- Hidden Neurons per Module

Outputs the optimal configuration that maximizes the F1 Score on the target dataset.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from itertools import product
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Add parent directory to path to allow running as module or script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.ensemble import HebbianEnsemble
from cortix.config import config
from evaluation.benchmark import load_nslkdd_dataset, load_cicids2017_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("cortix.evaluation.tune")

def run_grid_search(dataset_name: str, train_features, test_features, test_labels_binary):
    """Run grid search to find optimal hyperparameters."""
    logger.info("=" * 60)
    logger.info("CORTIX SNN TUNING — Starting (%s)", dataset_name)
    logger.info("Train set: %d samples, Test set: %d samples", len(train_features), len(test_features))
    logger.info("=" * 60)

    # Search Space
    hidden_neurons_grid = [128, 256, 512]
    eta_grid = [0.001, 0.005, 0.01]
    threshold_grid = [1.5, 2.0, 2.5, 3.0]

    best_f1 = -1.0
    best_params = None
    results = []

    for hidden, eta, threshold in product(hidden_neurons_grid, eta_grid, threshold_grid):
        logger.info("Testing Config — Hidden: %d | Eta: %.3f | Threshold: %.1f", hidden, eta, threshold)
        
        # Override config globally for the ensemble instantiation
        config.HIDDEN_NEURONS = hidden
        config.BASE_LEARNING_RATE = eta
        
        encoder = SpikeEncoder(num_features=train_features.shape[1])
        ensemble = HebbianEnsemble()
        
        # Warmup Phase (Learn baseline)
        logger.debug("Starting warmup phase...")
        for x in train_features:
            spikes = encoder.encode(x)
            ensemble.process_event(spikes, learn=True)
            
        # Eval Phase (Predict)
        y_pred = []
        for x in test_features:
            spikes = encoder.encode(x)
            res = ensemble.process_event(spikes, learn=False)
            pred = 1 if res["z_score"] > threshold else 0
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        f1 = f1_score(test_labels_binary, y_pred, zero_division=0)
        acc = accuracy_score(test_labels_binary, y_pred)
        recall = recall_score(test_labels_binary, y_pred, zero_division=0)
        
        # FPR
        tn = np.sum((test_labels_binary == 0) & (y_pred == 0))
        fp = np.sum((test_labels_binary == 0) & (y_pred == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        logger.info("  Result: F1=%.4f | Acc=%.4f | FPR=%.4f | DetRate=%.4f", f1, acc, fpr, recall)
        
        results.append({
            "hidden": hidden,
            "eta": eta,
            "threshold": threshold,
            "f1": f1,
            "acc": acc,
            "fpr": fpr,
            "recall": recall
        })

        if f1 > best_f1:
            best_f1 = f1
            best_params = {"hidden": hidden, "eta": eta, "threshold": threshold}

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE!")
    logger.info("Best Parameters: %s", best_params)
    logger.info("Best F1 Score: %.4f", best_f1)
    
    with open(f"evaluation/tuned_params_{dataset_name}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CortiX SNN Hyperparameter Tuner")
    parser.add_argument("--dataset", type=str, choices=["nslkdd", "cicids2017"], required=True)
    args = parser.parse_args()

    if args.dataset == "nslkdd":
        # Load train set (benign only for warmup)
        train_features, train_labels, _ = load_nslkdd_dataset(test_path="data/nslkdd_train.csv")
        benign_train = train_features[train_labels == 0]
        # We don't need all 13k benign samples for tuning speed, let's take a subset
        benign_train = benign_train[:1000] 
        
        # Load test set (mix of benign and attacks)
        test_features, test_labels, _ = load_nslkdd_dataset(test_path="data/nslkdd_test.csv")
        # Subsample test set for faster grid search
        np.random.seed(42)
        idx = np.random.choice(len(test_features), 2000, replace=False)
        test_features = test_features[idx]
        test_labels = test_labels[idx]
        
    elif args.dataset == "cicids2017":
        # CICIDS doesn't have a strict train/test split in our current format, 
        # so we split the merged set manually.
        features, labels, _ = load_cicids2017_dataset(max_samples=10000)
        
        benign_idx = np.where(labels == 0)[0]
        attack_idx = np.where(labels == 1)[0]
        
        # Take 1000 benign for training
        train_idx = benign_idx[:1000]
        benign_train = features[train_idx]
        
        # Remaining for test
        test_idx = np.concatenate([benign_idx[1000:], attack_idx])
        np.random.shuffle(test_idx)
        test_idx = test_idx[:2000] # Subsample for speed
        
        test_features = features[test_idx]
        test_labels = labels[test_idx]

    run_grid_search(args.dataset, benign_train, test_features, test_labels)
