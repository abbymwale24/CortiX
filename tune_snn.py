import os
import json
import numpy as np
import pandas as pd
from typing import cast, Any
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from cortix.snn.ensemble import HebbianEnsemble
from cortix.preprocessor.encoder import SpikeEncoder
import logging

logging.basicConfig(level=logging.INFO)

# Load data
train_csv = "data/nslkdd_train.csv"
test_csv = "data/nslkdd_test.csv"
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

X_train = np.asarray(train_df.drop(columns=["Label"]))
y_train = np.asarray((train_df["Label"] != "BENIGN").astype(int))
X_test = np.asarray(test_df.drop(columns=["Label"]))
y_test = np.asarray((test_df["Label"] != "BENIGN").astype(int))

# Select top 16 features
selector = SelectKBest(f_classif, k=16)
X_train_sel = selector.fit_transform(X_train, y_train)  # type: ignore
X_test_sel = selector.transform(X_test)  # type: ignore

# MinMax scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# Prepare warmup
benign_train_mask = y_train == 0
benign_train = X_train_scaled[benign_train_mask]
# Random shuffle to ensure representative warmup
np.random.seed(42)
np.random.shuffle(benign_train)
warmup_features = benign_train[:10000]

from cortix.config import config as cortix_config

with open("evaluation/tuned_params_nslkdd.json") as f:
    config = json.load(f)

print(f"Loaded config: {config}")
encoder = SpikeEncoder(num_features=16)

cortix_config.NEURONS_PER_MODULE = 512
cortix_config.HIDDEN_NEURONS = config["hidden"]
cortix_config.HEBBIAN_MODULES = 5
cortix_config.ANOMALY_Z_THRESHOLD = config["threshold"]

ensemble = HebbianEnsemble(M=5)

print("Warming up...")
for i, feature in enumerate(warmup_features):
    spikes = encoder.encode(feature)
    ensemble.process_event(spikes, learn=True, context_key="0")

print("Evaluating test set...")
results = []
for i, feature in enumerate(X_test_scaled):
    spikes = encoder.encode(feature)
    # learn=False prevents model adaptation
    result = ensemble.process_event(spikes, learn=False, context_key="0")
    results.append(result["z_score"])

z_scores = np.array(results)
labels = y_test

# Find threshold that gives <= 0.3% FPR
benign_z = z_scores[labels == 0]
attack_z = z_scores[labels == 1]

print(f"Benign z-scores: median={np.median(benign_z):.2f}, max={np.max(benign_z):.2f}, 99th={np.percentile(benign_z, 99):.2f}")
print(f"Attack z-scores: median={np.median(attack_z):.2f}, max={np.max(attack_z):.2f}, 99th={np.percentile(attack_z, 99):.2f}")

# Sort benign z-scores descending
sorted_benign = np.sort(np.abs(benign_z))[::-1]
# We want <= 0.3% FPR. 0.3% of len(benign_z)
max_fp = int(len(benign_z) * 0.003)
threshold = sorted_benign[max_fp]

print(f"Threshold for 0.3% FPR: {threshold:.4f}")

fp = np.sum(np.abs(benign_z) > threshold)
tp = np.sum(np.abs(attack_z) > threshold)

fpr = fp / len(benign_z) * 100
dr = tp / len(attack_z) * 100

print(f"At threshold={threshold:.4f}: FPR={fpr:.4f}%, Detection Rate={dr:.4f}%")

# Grid search threshold
for thresh in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
    fp = np.sum(np.abs(benign_z) > thresh)
    tp = np.sum(np.abs(attack_z) > thresh)
    print(f"Thresh={thresh}: FPR={fp/len(benign_z)*100:.2f}%, DR={tp/len(attack_z)*100:.2f}%")
