import numpy as np
import json
from cortix.preprocessor.encoder import SpikeEncoder
from cortix.snn.hebbian_module import HebbianModule
from cortix.snn.scorer import AnomalyScorer
from evaluation.benchmark import load_cicids2017_dataset, load_nslkdd_dataset
from cortix.config import config

def evaluate(dataset_name, load_fn, tune_file):
    print(f"\n--- Evaluating {dataset_name} ---")
    with open(tune_file) as f:
        params = json.load(f)
    
    config.HIDDEN_NEURONS = params.get("hidden", 256)
    config.NUM_INPUT_NEURONS = 512
    eta = params.get("eta", 0.001)

    features, labels, class_labels = load_fn()
    max_test = 5000
    features = features[:max_test]
    labels = labels[:max_test]

    encoder = SpikeEncoder(num_features=16, num_neurons=512)
    module = HebbianModule(512, config.HIDDEN_NEURONS, 0)
    scorer = AnomalyScorer(z_threshold=2.0)
    
    benign_idx = np.where(labels == 0)[0]
    warmup_n = int(len(benign_idx) * 0.5)
    
    for idx in benign_idx[:warmup_n]:
        spikes = encoder.encode(features[idx])
        post_spikes, act_mag = module.forward(spikes, t=0.0, eta=eta, learn=True)
        scorer.score(act_mag, update_baseline=True)

    test_idx = np.arange(len(features))
    test_idx = np.setdiff1d(test_idx, benign_idx[:warmup_n])
    
    z_benign = []
    z_attack = []
    
    for idx in test_idx:
        spikes = encoder.encode(features[idx])
        
        # Test without learning first
        post_spikes, act_mag = module.forward(spikes, t=0.0, eta=eta, learn=False)
        res = scorer.score(act_mag, update_baseline=False)
        z = res["z_score"]
        
        # CONTINUAL NEUROPLASTICITY: only learn if it's considered normal!
        if abs(z) < scorer.z_threshold:
            module.forward(spikes, t=0.0, eta=eta, learn=True)
            scorer.score(act_mag, update_baseline=True)

        if labels[idx] == 0:
            z_benign.append(abs(z))
        else:
            z_attack.append(abs(z))

    z_benign = np.array(z_benign)
    z_attack = np.array(z_attack)
    
    print(f"Benign abs(Z) - min: {z_benign.min():.2f}, max: {z_benign.max():.2f}, mean: {z_benign.mean():.2f}")
    print(f"Attack abs(Z) - min: {z_attack.min():.2f}, max: {z_attack.max():.2f}, mean: {z_attack.mean():.2f}")
    
    for thresh in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        fp = np.sum(z_benign >= thresh)
        tp = np.sum(z_attack >= thresh)
        fpr = fp / len(z_benign) if len(z_benign) > 0 else 0
        dr = tp / len(z_attack) if len(z_attack) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (prec * dr) / (prec + dr) if (prec + dr) > 0 else 0
        print(f"Thresh {thresh:.1f} -> FPR: {fpr:.2%}, DR: {dr:.2%}, F1: {f1:.4f}")

evaluate("CICIDS2017", lambda: load_cicids2017_dataset(max_samples=10000), "evaluation/tuned_params_cicids2017.json")
