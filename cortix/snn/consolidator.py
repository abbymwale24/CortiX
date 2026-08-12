"""
CortiX Module 2 — Prototype Consolidation

Periodically consolidates learned synaptic patterns across the ensemble
by clustering weights, pruning redundant connections, and removing dead neurons.
"""

import logging
import numpy as np
from sklearn.cluster import KMeans

from cortix.config import config
from cortix.snn.ensemble import HebbianEnsemble

logger = logging.getLogger("cortix.snn.consolidator")


class PrototypeConsolidator:
    """
    Periodic consolidation of learned neural synaptic connections.
    
    Acts as a 'sleep cycle' cleanup for the SNN engine:
    1. Clusters similar weight vectors (neuron tuning curves).
    2. Prunes synapses below threshold.
    3. Revitalises 'dead' (never active) neurons.
    """

    def __init__(self, ensemble: HebbianEnsemble):
        self.ensemble = ensemble
        self.n_input = config.NEURONS_PER_MODULE
        self.n_hidden = config.HIDDEN_NEURONS

    def consolidate(self, prune_threshold: float = 0.05) -> dict:
        """
        Consolidate synaptic weights for all modules.
        
        Args:
            prune_threshold: Synaptic weights below this value are set to 0.
            
        Returns:
            dict containing stats about the cleanup.
        """
        logger.info("Starting SNN prototype consolidation and cleanup")
        
        total_pruned_synapses = 0
        total_reset_neurons = 0

        for m_idx, module in enumerate(self.ensemble.modules):
            W = module.W  # shape (n_hidden, n_input)

            # 1. Dead Neuron Recovery
            # Dead neurons are those whose overall synapse strength is extremely low
            # or who never got post-synaptic spike activity.
            synaptic_strengths = np.sum(W, axis=1)
            dead_idx = np.where(synaptic_strengths < 0.1)[0]
            
            if len(dead_idx) > 0:
                logger.info(
                    "Module [%d]: recovering %d dead/inactive neurons",
                    m_idx,
                    len(dead_idx),
                )
                total_reset_neurons += len(dead_idx)
                
                # Re-initialise dead neurons with random, normalized weights
                for idx in dead_idx:
                    rng = getattr(module, "rng", np.random)
                    W[idx] = rng.uniform(0.1, 0.5, self.n_input).astype(np.float32)
                    W[idx] /= np.sum(W[idx]) + 1e-8

            # 2. Clustering & Consolidation
            # Group weights using KMeans to find redundant prototypes
            n_clusters = max(2, int(self.n_hidden * 0.8))
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
            kmeans.fit(W)
            
            # Replace weights slightly closer to their cluster centroids to reinforce patterns
            # (soft consolidation)
            for i in range(self.n_hidden):
                centroid = kmeans.cluster_centers_[kmeans.labels_[i]]
                # Drift 10% towards centroid
                W[i] = 0.9 * W[i] + 0.1 * centroid

            # 3. Synaptic Pruning
            # Zero out weights below prune_threshold to increase network sparsity
            small_weights = (W < prune_threshold) & (W > 0)
            num_pruned = np.sum(small_weights)
            W[small_weights] = 0.0
            total_pruned_synapses += num_pruned

            # Re-normalise to preserve activity scale
            row_sums = np.sum(W, axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
            W /= row_sums

            # Save cleaned weights back to module
            module.W = W.astype(np.float32)

        logger.info(
            "Consolidation complete: Pruned %d synapses, Recovered %d neurons",
            total_pruned_synapses,
            total_reset_neurons,
        )

        return {
            "pruned_synapses": int(total_pruned_synapses),
            "recovered_neurons": total_reset_neurons,
            "modules_cleaned": len(self.ensemble.modules),
        }
