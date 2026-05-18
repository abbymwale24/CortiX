"""
CortiX Module 2 — Hebbian Ensemble

Ensemble of M=5 independent Hebbian modules with majority voting
consensus, metaplasticity controller, and robust anomaly scoring.
"""

import logging
import time
import numpy as np

from cortix.config import config
from cortix.snn.hebbian_module import HebbianModule
from cortix.snn.metaplasticity import MetaplasticityController
from cortix.snn.scorer import AnomalyScorer

logger = logging.getLogger("cortix.snn.ensemble")


class HebbianEnsemble:
    """
    Hebbian SNN Ensemble.
    
    Coordinates M independent modules, aggregate consensus scores,
    and publishes detected anomalies.
    """

    def __init__(self, M: int = None):
        self.M = M or config.HEBBIAN_MODULES
        self.n_input = config.NEURONS_PER_MODULE
        self.n_hidden = config.HIDDEN_NEURONS

        # Initialize M Hebbian modules with distinct random seeds/variations
        self.modules = [
            HebbianModule(
                n_input=self.n_input,
                n_hidden=self.n_hidden,
                module_id=m,
            )
            for m in range(self.M)
        ]

        # Shared metaplasticity controller to adjust learning rate
        self.metaplasticity = MetaplasticityController(
            eta_0=config.HEBBIAN_LR,
            alpha=config.METAPLASTICITY_ALPHA,
        )

        # Scorer for MAD-based z-score and majority voting
        self.scorer = AnomalyScorer(
            window_size=config.SLIDING_WINDOW_SIZE,
            z_threshold=config.ANOMALY_Z_THRESHOLD,
        )

        # Warmup updates count
        self.total_processed = 0

        logger.info(
            "HebbianEnsemble initialised with M=%d independent modules",
            self.M,
        )

    def process_event(
        self,
        spike_vector: np.ndarray,
        timestamp: float = None,
        learn: bool = True,
    ) -> dict:
        """
        Process a single encoded event through all modules in the ensemble.
        
        Args:
            spike_vector: Encoded binary spike vector of shape (n_input,)
            timestamp: Event timestamp (default: current system time)
            learn: Whether to update weights online
            
        Returns:
            A results dictionary including anomaly decision and z-score.
        """
        t0 = time.perf_counter_ns()
        t = timestamp or time.time()
        self.total_processed += 1

        # 1. Update metaplasticity to get current learning rate η
        eta = self.metaplasticity.eta

        module_activations = []
        module_spikes_list = []

        # 2. Run forward pass and STDP for each module
        for module in self.modules:
            post_spikes, act_mag = module.forward(
                spike_vector, t=t, eta=eta, learn=learn
            )
            module_activations.append(act_mag)
            module_spikes_list.append(post_spikes)

        # 3. Calculate ensemble majority vote consensus
        vote_result = self.scorer.majority_vote(module_activations)

        # 4. Update metaplasticity history with consensus post-synaptic activity
        # Use average activation pattern across modules
        mean_spikes = np.mean(module_spikes_list, axis=0)
        self.metaplasticity.update(mean_spikes)

        # Benchmark hot-path duration
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        self.scorer.log_latency(latency_ms)

        # Compile comprehensive result
        result = {
            "timestamp": t,
            "latency_ms": latency_ms,
            "consensus_score": float(np.median(module_activations)),
            "is_anomaly": vote_result["is_anomaly"],
            "z_score": vote_result["consensus_z_score"],
            "votes": vote_result["anomaly_votes"],
            "total_modules": vote_result["total_modules"],
            "vote_ratio": vote_result["vote_ratio"],
            "learning_rate": eta,
            "warming_up": self.total_processed < 50,
        }

        return result

    def get_latency_profile(self) -> dict:
        """Return p50 and p99 hot-path latencies."""
        return self.scorer.get_latency_stats()

    def reset(self):
        """Reset all state."""
        for m in self.modules:
            m.reset_traces()
        self.metaplasticity.reset()
        self.scorer.reset()
        self.total_processed = 0
