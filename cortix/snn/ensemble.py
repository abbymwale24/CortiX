"""
CortiX Module 2 — Hebbian Ensemble

Ensemble of M=5 independent Hebbian modules with majority voting
consensus, metaplasticity controller, and robust anomaly scoring.

Each module maintains an independent AnomalyScorer so that per-module
activation baselines do not contaminate each other.
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

    Each module has its own AnomalyScorer so that the sliding-window
    baseline is built from ONE activation per event per module — not
    M activations per event dumped into a shared window.
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

        # Each module gets an independent anomaly scorer
        self.scorers = [
            AnomalyScorer(
                window_size=config.SLIDING_WINDOW_SIZE,
                z_threshold=config.ANOMALY_Z_THRESHOLD,
            )
            for _ in range(self.M)
        ]

        # Shared metaplasticity controller to adjust learning rate
        self.metaplasticity = MetaplasticityController(
            eta_0=config.HEBBIAN_LR,
            alpha=config.METAPLASTICITY_ALPHA,
        )

        # Global latency tracker (shared across modules)
        self._latency_scorer = AnomalyScorer()

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
        per_module_results = []

        # 2. Run forward pass and STDP for each module, score independently
        for i, module in enumerate(self.modules):
            post_spikes, act_mag = module.forward(
                spike_vector, t=t, eta=eta, learn=learn
            )
            module_activations.append(act_mag)
            module_spikes_list.append(post_spikes)

            # Each module scores against its own baseline
            score_result = self.scorers[i].score(act_mag)
            per_module_results.append(score_result)

        # 3. Majority vote: how many independent modules flag anomaly?
        anomaly_votes = sum(1 for r in per_module_results if r["is_anomaly"])
        is_anomaly = anomaly_votes > self.M / 2

        # Consensus z-score: median of per-module z-scores
        z_scores = [r["z_score"] for r in per_module_results]
        consensus_z = float(np.median(z_scores))

        # 4. Update metaplasticity history with consensus post-synaptic activity
        mean_spikes = np.mean(module_spikes_list, axis=0)
        self.metaplasticity.update(mean_spikes)

        # Benchmark hot-path duration
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        self._latency_scorer.log_latency(latency_ms)

        # Check if still in warmup (any module warming up means ensemble is)
        warming_up = self.total_processed < 50 or any(
            r["warming_up"] for r in per_module_results
        )

        # Compile comprehensive result
        result = {
            "timestamp": t,
            "latency_ms": latency_ms,
            "consensus_score": float(np.median(module_activations)),
            "is_anomaly": is_anomaly,
            "z_score": consensus_z,
            "votes": anomaly_votes,
            "total_modules": self.M,
            "vote_ratio": anomaly_votes / max(self.M, 1),
            "learning_rate": eta,
            "warming_up": warming_up,
        }

        return result

    def get_latency_profile(self) -> dict:
        """Return p50 and p99 hot-path latencies."""
        return self._latency_scorer.get_latency_stats()

    def reset(self):
        """Reset all state."""
        for m in self.modules:
            m.reset_traces()
        for s in self.scorers:
            s.reset()
        self.metaplasticity.reset()
        self._latency_scorer.reset()
        self.total_processed = 0

