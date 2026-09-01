"""
CortiX Module 2 — Hebbian Ensemble

Ensemble of M=5 independent Hebbian modules with majority voting
consensus, metaplasticity controller, and robust anomaly scoring.

Each module maintains an independent AnomalyScorer so that per-module
activation baselines do not contaminate each other.

Reproducibility
----------------
The ensemble now accepts an explicit `seed`. It uses np.random.SeedSequence
to spawn M independent, statistically-decorrelated child seeds — one per
module — rather than naive derivations like `seed + module_id`, which can
produce correlated random streams for related seed values.

If `seed` is None, behaviour falls back to nondeterministic (OS entropy)
initialisation, same as before the fix — so this is strictly additive and
does not change default behaviour unless a seed is explicitly requested.
"""

import logging
import time
from typing import Optional
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

    def __init__(self, M: Optional[int] = None, seed: Optional[int] = None, n_input: Optional[int] = None):
        self.M = M or config.HEBBIAN_MODULES
        self.n_input = n_input or config.NEURONS_PER_MODULE
        # Auto-scale hidden neurons to maintain ~2:1 compression with input.
        # For 512 input → max(256, 256) = 256 (unchanged, matches synthetic).
        # For 952 input → max(256, 476) = 476 (was 256, too compressed).
        self.n_hidden = max(config.HIDDEN_NEURONS, self.n_input // 2)
        self.seed = config.RANDOM_SEED if seed is None else seed

        # ── Spawn M independent, decorrelated child seeds ──
        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(self.M)

        # Initialize M Hebbian modules with distinct, reproducible seeds
        self.modules = [
            HebbianModule(
                n_input=self.n_input,
                n_hidden=self.n_hidden,
                module_id=m,
                seed=child_seeds[m],
            )
            for m in range(self.M)
        ]

        # Each module gets an independent anomaly scorer
        self.scorers = [
            AnomalyScorer(
                window_size=config.SLIDING_WINDOW_SIZE,
                z_threshold=config.ANOMALY_Z_THRESHOLD,
                adaptive_threshold=config.ADAPTIVE_THRESHOLD,
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
            "HebbianEnsemble initialised with M=%d modules (%d→%d, seed=%s)",
            self.M,
            self.n_input,
            self.n_hidden,
            self.seed,
        )

    def process_event(
        self,
        spike_vector: np.ndarray,
        timestamp: Optional[float] = None,
        learn: bool = True,
        context_key: Optional[str] = None,
        neuromodulator_M: float = 1.0,
    ) -> dict:
        """
        Process a single encoded event through all modules in the ensemble.

        Args:
            spike_vector: Encoded binary spike vector of shape (n_input,)
            timestamp: Event timestamp (default: current system time)
            learn: Whether to update weights online
            context_key: Service/protocol context identifier for localized baselines
            neuromodulator_M: Factor 3 neuromodulator intensity (dopamine/danger signal)

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
                spike_vector, t=t, eta=eta, learn=learn, neuromodulator_M=neuromodulator_M
            )
            module_activations.append(act_mag)
            module_spikes_list.append(post_spikes)

            # Each module scores against its own baseline (per context if provided)
            score_result = self.scorers[i].score(
                act_mag, context_key=context_key, update_baseline=learn
            )
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
            "hidden_state": np.concatenate(module_spikes_list)
        }

        return result

    def get_latency_profile(self) -> dict:
        """Return p50 and p99 hot-path latencies."""
        return self._latency_scorer.get_latency_stats()

    def calibrate_thresholds(self, percentile: float = 95.0):
        """
        Calibrate all per-module scorer thresholds from warmup z-scores.

        Call this ONCE after warmup (benign-only learning phase) to set
        each scorer's z-threshold to the given percentile of the observed
        benign z-score distribution.  Default p95 balances detection recall
        (≥70%) with manageable FPR for the hybrid classifier to clean up.
        """
        for i, scorer in enumerate(self.scorers):
            scorer.calibrate_threshold(percentile=percentile)
        logger.info("All %d module scorers calibrated (p%.1f)", self.M, percentile)

    def reset(self):
        """Reset all state."""
        for m in self.modules:
            m.reset_traces()
        for s in self.scorers:
            s.reset()
        self.metaplasticity.reset()
        self._latency_scorer.reset()
        self.total_processed = 0

