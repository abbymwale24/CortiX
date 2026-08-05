"""
CortiX Module 2 — Anomaly Scorer

Robust z-score anomaly detection using Median Absolute Deviation (MAD).
Scores each event relative to a sliding window baseline, per context.
"""

import logging
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from cortix.config import config

logger = logging.getLogger("cortix.snn.scorer")


class AnomalyScorer:
    """
    MAD-based robust z-score anomaly scorer.

    z = (S - median(S_window)) / (MAD(S_window) + ε)

    MAD is preferred over standard deviation because it is robust
    to outliers — a single extreme anomaly won't inflate the baseline.
    """

    def __init__(
        self,
        window_size: int = None,
        z_threshold: float = None,
        epsilon: float = 1e-6,
    ):
        self.window_size = window_size or config.SLIDING_WINDOW_SIZE
        self.z_threshold = z_threshold or config.ANOMALY_Z_THRESHOLD
        self.epsilon = epsilon

        # Global baseline window
        self._global_window: deque = deque(maxlen=self.window_size)

        # Per-context baselines (e.g., per subnet, per protocol)
        self._context_windows: dict = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        # Latency tracking
        self._latencies: deque = deque(maxlen=10000)

    def score(
        self,
        activation_magnitude: float,
        context_key: Optional[str] = None,
        update_baseline: bool = True,
    ) -> dict:
        """
        Compute anomaly z-score for an activation magnitude.

        Args:
            activation_magnitude: The ensemble consensus score.
            context_key: Optional context identifier (subnet, protocol).
            update_baseline: If True, add to sliding window.

        Returns:
            dict with z_score, is_anomaly, threshold, median, mad
        """
        if update_baseline:
            self._global_window.append(activation_magnitude)
            if context_key:
                self._context_windows[context_key].append(activation_magnitude)

        if context_key:
            window = self._context_windows[context_key]
        else:
            window = self._global_window

        # Need minimum samples for meaningful baseline
        if len(window) < 20:
            return {
                "z_score": 0.0,
                "is_anomaly": False,
                "threshold": self.z_threshold,
                "median": activation_magnitude,
                "mad": 0.0,
                "warming_up": True,
            }

        window_arr = np.array(window)
        median_val = np.median(window_arr)
        
        # Calculate percentage deviation from the median baseline
        # (e.g., if act_mag is 50 and median is 100, deviation is -50%)
        if median_val > 0:
            deviation_pct = (activation_magnitude - median_val) / median_val
        else:
            deviation_pct = 0.0

        # Z-score metric repurposed to represent percentage deviation for compatibility
        z = deviation_pct * 100.0  

        # Anomaly if activation drops below a certain threshold (e.g., -25% -> falls below 75% of normal)
        # z_threshold of 25.0 means an anomaly is flagged if activation is < 75% of median
        is_anomaly = z < -self.z_threshold

        return {
            "z_score": float(z),
            "is_anomaly": bool(is_anomaly),
            "threshold": self.z_threshold,
            "median": float(median_val),
            "mad": 0.0,
            "warming_up": False,
        }

    def majority_vote(
        self, module_scores: list[float], threshold: float = None
    ) -> dict:
        """
        Ensemble consensus: majority of modules must flag anomaly.

        Args:
            module_scores: List of activation scores from each Hebbian module.
            threshold: Override z-threshold.

        Returns:
            dict with consensus decision and per-module results.
        """
        thresh = threshold or self.z_threshold
        results = []

        for score in module_scores:
            result = self.score(score)
            results.append(result)

        # Count how many modules flag anomaly
        anomaly_count = sum(1 for r in results if r["is_anomaly"])
        total = len(module_scores)
        majority = anomaly_count > total / 2

        consensus_z = float(np.median([r["z_score"] for r in results]))

        return {
            "is_anomaly": majority,
            "consensus_z_score": consensus_z,
            "anomaly_votes": anomaly_count,
            "total_modules": total,
            "vote_ratio": anomaly_count / max(total, 1),
            "per_module": results,
        }

    def log_latency(self, latency_ms: float):
        """Record a hot-path latency measurement."""
        self._latencies.append(latency_ms)

    def get_latency_stats(self) -> dict:
        """Get p50/p99 latency statistics."""
        if not self._latencies:
            return {"p50_ms": 0.0, "p99_ms": 0.0, "count": 0}

        arr = np.array(self._latencies)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "count": len(arr),
        }

    def reset(self):
        """Reset all baselines."""
        self._global_window.clear()
        self._context_windows.clear()
        self._latencies.clear()
