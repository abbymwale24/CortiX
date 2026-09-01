"""
CortiX Module 2 — Anomaly Scorer

Robust z-score anomaly detection using Median Absolute Deviation (MAD).
Scores each event relative to a sliding window baseline, per context.

Anomaly modes
-------------
- "upper"     : anomaly if z >  threshold   (attacks produce HIGHER activation)
- "lower"     : anomaly if z <  threshold   (attacks produce LOWER activation)
- "bilateral" : anomaly if |z| > threshold  (attacks deviate in EITHER direction)

Which mode is correct is an empirical property of the encoder/dataset
combination, not a fixed assumption -- different encoder configs (e.g.
thalamic top-k gating vs dense Gaussian coding) can flip which direction
attacks actually deviate in. Measure with evaluation/direction_check.py
before assuming "upper" (the historical default) is right for a given
config.
"""

import logging
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from cortix.config import config

logger = logging.getLogger("cortix.snn.scorer")


class WindowBuffer:
    """Pre-allocated ring buffer for zero-allocation window statistics."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = np.empty(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def append(self, val: float):
        self.buf[self.ptr] = val
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_view(self) -> np.ndarray:
        return self.buf[:self.size]

    def __len__(self) -> int:
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0


class AnomalyScorer:
    """
    MAD-based robust z-score anomaly scorer.

    z = (S - median(S_window)) / (MAD(S_window) + ε)

    MAD is preferred over standard deviation because it is robust
    to outliers — a single extreme anomaly won't inflate the baseline.

    Adaptive threshold calibration
    --------------------------------
    After warmup, the scorer can calibrate its z-threshold from the
    observed benign warmup z-score distribution, adapting to the actual
    activation distribution rather than relying on a static config value.
    Calibration direction matches anomaly_mode:
      - "upper"     -> threshold = high percentile  (e.g. p95: only the
                       top 5% of benign warmup would be flagged)
      - "lower"     -> threshold = low percentile   (e.g. p5: only the
                       bottom 5% of benign warmup would be flagged)
      - "bilateral" -> threshold = high percentile of |z|
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        z_threshold: Optional[float] = None,
        epsilon: float = 1e-6,
        anomaly_mode: Optional[str] = None,
        adaptive_threshold: Optional[bool] = None,
    ):
        self.window_size = window_size or config.SLIDING_WINDOW_SIZE
        self.z_threshold = z_threshold or config.ANOMALY_Z_THRESHOLD
        self.epsilon = epsilon
        self.anomaly_mode = anomaly_mode or config.ANOMALY_MODE
        self.adaptive_threshold = config.ADAPTIVE_THRESHOLD if adaptive_threshold is None else adaptive_threshold

        if self.anomaly_mode not in ("upper", "lower", "bilateral"):
            raise ValueError(
                f"anomaly_mode must be 'upper', 'lower', or 'bilateral', got {self.anomaly_mode!r}"
            )

        # Global baseline window
        self._global_window = WindowBuffer(capacity=self.window_size)

        # Per-context baselines (e.g., per subnet, per protocol)
        self._context_windows: dict[str, WindowBuffer] = defaultdict(
            lambda: WindowBuffer(capacity=self.window_size)
        )

        # Latency tracking
        self._latencies: deque = deque(maxlen=10000)

        # Adaptive threshold state
        self._calibrated = False
        self._warmup_z_scores: list[float] = []

    def calibrate_threshold(self, percentile: float = 99.7):
        """
        Calibrate z-threshold from the warmup z-score distribution.

        `percentile` is interpreted as "how selective the threshold should
        be", regardless of mode -- e.g. percentile=95 always means "only
        ~5% of benign warmup traffic would be flagged", whichever
        direction that 5% sits in:
          - upper:     threshold = p{percentile}       of z      (top tail)
          - lower:     threshold = p{100-percentile}    of z      (bottom tail)
          - bilateral: threshold = p{percentile}        of |z|    (both tails)
        """
        if not self.adaptive_threshold:
            return

        if len(self._warmup_z_scores) < 50:
            return  # Not enough data to calibrate
        z_arr = np.array(self._warmup_z_scores)

        if self.anomaly_mode == "bilateral":
            calibrated = float(np.percentile(np.abs(z_arr), percentile))
            # Floor at 1.5 to prevent a degenerately small threshold
            calibrated = max(calibrated, 1.5)
        elif self.anomaly_mode == "lower":
            calibrated = float(np.percentile(z_arr, 100.0 - percentile))
            # Ceiling at -1.5 to prevent a degenerately small (near-zero
            # or positive) threshold -- mirror of the "upper" floor.
            calibrated = min(calibrated, -1.5)
        else:  # "upper"
            calibrated = float(np.percentile(z_arr, percentile))
            calibrated = max(calibrated, 1.5)

        logger.info(
            "Adaptive threshold calibrated (mode=%s): %.3f → %.3f (from %d warmup z-scores, p%.1f)",
            self.anomaly_mode, self.z_threshold, calibrated, len(self._warmup_z_scores), percentile,
        )
        self.z_threshold = calibrated
        self._calibrated = True
        self._warmup_z_scores.clear()

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

        if context_key and len(self._context_windows[context_key]) >= 20:
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

        window_arr = window.get_view()
        median_val = float(np.median(window_arr))
        mad_val = float(np.median(np.abs(window_arr - median_val)))

        # Standard deviation scale estimate from MAD (sigma ≈ 1.4826 * MAD)
        scale = 1.4826 * mad_val
        if scale < 1e-4:
            scale = 1e-4

        # Robust MAD z-score
        z = (activation_magnitude - median_val) / scale

        # Collect warmup z-scores for adaptive threshold calibration
        if update_baseline and self.adaptive_threshold and not self._calibrated:
            self._warmup_z_scores.append(z)

        # Anomaly check depends on mode:
        #   "upper"     — one-tailed: only positive spikes are anomalous
        #                 (attacks produce HIGHER activation than benign)
        #   "lower"     — one-tailed: only negative spikes are anomalous
        #                 (attacks produce LOWER activation than benign —
        #                 empirically the case for some sparse/gated
        #                 encoder configs; verify with direction_check.py
        #                 before assuming "upper" is correct)
        #   "bilateral" — two-tailed: both spikes AND drops are anomalous
        if self.anomaly_mode == "bilateral":
            is_anomaly = abs(z) > self.z_threshold
        elif self.anomaly_mode == "lower":
            is_anomaly = z < self.z_threshold
        else:  # "upper"
            is_anomaly = z > self.z_threshold

        return {
            "z_score": z,
            "is_anomaly": is_anomaly,
            "threshold": self.z_threshold,
            "median": median_val,
            "mad": mad_val,
            "warming_up": False,
        }

    def majority_vote(
        self, module_scores: list[float], threshold: Optional[float] = None
    ) -> dict:
        """
        Ensemble consensus: majority of modules must flag anomaly.

        Args:
            module_scores: List of activation scores from each Hebbian module.
            threshold: Override z-threshold.

        Returns:
            dict with consensus decision and per-module results.
        """
        # NOTE: `threshold` override here does not currently change
        # self.anomaly_mode's comparison direction inside score() -- it
        # only overrides self.z_threshold's numeric value if you wire it
        # through. Left as-is to match prior behaviour; not used by the
        # main ensemble path (HebbianEnsemble calls score() directly).
        results = []

        for module_score in module_scores:
            result = self.score(module_score)
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
        self._calibrated = False
        self._warmup_z_scores.clear()
