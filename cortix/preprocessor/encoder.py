"""
CortiX Module 1 — Spike Encoder

Converts numeric feature vectors into sparse binary spike vectors
for the Hebbian SNN engine using Gaussian population coding.

Encoding methods:
  - Rate coding: higher value → more spikes in time window
  - Temporal coding: spike timing encodes feature magnitude
  - Population coding: 512 input neurons via Gaussian receptive fields
"""

import logging
from typing import Optional

import numpy as np

from cortix.config import config

logger = logging.getLogger("cortix.preprocessor.encoder")


class SpikeEncoder:
    """
    Gaussian population coding spike encoder.

    Each input feature is encoded across multiple neurons using
    overlapping Gaussian receptive fields. The output is a sparse
    binary vector of shape (num_neurons,).
    """

    def __init__(
        self,
        num_features: int = 16,
        num_neurons: int | None = None,
        num_centers_per_feature: int | None = None,
        beta: float = 1.5,
        thalamic_gate: bool | None = None,
        thalamic_eps: float | None = None,
    ):
        """
        Args:
            num_features: Number of input features.
            num_neurons: Total output neurons (default from config, auto-scaled up
                         for high-dimensional inputs to ensure ≥8 neurons per feature).
            num_centers_per_feature: Gaussian centers per feature.
            beta: Width scaling factor for Gaussian receptive fields.
            thalamic_gate: Whether to enable Thalamic sensory gating (zero-suppression & strict sparsity).
            thalamic_eps: Value threshold below which dummy/inactive features are silenced.
        """
        self.num_features = num_features
        # Auto-scale: guarantee at least 8 neurons per feature for encoding
        # resolution.  For 16 features → 512 (unchanged).  For 119 features
        # → max(512, 119*8) = 952, giving each feature enough receptive-field
        # centres to discriminate fine-grained value differences.
        min_neurons = num_features * 8
        default_neurons = config.NUM_INPUT_NEURONS
        self.num_neurons = num_neurons or max(default_neurons, min_neurons)

        self.num_centers = (
            num_centers_per_feature or config.RECEPTIVE_FIELD_CENTERS
        )
        self.beta = beta
        self.thalamic_gate = (
            thalamic_gate if thalamic_gate is not None else config.THALAMIC_GATE_ENABLED
        )
        self.thalamic_eps = (
            thalamic_eps if thalamic_eps is not None else config.THALAMIC_ZERO_SUPPRESSION_EPS
        )

        # Ensure total neurons = num_features * num_centers (pad if needed)
        self.neurons_per_feature = self.num_neurons // self.num_features
        self.actual_neurons = self.neurons_per_feature * self.num_features

        # Adaptive top-K: cap at ~25% of neurons_per_feature to avoid
        # saturation.  For 32 neurons/feat → min(3, 8) = 3. For 4 → min(3, 1) = 1.
        self.effective_top_k = min(
            config.THALAMIC_TOP_K,
            max(1, self.neurons_per_feature // 4),
        )

        # Pre-compute Gaussian centers and widths for each feature
        # Centers uniformly spaced in [0, 1]
        self._centers = np.linspace(
            0, 1, self.neurons_per_feature
        ).astype(np.float32)

        # Width = distance between adjacent centers * beta
        if self.neurons_per_feature > 1:
            self._sigma = (
                self.beta
                * (self._centers[1] - self._centers[0])
            )
        else:
            self._sigma = 0.5

        # Online min-max normalisation state
        self._feature_min = np.zeros(num_features, dtype=np.float32)
        self._feature_max = np.ones(num_features, dtype=np.float32)
        self._update_count = 0
        self._warmup = 100  # samples before normalisation stabilises

        logger.info(
            "SpikeEncoder: %d features → %d neurons (%d per feature, σ=%.3f, thalamic_gate=%s, top_k=%d)",
            num_features,
            self.actual_neurons,
            self.neurons_per_feature,
            self._sigma,
            self.thalamic_gate,
            self.effective_top_k,
        )

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode a feature vector to a sparse binary spike vector.

        Args:
            features: numpy array of shape (num_features,) with values in ~[0, 1].

        Returns:
            Sparse binary array of shape (actual_neurons,).
        """
        features = np.asarray(features, dtype=np.float32)

        # Online min-max normalisation
        features = self._normalise(features)

        spikes = np.zeros(self.num_neurons, dtype=np.float32)

        if self.thalamic_gate:
            # ── Vectorised Multi-Spike Thalamic Gating ──
            # Each feature fires its top-K nearest receptive field centres
            # (K=1 is the original ultra-sparse mode; K=3 default for better
            # discriminability without sacrificing sparsity too much).
            top_k = self.effective_top_k
            vals = np.clip(features[:self.num_features], 0.0, 1.0)  # (F,)
            # Compute distances to all centres for all features at once
            # vals[:, None] is (F, 1), self._centers[None, :] is (1, C)
            dists = np.abs(vals[:, np.newaxis] - self._centers[np.newaxis, :])  # (F, C)
            # Suppress features below thalamic threshold
            active_mask = vals >= self.thalamic_eps  # (F,)
            # Find top-K nearest centres per feature
            k = min(top_k, self.neurons_per_feature)
            if k >= self.neurons_per_feature:
                # All neurons fire for active features
                for i in range(self.num_features):
                    if active_mask[i]:
                        start = i * self.neurons_per_feature
                        spikes[start:start + self.neurons_per_feature] = 1.0
            else:
                # Top-K nearest centres per feature (vectorised)
                nearest_k = np.argpartition(dists, k, axis=1)[:, :k]  # (F, K)
                for i in range(self.num_features):
                    if active_mask[i]:
                        start = i * self.neurons_per_feature
                        spikes[start + nearest_k[i]] = 1.0
        else:
            # ── Vectorised Gaussian Population Coding ──
            # Compute all activations in one matrix operation instead of
            # a Python for-loop. This directly fixes the latency blowup
            # on high-dimensional datasets like CICIDS2017.
            vals = np.clip(features[:self.num_features], 0.0, 1.0)  # (F,)
            # (F, 1) - (1, C) -> (F, C) Gaussian activations
            activations = np.exp(
                -((vals[:, np.newaxis] - self._centers[np.newaxis, :]) ** 2)
                / (2 * self._sigma ** 2)
            )  # shape (F, C)
            threshold = 0.3
            spike_matrix = (activations > threshold).astype(np.float32)  # (F, C)
            # Flatten into the spike vector
            spikes[:self.actual_neurons] = spike_matrix.ravel()

        return spikes

    def encode_deterministic(self, features: np.ndarray) -> np.ndarray:
        """
        Deterministic encoding (no stochastic spikes) — for testing.
        """
        if self.thalamic_gate:
            return self.encode(features)

        features = np.asarray(features, dtype=np.float32)
        features = self._normalise(features)

        spikes = np.zeros(self.num_neurons, dtype=np.float32)
        threshold = 0.5

        for i in range(self.num_features):
            value = np.clip(features[i], 0.0, 1.0)
            start_idx = i * self.neurons_per_feature
            end_idx = start_idx + self.neurons_per_feature

            activations = np.exp(
                -((value - self._centers) ** 2) / (2 * self._sigma ** 2)
            )
            spikes[start_idx:end_idx] = (activations > threshold).astype(
                np.float32
            )

        return spikes

    def encode_batch(self, feature_batch: np.ndarray) -> np.ndarray:
        """
        Encode a batch of feature vectors.

        Args:
            feature_batch: shape (batch_size, num_features)

        Returns:
            shape (batch_size, num_neurons)
        """
        batch_size = feature_batch.shape[0]
        result = np.zeros(
            (batch_size, self.num_neurons), dtype=np.float32
        )
        for idx in range(batch_size):
            result[idx] = self.encode(feature_batch[idx])
        return result

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        """Online min-max normalisation with exponential moving update."""
        # If features are already in [0, 1], return directly
        if np.all(features >= 0.0) and np.all(features <= 1.0):
            return features
            
        self._update_count += 1

        if self._update_count <= self._warmup:
            # During warmup, update min/max directly
            self._feature_min = np.minimum(self._feature_min, features)
            self._feature_max = np.maximum(self._feature_max, features)
        else:
            self._feature_min = np.minimum(self._feature_min, features)
            self._feature_max = np.maximum(self._feature_max, features)

        # Normalise to [0, 1]
        denom = self._feature_max - self._feature_min
        denom = np.where(denom < 1e-8, 1.0, denom)
        normalised = (features - self._feature_min) / denom
        return np.clip(normalised, 0.0, 1.0)

    @property
    def sparsity(self) -> float:
        """Expected sparsity ratio of output vectors."""
        return 1.0 - (1.0 / self.neurons_per_feature)

    def reset_normalisation(self):
        """Reset the online normalisation state."""
        self._feature_min = np.zeros(self.num_features, dtype=np.float32)
        self._feature_max = np.ones(self.num_features, dtype=np.float32)
        self._update_count = 0
