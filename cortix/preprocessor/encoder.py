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
        num_neurons: int = None,
        num_centers_per_feature: int = None,
        beta: float = 1.5,
    ):
        """
        Args:
            num_features: Number of input features.
            num_neurons: Total output neurons (default from config).
            num_centers_per_feature: Gaussian centers per feature.
            beta: Width scaling factor for Gaussian receptive fields.
        """
        self.num_features = num_features
        self.num_neurons = num_neurons or config.NUM_INPUT_NEURONS
        self.num_centers = (
            num_centers_per_feature or config.RECEPTIVE_FIELD_CENTERS
        )
        self.beta = beta

        # Ensure total neurons = num_features * num_centers (pad if needed)
        self.neurons_per_feature = self.num_neurons // self.num_features
        self.actual_neurons = self.neurons_per_feature * self.num_features

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
            "SpikeEncoder: %d features → %d neurons (%d per feature, σ=%.3f)",
            num_features,
            self.actual_neurons,
            self.neurons_per_feature,
            self._sigma,
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

        # Population coding via Gaussian receptive fields
        spikes = np.zeros(self.actual_neurons, dtype=np.float32)

        for i in range(self.num_features):
            value = np.clip(features[i], 0.0, 1.0)
            start_idx = i * self.neurons_per_feature
            end_idx = start_idx + self.neurons_per_feature

            # Gaussian activation: exp(-||value - center||² / (2σ²))
            activations = np.exp(
                -((value - self._centers) ** 2) / (2 * self._sigma ** 2)
            )

            # Rate coding: threshold to binary spikes
            # Higher activation → higher probability of spike
            spike_probs = activations
            spikes[start_idx:end_idx] = (
                spike_probs > np.random.rand(self.neurons_per_feature)
            ).astype(np.float32)

        return spikes

    def encode_deterministic(self, features: np.ndarray) -> np.ndarray:
        """
        Deterministic encoding (no stochastic spikes) — for testing.

        Uses a fixed threshold instead of random sampling.
        """
        features = np.asarray(features, dtype=np.float32)
        features = self._normalise(features)

        spikes = np.zeros(self.actual_neurons, dtype=np.float32)
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
            shape (batch_size, actual_neurons)
        """
        batch_size = feature_batch.shape[0]
        result = np.zeros(
            (batch_size, self.actual_neurons), dtype=np.float32
        )
        for idx in range(batch_size):
            result[idx] = self.encode(feature_batch[idx])
        return result

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        """Online min-max normalisation with exponential moving update."""
        self._update_count += 1

        if self._update_count <= self._warmup:
            # During warmup, update min/max directly
            self._feature_min = np.minimum(self._feature_min, features)
            self._feature_max = np.maximum(self._feature_max, features)
        else:
            # After warmup, use exponential moving average
            alpha = 0.01
            self._feature_min = (
                (1 - alpha) * self._feature_min + alpha * features
            )
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
