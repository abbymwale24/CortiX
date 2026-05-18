"""
CortiX Module 2 — Metaplasticity Controller

Adapts the learning rate η dynamically based on recent activation variance.
Prevents both runaway plasticity and learning stagnation.

η(t) = η_0 / (1 + α * σ_y(t))
where σ_y(t) = activation variance in recent window
"""

import logging
from collections import deque

import numpy as np

from cortix.config import config

logger = logging.getLogger("cortix.snn.metaplasticity")


class MetaplasticityController:
    """
    Adaptive learning rate controller based on post-synaptic activation variance.

    When neurons are very active (high variance), learning rate decreases
    to prevent catastrophic forgetting. When activity is stable,
    learning rate increases to enable faster adaptation to new patterns.
    """

    def __init__(
        self,
        eta_0: float = None,
        alpha: float = None,
        window_size: int = 200,
        min_eta: float = 1e-6,
        max_eta: float = 0.01,
    ):
        """
        Args:
            eta_0: Base learning rate.
            alpha: Sensitivity to activation variance.
            window_size: Number of recent activations to consider.
            min_eta: Floor for learning rate.
            max_eta: Ceiling for learning rate.
        """
        self.eta_0 = eta_0 or config.HEBBIAN_LR
        self.alpha = alpha or config.METAPLASTICITY_ALPHA
        self.window_size = window_size
        self.min_eta = min_eta
        self.max_eta = max_eta

        self._activation_history: deque = deque(maxlen=window_size)
        self._current_eta = self.eta_0
        self._update_count = 0

    def update(self, post_activation: np.ndarray) -> float:
        """
        Record a new activation pattern and compute the adapted learning rate.

        Args:
            post_activation: Post-synaptic activation vector.

        Returns:
            Adapted learning rate η(t).
        """
        # Compute activation magnitude (L2 norm or variance)
        activation_magnitude = np.var(post_activation)
        self._activation_history.append(activation_magnitude)
        self._update_count += 1

        if len(self._activation_history) < 10:
            # Not enough history — use base rate
            self._current_eta = self.eta_0
            return self._current_eta

        # Compute variance of recent activation magnitudes
        sigma_y = np.std(list(self._activation_history))

        # Metaplasticity rule: η(t) = η_0 / (1 + α * σ_y(t))
        self._current_eta = self.eta_0 / (1.0 + self.alpha * sigma_y)

        # Clamp to [min_eta, max_eta]
        self._current_eta = np.clip(
            self._current_eta, self.min_eta, self.max_eta
        )

        return self._current_eta

    @property
    def eta(self) -> float:
        """Current adapted learning rate."""
        return self._current_eta

    @property
    def activation_variance(self) -> float:
        """Current activation variance (σ_y)."""
        if len(self._activation_history) < 2:
            return 0.0
        return float(np.std(list(self._activation_history)))

    def reset(self):
        """Reset metaplasticity state."""
        self._activation_history.clear()
        self._current_eta = self.eta_0
        self._update_count = 0

    def get_stats(self) -> dict:
        return {
            "eta": self._current_eta,
            "eta_0": self.eta_0,
            "alpha": self.alpha,
            "activation_variance": self.activation_variance,
            "history_size": len(self._activation_history),
            "updates": self._update_count,
        }
