"""
CortiX Module 2 — Hebbian Module

A single Spiking Neural Network module using unsupervised classical Hebbian 
learning, STDP trace updates, Oja's normalisation, and kWTA sparsity.
"""

import logging
import time
import numpy as np

from cortix.config import config
from cortix.snn.stdp import stdp_update_fast, oja_normalise, kwta

logger = logging.getLogger("cortix.snn.hebbian_module")


class HebbianModule:
    """
    A single neuro-inspired classification/anomaly detection module.
    
    Contains input-to-hidden synaptic weights W, eligibility traces,
    and runs local Hebbian/STDP learning.
    """

    def __init__(
        self,
        n_input: int = None,
        n_hidden: int = None,
        module_id: int = 0,
    ):
        self.n_input = n_input or config.NEURONS_PER_MODULE
        self.n_hidden = n_hidden or config.HIDDEN_NEURONS
        self.module_id = module_id

        # Initialize weights randomly, normalized to prevent runaway
        self.W = np.random.uniform(0.1, 0.5, (self.n_hidden, self.n_input)).astype(np.float32)
        # Normalize weights along the input dimension
        self.W /= np.sum(self.W, axis=1, keepdims=True) + 1e-8

        # Spike eligibility traces for online STDP
        self.pre_trace = np.zeros(self.n_input, dtype=np.float32)
        self.post_trace = np.zeros(self.n_hidden, dtype=np.float32)

        # Last spike time per neuron for exact calculations (optional)
        self.pre_times = np.zeros(self.n_input, dtype=np.float32)
        self.post_times = np.zeros(self.n_hidden, dtype=np.float32)

        logger.info(
            "HebbianModule [%d] initialised: %d input → %d hidden",
            self.module_id,
            self.n_input,
            self.n_hidden,
        )

    def forward(
        self,
        x: np.ndarray,
        t: float,
        eta: float = 0.001,
        learn: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Forward pass of a single Hebbian module.
        
        Args:
            x: Input binary spike vector of shape (n_input,)
            t: Current time in seconds
            eta: Learning rate (from metaplasticity controller)
            learn: If True, update weights online
            
        Returns:
            (post_spikes, activation_magnitude)
        """
        # Linear activation: W * x
        # x is binary (0 or 1 spikes)
        raw_activations = self.W @ x  # shape (n_hidden,)

        # Apply k-Winner-Take-All (10% sparsity)
        k = max(1, int(self.n_hidden * 0.1))
        winners = kwta(raw_activations, k)

        # Output is binary: did the neuron fire?
        post_spikes = (winners > 0).astype(np.float32)

        # Activation magnitude is the sum of raw winner activations
        activation_magnitude = float(np.sum(winners))

        if learn and np.any(post_spikes > 0) and np.any(x > 0):
            # 1. Update weights via fast trace-based STDP
            self.W, self.pre_trace, self.post_trace = stdp_update_fast(
                self.W,
                pre_spikes=x,
                post_spikes=post_spikes,
                current_time=t,
                pre_trace=self.pre_trace,
                post_trace=self.post_trace,
                A_plus=config.STDP_A_PLUS,
                A_minus=config.STDP_A_MINUS,
                tau_plus=config.STDP_TAU_PLUS,
                tau_minus=config.STDP_TAU_MINUS,
                dt=1e-3,  # standard step
            )

            # 2. Apply Oja normalization to stabilize weight growth
            self.W = oja_normalise(
                self.W,
                post_activation=winners,
                pre_input=x,
                eta=eta,
            )

            # Update spike times for records
            self.pre_times[x > 0] = t
            self.post_times[post_spikes > 0] = t

        return post_spikes, activation_magnitude

    def reset_traces(self):
        """Reset eligibility traces."""
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.pre_times.fill(0.0)
        self.post_times.fill(0.0)
