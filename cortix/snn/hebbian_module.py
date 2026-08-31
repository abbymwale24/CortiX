"""
CortiX Module 2 — Hebbian Module

A single Spiking Neural Network module using unsupervised classical Hebbian
learning, STDP trace updates, Oja's normalisation, and kWTA sparsity.

Scoring design
--------------
The anomaly score is derived entirely from post-kWTA (winner) statistics,
avoiding two failure modes of centroid-based scoring:

  1. Centroid instability during warmup — weights are still changing, so
     the accumulated centroid does not yet represent stable benign patterns.
  2. Cosine-distance collapse — when attack activations happen to point in
     roughly the same direction as benign ones, cos_dist ≈ 0 and the
     composite formula degenerates.

Instead we use two winner-derived signals:
  • winner_energy      = sum of top-k activation values
  • winner_concentration = max(winners) / (winner_energy + ε)

Combined score = winner_energy × (1 + winner_concentration)

This is always well-defined, requires no baseline accumulation, and
captures both the magnitude and the sharpness of the winning response.
Attack traffic fires different neurons at atypical weights, producing
a distinctly different energy × concentration value than benign traffic.

Reproducibility
----------------
Weight initialisation now uses a LOCAL np.random.Generator seeded
explicitly per-module, instead of the global unseeded np.random state.
This means:
  - Same seed -> identical initial W -> identical benchmark results.
  - Different modules (via HebbianEnsemble) get independent, decorrelated
    seeds (spawned from a single master SeedSequence), so the ensemble
    is not just 5 copies of the same random draw.
  - No dependence on process-start entropy, so runs are comparable
    across machines and across time.
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
        n_input: int | None = None,
        n_hidden: int | None = None,
        module_id: int = 0,
        seed: "int | np.random.SeedSequence | None" = None,
    ):
        self.n_input = n_input or config.NEURONS_PER_MODULE
        self.n_hidden = n_hidden or config.HIDDEN_NEURONS
        self.module_id = module_id

        # ── Reproducible, module-local RNG ──
        # Using a local Generator (not np.random.uniform, which draws from
        # global unseeded state) means:
        #   1. Weight init is fully determined by `seed`.
        #   2. Other libraries mutating np.random's global state (matplotlib,
        #      sklearn, etc.) can never perturb this module's randomness.
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.rng = self._rng

        # Initialize weights randomly, normalized to prevent runaway
        self.W = self._rng.uniform(0.1, 0.5, (self.n_hidden, self.n_input)).astype(np.float32)
        # Normalize weights along the input dimension
        self.W /= np.sum(self.W, axis=1, keepdims=True) + 1e-8

        # Spike eligibility traces for online STDP
        self.pre_trace = np.zeros(self.n_input, dtype=np.float32)
        self.post_trace = np.zeros(self.n_hidden, dtype=np.float32)

        # Last spike time per neuron for exact calculations (optional)
        self.pre_times = np.zeros(self.n_input, dtype=np.float32)
        self.post_times = np.zeros(self.n_hidden, dtype=np.float32)

        # Forward-pass counter (used for logging / warmup gating only).
        self._learn_count: int = 0

        logger.info(
            "HebbianModule [%d] initialised: %d input → %d hidden (seed=%s)",
            self.module_id,
            self.n_input,
            self.n_hidden,
            self.seed,
        )

    def forward(
        self,
        x: np.ndarray,
        t: float,
        eta: float = 0.001,
        learn: bool = True,
        neuromodulator_M: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """
        Forward pass of a single Hebbian module.

        Args:
            x: Input binary spike vector of shape (n_input,)
            t: Current time in seconds
            eta: Learning rate (from metaplasticity controller)
            learn: If True, update weights online
            neuromodulator_M: Factor 3 neuromodulator intensity (dopamine/danger signal)

        Returns:
            (post_spikes, activation_magnitude)
                activation_magnitude depends on config.SNN_SCORING_MODE:
                  "reconstruction": input-space reconstruction error
                  "winner_energy":  winner_energy × (1 + concentration)
        """
        # ── Linear activation: W * x ──
        # x is binary (0 or 1 spikes)
        raw_activations = self.W @ x  # shape (n_hidden,)

        # ── k-Winner-Take-All ──
        k = max(1, int(self.n_hidden * config.KWTA_SPARSITY))
        winners = kwta(raw_activations, k)

        # Output is binary: did the neuron fire?
        post_spikes = (winners > 0).astype(np.float32)

        # ── Anomaly score ──
        if config.SNN_SCORING_MODE == "reconstruction":
            # Cosine-distance reconstruction scoring.
            # x_hat = W.T @ winners is the network's "reconstruction" of the
            # input from its sparse hidden representation.
            # Cosine similarity is scale-invariant and measures directional
            # alignment: familiar traffic → high similarity → low score.
            x_hat = self.W.T @ winners          # (n_input,)
            dot = float(np.dot(x, x_hat))
            norm_x = float(np.linalg.norm(x)) + 1e-8
            norm_xhat = float(np.linalg.norm(x_hat)) + 1e-8
            cosine_sim = dot / (norm_x * norm_xhat)
            # Anomaly score: 1 - similarity (higher = more anomalous)
            activation_magnitude = 1.0 - cosine_sim
        else:
            # Original winner-energy scoring
            winner_energy = float(np.sum(winners))
            winner_max    = float(np.max(winners))
            concentration = winner_max / (winner_energy + 1e-8)
            activation_magnitude = winner_energy * (1.0 + concentration)

        # ── Count learning steps ──
        if learn:
            self._learn_count += 1

        if learn and np.any(post_spikes > 0) and np.any(x > 0):
            # Effective learning rate modulated by Factor 3 (Neuromodulator)
            effective_eta = (
                eta * neuromodulator_M
                if config.NEUROMODULATION_ENABLED
                else eta
            )

            if effective_eta > 1e-7:
                # 1. Update weights via fast trace-based STDP (amplitudes scaled by neuromodulator)
                m_scale = neuromodulator_M if config.NEUROMODULATION_ENABLED else 1.0
                self.W, self.pre_trace, self.post_trace = stdp_update_fast(
                    self.W,
                    pre_spikes=x,
                    post_spikes=post_spikes,
                    current_time=t,
                    pre_trace=self.pre_trace,
                    post_trace=self.post_trace,
                    A_plus=config.STDP_A_PLUS * m_scale,
                    A_minus=config.STDP_A_MINUS * m_scale,
                    tau_plus=config.STDP_TAU_PLUS,
                    tau_minus=config.STDP_TAU_MINUS,
                    dt=1e-3,  # standard step
                )

                # 2. Apply Oja normalisation to stabilise weight growth.
                self.W = oja_normalise(
                    self.W,
                    post_activation=winners,
                    pre_input=x,
                    eta=effective_eta,
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
        self._learn_count = 0
