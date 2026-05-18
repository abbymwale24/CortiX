"""
CortiX Module 2 — STDP Learning Rules

Implements Spike-Timing Dependent Plasticity (STDP) and Oja normalisation
for online unsupervised weight updates in the Hebbian modules.
"""

import logging
import numpy as np

logger = logging.getLogger("cortix.snn.stdp")


def stdp_update(
    W: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    pre_times: np.ndarray,
    post_times: np.ndarray,
    A_plus: float = 0.03,
    A_minus: float = 0.035,
    tau_plus: float = 20e-3,
    tau_minus: float = 20e-3,
) -> np.ndarray:
    """
    Apply STDP weight update rule.

    Δw_ij = A+ * exp(-Δt / τ+)  if Δt > 0  (pre before post → potentiate)
    Δw_ij = -A- * exp(Δt / τ-)  if Δt < 0  (post before pre → depress)

    Args:
        W: Weight matrix (n_hidden, n_input)
        pre_spikes: Binary pre-synaptic spike vector (n_input,)
        post_spikes: Binary post-synaptic spike vector (n_hidden,)
        pre_times: Spike times for pre-synaptic neurons (n_input,)
        post_times: Spike times for post-synaptic neurons (n_hidden,)
        A_plus: LTP amplitude
        A_minus: LTD amplitude
        tau_plus: LTP time constant
        tau_minus: LTD time constant

    Returns:
        Updated weight matrix.
    """
    # Find active pre and post synaptic neurons
    pre_active = np.where(pre_spikes > 0)[0]
    post_active = np.where(post_spikes > 0)[0]

    if len(pre_active) == 0 or len(post_active) == 0:
        return W

    dW = np.zeros_like(W)

    for j in post_active:
        for i in pre_active:
            dt = post_times[j] - pre_times[i]

            if dt > 0:
                # Pre before post → Long-Term Potentiation (LTP)
                dW[j, i] += A_plus * np.exp(-dt / tau_plus)
            elif dt < 0:
                # Post before pre → Long-Term Depression (LTD)
                dW[j, i] -= A_minus * np.exp(dt / tau_minus)

    W += dW

    # Clamp weights to [0, 1]
    np.clip(W, 0.0, 1.0, out=W)

    return W


def stdp_update_fast(
    W: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    current_time: float,
    pre_trace: np.ndarray,
    post_trace: np.ndarray,
    A_plus: float = 0.03,
    A_minus: float = 0.035,
    tau_plus: float = 20e-3,
    tau_minus: float = 20e-3,
    dt: float = 1e-3,
) -> tuple:
    """
    Fast vectorised STDP using eligibility traces (online approximation).

    Instead of tracking individual spike times, we maintain exponentially
    decaying traces for each neuron.

    Args:
        W: Weight matrix (n_hidden, n_input)
        pre_spikes: Binary pre-synaptic spikes (n_input,)
        post_spikes: Binary post-synaptic spikes (n_hidden,)
        current_time: Current simulation time
        pre_trace: Pre-synaptic eligibility trace (n_input,)
        post_trace: Post-synaptic eligibility trace (n_hidden,)
        dt: Simulation timestep

    Returns:
        (updated W, updated pre_trace, updated post_trace)
    """
    # Decay traces
    decay_pre = np.exp(-dt / tau_plus)
    decay_post = np.exp(-dt / tau_minus)

    pre_trace *= decay_pre
    post_trace *= decay_post

    # Update traces with new spikes
    pre_trace += pre_spikes
    post_trace += post_spikes

    # LTP: post spike → strengthen connections from recently active pre neurons
    if np.any(post_spikes > 0):
        post_idx = post_spikes > 0
        W[post_idx] += A_plus * np.outer(
            post_spikes[post_idx], pre_trace
        )

    # LTD: pre spike → weaken connections to recently active post neurons
    if np.any(pre_spikes > 0):
        dW_ltd = A_minus * np.outer(post_trace, pre_spikes)
        W -= dW_ltd

    # Clamp weights
    np.clip(W, 0.0, 1.0, out=W)

    return W, pre_trace, post_trace


def oja_normalise(
    W: np.ndarray,
    post_activation: np.ndarray,
    pre_input: np.ndarray,
    eta: float = 0.001,
) -> np.ndarray:
    """
    Oja's normalisation rule — prevents weight runaway.

    Δw_ij = η * (x_i * y_j - y_j² * w_ij)

    This drives the weight vector toward the principal component
    of the input distribution while keeping ||w|| bounded.

    Args:
        W: Weight matrix (n_hidden, n_input)
        post_activation: Post-synaptic activations (n_hidden,)
        pre_input: Pre-synaptic input (n_input,)
        eta: Learning rate

    Returns:
        Updated weight matrix.
    """
    y = post_activation.reshape(-1, 1)  # (n_hidden, 1)
    x = pre_input.reshape(1, -1)       # (1, n_input)

    # Oja rule: Δw = η * y * (x - y * w)
    dW = eta * (y @ x - (y ** 2) * W)

    W += dW
    np.clip(W, 0.0, 1.0, out=W)

    return W


def kwta(activations: np.ndarray, k: int) -> np.ndarray:
    """
    k-Winner-Take-All sparsification.

    Only the top-k neurons remain active; all others are zeroed.

    Args:
        activations: Neuron activation values (n_hidden,)
        k: Number of winners to keep

    Returns:
        Sparse activation vector with only k non-zero entries.
    """
    result = np.zeros_like(activations)
    if k >= len(activations):
        return activations.copy()

    # Find indices of top-k activations
    top_k_idx = np.argpartition(activations, -k)[-k:]
    result[top_k_idx] = activations[top_k_idx]

    return result
