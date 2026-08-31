"""
CortiX — Unit Tests for SNN Core Modules

Tests the Hebbian module, STDP learning rules, spike encoder,
anomaly scorer, metaplasticity controller, ensemble, and consolidator.
"""

import time
import numpy as np
import pytest

from cortix.snn.stdp import stdp_update, stdp_update_fast, oja_normalise, kwta
from cortix.snn.hebbian_module import HebbianModule
from cortix.snn.ensemble import HebbianEnsemble
from cortix.snn.scorer import AnomalyScorer
from cortix.snn.metaplasticity import MetaplasticityController
from cortix.snn.consolidator import PrototypeConsolidator
from cortix.preprocessor.encoder import SpikeEncoder


# ──────────────────────────────────────────────
# STDP Learning Rules
# ──────────────────────────────────────────────

class TestSTDP:
    """Test Spike-Timing Dependent Plasticity rules."""

    def test_stdp_update_potentiates_causal_pairs(self):
        """Pre-before-post (causal) should strengthen weights (LTP)."""
        W = np.full((2, 3), 0.5, dtype=np.float32)
        pre_spikes = np.array([1, 0, 1], dtype=np.float32)
        post_spikes = np.array([1, 0], dtype=np.float32)
        pre_times = np.array([0.010, 0.0, 0.010], dtype=np.float32)   # Pre at 10ms
        post_times = np.array([0.015, 0.0], dtype=np.float32)          # Post at 15ms (after pre)

        W_updated = stdp_update(W, pre_spikes, post_spikes, pre_times, post_times)

        # Post neuron 0 fired after pre neurons 0 and 2 → weights should increase
        assert W_updated[0, 0] > 0.5, "LTP should increase weight for causal pair"
        assert W_updated[0, 2] > 0.5, "LTP should increase weight for causal pair"
        # Non-active connections should remain unchanged
        assert W_updated[1, 0] == pytest.approx(0.5), "Inactive post neuron weight should not change"

    def test_stdp_update_depresses_anticausal_pairs(self):
        """Post-before-pre (anti-causal) should weaken weights (LTD)."""
        W = np.full((2, 3), 0.5, dtype=np.float32)
        pre_spikes = np.array([1, 0, 0], dtype=np.float32)
        post_spikes = np.array([1, 0], dtype=np.float32)
        pre_times = np.array([0.020, 0.0, 0.0], dtype=np.float32)      # Pre at 20ms
        post_times = np.array([0.010, 0.0], dtype=np.float32)           # Post at 10ms (before pre)

        W_updated = stdp_update(W, pre_spikes, post_spikes, pre_times, post_times)

        # Post fired before pre → weight should decrease (LTD)
        assert W_updated[0, 0] < 0.5, "LTD should decrease weight for anti-causal pair"

    def test_stdp_weights_clamped_0_1(self):
        """Weights must stay in [0, 1] range after STDP update."""
        W = np.full((3, 4), 0.95, dtype=np.float32)
        pre_spikes = np.array([1, 1, 1, 1], dtype=np.float32)
        post_spikes = np.array([1, 1, 1], dtype=np.float32)
        pre_times = np.array([0.01] * 4, dtype=np.float32)
        post_times = np.array([0.02] * 3, dtype=np.float32)

        W_updated = stdp_update(W, pre_spikes, post_spikes, pre_times, post_times,
                                A_plus=0.5)  # Large A_plus to force exceeding 1.0

        assert np.all(W_updated >= 0.0), "Weights must be >= 0"
        assert np.all(W_updated <= 1.0), "Weights must be <= 1"

    def test_stdp_no_spikes_no_change(self):
        """If no neurons fire, weights should remain unchanged."""
        W = np.full((2, 3), 0.5, dtype=np.float32)
        pre_spikes = np.zeros(3, dtype=np.float32)
        post_spikes = np.zeros(2, dtype=np.float32)

        W_updated = stdp_update(W, pre_spikes, post_spikes,
                                np.zeros(3), np.zeros(2))

        np.testing.assert_array_equal(W, W_updated)


class TestSTDPFast:
    """Test trace-based fast STDP (online approximation)."""

    def test_fast_stdp_updates_traces(self):
        """Traces should increase when neurons spike."""
        W = np.full((2, 3), 0.5, dtype=np.float32)
        pre_trace = np.zeros(3, dtype=np.float32)
        post_trace = np.zeros(2, dtype=np.float32)

        pre_spikes = np.array([1, 0, 1], dtype=np.float32)
        post_spikes = np.array([1, 0], dtype=np.float32)

        _, new_pre_trace, new_post_trace = stdp_update_fast(
            W, pre_spikes, post_spikes, current_time=0.0,
            pre_trace=pre_trace, post_trace=post_trace,
        )

        assert new_pre_trace[0] > 0, "Pre trace should increase for spiking neuron"
        assert new_pre_trace[2] > 0, "Pre trace should increase for spiking neuron"
        assert new_post_trace[0] > 0, "Post trace should increase for spiking neuron"
        # Non-spiking neurons should have decayed traces (from 0, still ~0)
        assert new_pre_trace[1] == pytest.approx(0.0, abs=1e-6)

    def test_fast_stdp_traces_decay(self):
        """Traces should decay exponentially without new spikes."""
        W = np.full((2, 3), 0.5, dtype=np.float32)
        pre_trace = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        post_trace = np.array([1.0, 1.0], dtype=np.float32)

        no_spikes_pre = np.zeros(3, dtype=np.float32)
        no_spikes_post = np.zeros(2, dtype=np.float32)

        _, new_pre, new_post = stdp_update_fast(
            W, no_spikes_pre, no_spikes_post, current_time=0.001,
            pre_trace=pre_trace, post_trace=post_trace,
        )

        assert np.all(new_pre < 1.0), "Traces should decay without spikes"
        assert np.all(new_post < 1.0), "Traces should decay without spikes"

    def test_fast_stdp_weights_clamped(self):
        """Fast STDP should also clamp weights to [0, 1]."""
        W = np.full((2, 3), 0.98, dtype=np.float32)
        pre_trace = np.zeros(3, dtype=np.float32)
        post_trace = np.zeros(2, dtype=np.float32)

        pre_spikes = np.ones(3, dtype=np.float32)
        post_spikes = np.ones(2, dtype=np.float32)

        W_new, _, _ = stdp_update_fast(
            W, pre_spikes, post_spikes, current_time=0.0,
            pre_trace=pre_trace, post_trace=post_trace,
            A_plus=0.5,
        )

        assert np.all(W_new >= 0.0)
        assert np.all(W_new <= 1.0)


class TestOjaNormalise:
    """Test Oja's normalisation rule."""

    def test_oja_bounded_weights(self):
        """Oja normalisation should keep weights bounded."""
        W = np.random.uniform(0.3, 0.7, (4, 5)).astype(np.float32)
        post = np.random.uniform(0, 1, 4).astype(np.float32)
        pre = np.random.uniform(0, 1, 5).astype(np.float32)

        W_updated = oja_normalise(W, post, pre, eta=0.01)

        assert np.all(W_updated >= 0.0)
        assert np.all(W_updated <= 1.0)

    def test_oja_converges_to_pca(self):
        """After many updates with the same input, weight vectors should stabilise."""
        W = np.random.uniform(0.1, 0.9, (2, 3)).astype(np.float32)
        x = np.array([0.8, 0.1, 0.3], dtype=np.float32)

        for _ in range(500):
            y = W @ x
            W = oja_normalise(W, y, x, eta=0.005)

        # After convergence, norms should be stable
        norms = np.linalg.norm(W, axis=1)
        # Do another round
        y = W @ x
        W2 = oja_normalise(W, y, x, eta=0.005)
        norms2 = np.linalg.norm(W2, axis=1)

        np.testing.assert_allclose(norms, norms2, atol=0.05,
                                   err_msg="Oja weights should stabilise")


class TestKWTA:
    """Test k-Winner-Take-All sparsification."""

    def test_kwta_correct_sparsity(self):
        """Only k neurons should be non-zero after kWTA."""
        activations = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7, 0.6, 0.15],
                               dtype=np.float32)
        k = 3
        result = kwta(activations, k)

        assert np.count_nonzero(result) == k, f"Expected {k} non-zero, got {np.count_nonzero(result)}"

    def test_kwta_keeps_top_values(self):
        """kWTA should keep the top-k activation values."""
        activations = np.array([0.1, 0.9, 0.5, 0.8], dtype=np.float32)
        k = 2
        result = kwta(activations, k)

        # Top-2 are indices 1 (0.9) and 3 (0.8)
        assert result[1] == pytest.approx(0.9)
        assert result[3] == pytest.approx(0.8)
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_kwta_k_equals_length(self):
        """When k >= len, all activations should pass through."""
        activations = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        result = kwta(activations, k=3)
        np.testing.assert_array_almost_equal(result, activations)

    def test_kwta_k_one(self):
        """k=1 should keep only the single highest activation."""
        activations = np.array([0.2, 0.8, 0.5], dtype=np.float32)
        result = kwta(activations, k=1)
        assert np.count_nonzero(result) == 1
        assert result[1] == pytest.approx(0.8)


# ──────────────────────────────────────────────
# Hebbian Module
# ──────────────────────────────────────────────

class TestHebbianModule:
    """Test the HebbianModule forward pass."""

    def test_forward_output_shape(self):
        """Forward pass should return correct output shapes."""
        module = HebbianModule(n_input=32, n_hidden=16, module_id=0)
        x = np.random.randint(0, 2, 32).astype(np.float32)
        post_spikes, activation_mag = module.forward(x, t=0.0)

        assert post_spikes.shape == (16,), f"Expected shape (16,), got {post_spikes.shape}"
        assert isinstance(activation_mag, float)

    def test_forward_binary_output(self):
        """Post-synaptic spikes should be binary (0 or 1)."""
        module = HebbianModule(n_input=32, n_hidden=16, module_id=0)
        x = np.random.randint(0, 2, 32).astype(np.float32)
        post_spikes, _ = module.forward(x, t=0.0)

        unique_vals = set(post_spikes.tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Spikes must be binary, got {unique_vals}"

    def test_forward_sparsity(self):
        """kWTA should enforce ~10% sparsity in the hidden layer."""
        module = HebbianModule(n_input=64, n_hidden=100, module_id=0)
        x = np.random.randint(0, 2, 64).astype(np.float32)
        post_spikes, _ = module.forward(x, t=0.0)

        active = np.count_nonzero(post_spikes)
        expected_k = max(1, int(100 * 0.1))  # 10 neurons
        assert active == expected_k, f"Expected {expected_k} active, got {active}"

    def test_forward_no_learn(self):
        """When learn=False, weights should not change."""
        module = HebbianModule(n_input=32, n_hidden=16, module_id=0)
        W_before = module.W.copy()
        x = np.random.randint(0, 2, 32).astype(np.float32)

        module.forward(x, t=0.0, learn=False)
        np.testing.assert_array_equal(W_before, module.W,
                                      err_msg="Weights should not change with learn=False")

    def test_forward_learning_changes_weights(self):
        """When learn=True and there are spikes, weights should update."""
        module = HebbianModule(n_input=32, n_hidden=16, module_id=0)
        W_before = module.W.copy()

        # Use a dense input to guarantee some post-synaptic activity
        x = np.ones(32, dtype=np.float32)
        module.forward(x, t=0.0, eta=0.01, learn=True)

        assert not np.array_equal(W_before, module.W), \
            "Weights should change after learning with active spikes"

    def test_reset_traces(self):
        """Reset should zero all traces."""
        module = HebbianModule(n_input=32, n_hidden=16, module_id=0)

        # Run a forward pass to populate traces
        x = np.ones(32, dtype=np.float32)
        module.forward(x, t=1.0)

        module.reset_traces()

        np.testing.assert_array_equal(module.pre_trace, np.zeros(32))
        np.testing.assert_array_equal(module.post_trace, np.zeros(16))
        np.testing.assert_array_equal(module.pre_times, np.zeros(32))
        np.testing.assert_array_equal(module.post_times, np.zeros(16))

    def test_seeded_rng_reproducibility(self):
        """HebbianModules initialized with the same seed must have identical initial weights."""
        m1 = HebbianModule(n_input=32, n_hidden=16, module_id=0, seed=1234)
        m2 = HebbianModule(n_input=32, n_hidden=16, module_id=0, seed=1234)
        np.testing.assert_array_equal(m1.W, m2.W)

    def test_different_seeds_produce_different_weights(self):
        """HebbianModules initialized with different seeds/module_ids must produce different weights."""
        m1 = HebbianModule(n_input=32, n_hidden=16, module_id=0)
        m2 = HebbianModule(n_input=32, n_hidden=16, module_id=1)
        assert not np.array_equal(m1.W, m2.W)


# ──────────────────────────────────────────────
# Anomaly Scorer
# ──────────────────────────────────────────────

class TestAnomalyScorer:
    """Test MAD-based anomaly scoring."""

    def test_warmup_phase_not_anomaly(self):
        """During warmup (< 20 samples), nothing should be flagged as anomaly."""
        scorer = AnomalyScorer(window_size=100, z_threshold=3.5)

        for i in range(19):
            result = scorer.score(float(i))
            assert result["warming_up"] is True
            assert result["is_anomaly"] is False

    def test_normal_traffic_not_anomaly(self):
        """Consistent normal values should not trigger anomaly."""
        scorer = AnomalyScorer(window_size=100, z_threshold=3.5)

        # Build baseline
        for _ in range(50):
            scorer.score(1.0)

        # Test with normal value
        result = scorer.score(1.0)
        assert result["is_anomaly"] is False

    def test_extreme_value_is_anomaly(self):
        """A value far from baseline should trigger anomaly."""
        scorer = AnomalyScorer(window_size=100, z_threshold=3.0)

        # Build baseline with stable values
        for _ in range(50):
            scorer.score(1.0)

        # Inject extreme outlier
        result = scorer.score(100.0)
        assert result["is_anomaly"] is True
        assert result["z_score"] > 3.0

    def test_majority_vote_consensus(self):
        """Majority voting should require > 50% agreement."""
        scorer = AnomalyScorer(window_size=100, z_threshold=3.0)

        # Build baseline
        for _ in range(50):
            scorer.score(1.0)

        # 5 modules: 3 anomalous, 2 normal → majority is anomaly
        scores = [1.0, 100.0, 100.0, 100.0, 1.0]
        result = scorer.majority_vote(scores)

        assert result["anomaly_votes"] >= 3, "At least 3 of 5 should flag anomaly"
        assert result["total_modules"] == 5

    def test_majority_vote_no_consensus(self):
        """If minority flags anomaly, consensus should be benign."""
        scorer = AnomalyScorer(window_size=100, z_threshold=3.0)

        # Build baseline
        for _ in range(50):
            scorer.score(1.0)

        # Only 1 of 5 is anomalous → no majority
        scores = [1.0, 1.0, 1.0, 100.0, 1.0]
        result = scorer.majority_vote(scores)

        assert result["anomaly_votes"] <= 2

    def test_per_context_baselines(self):
        """Different contexts should maintain independent baselines."""
        scorer = AnomalyScorer(window_size=100, z_threshold=3.0)

        # Build baseline for context A
        for _ in range(30):
            scorer.score(1.0, context_key="subnet_A")

        # Build baseline for context B with higher values
        for _ in range(30):
            scorer.score(10.0, context_key="subnet_B")

        # Value 10.0 is anomalous in context A but normal in context B
        result_a = scorer.score(10.0, context_key="subnet_A")
        result_b = scorer.score(10.0, context_key="subnet_B")

        assert result_a["z_score"] > result_b["z_score"], \
            "10.0 should have higher z-score in subnet_A (baseline ~1.0) than subnet_B (baseline ~10.0)"

    def test_latency_tracking(self):
        """Latency stats should be computed correctly."""
        scorer = AnomalyScorer()
        for i in range(100):
            scorer.log_latency(float(i))

        stats = scorer.get_latency_stats()
        assert stats["count"] == 100
        assert stats["p50_ms"] == pytest.approx(49.5, abs=1.0)
        assert stats["min_ms"] == pytest.approx(0.0)
        assert stats["max_ms"] == pytest.approx(99.0)

    def test_reset_clears_state(self):
        """Reset should clear all windows and latencies."""
        scorer = AnomalyScorer()
        for _ in range(50):
            scorer.score(1.0)
        scorer.log_latency(5.0)

        scorer.reset()

        stats = scorer.get_latency_stats()
        assert stats["count"] == 0


# ──────────────────────────────────────────────
# Metaplasticity Controller
# ──────────────────────────────────────────────

class TestMetaplasticityController:
    """Test adaptive learning rate controller."""

    def test_initial_eta_is_base_rate(self):
        """Initial learning rate should be the base rate."""
        controller = MetaplasticityController(eta_0=0.001, alpha=10.0)
        assert controller.eta == pytest.approx(0.001)

    def test_eta_decreases_with_high_variance(self):
        """High activation variance should reduce learning rate."""
        controller = MetaplasticityController(eta_0=0.01, alpha=10.0, window_size=50)

        # Feed stable activations first
        for _ in range(20):
            controller.update(np.array([0.5, 0.5, 0.5]))

        stable_eta = controller.eta

        # Now feed high-variance activations
        for i in range(30):
            controller.update(np.random.uniform(0, 1, 3).astype(np.float32))

        high_var_eta = controller.eta

        assert high_var_eta < stable_eta, \
            f"η should decrease with high variance: {high_var_eta} >= {stable_eta}"

    def test_eta_clamped(self):
        """Learning rate should stay within [min_eta, max_eta]."""
        controller = MetaplasticityController(
            eta_0=0.001, alpha=10.0, min_eta=1e-6, max_eta=0.01
        )

        # Drive with extreme variance
        for _ in range(100):
            controller.update(np.random.uniform(0, 100, 10).astype(np.float32))

        assert controller.eta >= 1e-6
        assert controller.eta <= 0.01

    def test_reset(self):
        """Reset should restore base learning rate."""
        controller = MetaplasticityController(eta_0=0.005, alpha=10.0)

        for _ in range(50):
            controller.update(np.random.uniform(0, 1, 5).astype(np.float32))

        controller.reset()
        assert controller.eta == pytest.approx(0.005)

    def test_stats_output(self):
        """get_stats should return valid dictionary."""
        controller = MetaplasticityController(eta_0=0.001, alpha=5.0)
        stats = controller.get_stats()

        assert "eta" in stats
        assert "eta_0" in stats
        assert "alpha" in stats
        assert "activation_variance" in stats
        assert "updates" in stats
        assert stats["eta_0"] == 0.001


# ──────────────────────────────────────────────
# Hebbian Ensemble
# ──────────────────────────────────────────────

class TestHebbianEnsemble:
    """Test the M-module Hebbian ensemble."""

    def test_ensemble_creates_correct_modules(self):
        """Ensemble should create M independent modules."""
        ensemble = HebbianEnsemble(M=3)
        assert len(ensemble.modules) == 3
        assert ensemble.M == 3

    def test_process_event_returns_valid_result(self):
        """process_event should return a complete result dictionary."""
        ensemble = HebbianEnsemble(M=3)
        x = np.random.randint(0, 2, 512).astype(np.float32)

        result = ensemble.process_event(x)

        assert "timestamp" in result
        assert "latency_ms" in result
        assert "consensus_score" in result
        assert "is_anomaly" in result
        assert "z_score" in result
        assert "votes" in result
        assert "total_modules" in result
        assert "vote_ratio" in result
        assert "learning_rate" in result
        assert "warming_up" in result

    def test_process_event_increments_counter(self):
        """Event processing should increment the counter."""
        ensemble = HebbianEnsemble(M=2)
        x = np.random.randint(0, 2, 512).astype(np.float32)

        assert ensemble.total_processed == 0
        ensemble.process_event(x)
        assert ensemble.total_processed == 1
        ensemble.process_event(x)
        assert ensemble.total_processed == 2

    def test_warmup_phase(self):
        """First 50 events should report warming_up=True."""
        ensemble = HebbianEnsemble(M=2)
        x = np.random.randint(0, 2, 512).astype(np.float32)

        result = ensemble.process_event(x)
        assert result["warming_up"] is True

    def test_after_warmup(self):
        """After 50 events, warming_up should be False."""
        ensemble = HebbianEnsemble(M=2)
        x = np.random.randint(0, 2, 512).astype(np.float32)

        for _ in range(51):
            result = ensemble.process_event(x)

        assert result["warming_up"] is False

    def test_reset_clears_all_state(self):
        """Reset should zero counters and clear module traces."""
        ensemble = HebbianEnsemble(M=2)
        x = np.ones(512, dtype=np.float32)

        for _ in range(10):
            ensemble.process_event(x)

        ensemble.reset()
        assert ensemble.total_processed == 0

    def test_ensemble_seed_reproducibility(self):
        """Ensembles with the same seed must produce identical weights across all modules."""
        e1 = HebbianEnsemble(M=3, seed=42)
        e2 = HebbianEnsemble(M=3, seed=42)
        for m1, m2 in zip(e1.modules, e2.modules):
            np.testing.assert_array_equal(m1.W, m2.W)


# ──────────────────────────────────────────────
# Spike Encoder
# ──────────────────────────────────────────────

class TestSpikeEncoder:
    """Test Gaussian population coding spike encoder."""

    def test_output_shape(self):
        """Encoder output should match expected neuron count."""
        encoder = SpikeEncoder(num_features=16, num_neurons=512)
        features = np.random.uniform(0, 1, 16).astype(np.float32)
        spikes = encoder.encode(features)

        assert spikes.shape == (512,), f"Expected (512,), got {spikes.shape}"

    def test_output_binary(self):
        """Spike output should be binary (0 or 1)."""
        encoder = SpikeEncoder(num_features=8, num_neurons=64)
        features = np.random.uniform(0, 1, 8).astype(np.float32)
        spikes = encoder.encode(features)

        unique = set(spikes.tolist())
        assert unique.issubset({0.0, 1.0}), f"Expected binary, got {unique}"

    def test_deterministic_output(self):
        """Deterministic encoding should be repeatable."""
        encoder = SpikeEncoder(num_features=8, num_neurons=64)
        features = np.array([0.5] * 8, dtype=np.float32)

        spikes1 = encoder.encode_deterministic(features)
        spikes2 = encoder.encode_deterministic(features)

        np.testing.assert_array_equal(spikes1, spikes2)

    def test_different_inputs_different_spikes(self):
        """Different input features should produce different spike patterns."""
        encoder = SpikeEncoder(num_features=8, num_neurons=64)

        low = encoder.encode_deterministic(np.full(8, 0.1, dtype=np.float32))
        high = encoder.encode_deterministic(np.full(8, 0.9, dtype=np.float32))

        assert not np.array_equal(low, high), "Different inputs should produce different spikes"

    def test_batch_encoding(self):
        """Batch encoding should handle multiple samples."""
        encoder = SpikeEncoder(num_features=8, num_neurons=64)
        batch = np.random.uniform(0, 1, (10, 8)).astype(np.float32)

        result = encoder.encode_batch(batch)
        assert result.shape == (10, 64), f"Expected (10, 64), got {result.shape}"

    def test_normalisation_warmup(self):
        """Online normalisation should adapt during warmup period."""
        encoder = SpikeEncoder(num_features=4, num_neurons=32)

        # Feed values to update normalisation state
        for _ in range(50):
            encoder.encode(np.array([100, 200, 300, 400], dtype=np.float32))

        # After warmup, min/max should have adapted
        assert encoder._feature_min[0] < 100, "Min should track below initial values"

    def test_reset_normalisation(self):
        """Resetting normalisation should restore initial state."""
        encoder = SpikeEncoder(num_features=4, num_neurons=32)

        for _ in range(20):
            encoder.encode(np.array([10, 20, 30, 40], dtype=np.float32))

        encoder.reset_normalisation()
        assert encoder._update_count == 0
        np.testing.assert_array_equal(encoder._feature_min, np.zeros(4))

    def test_thalamic_gate_zero_suppression(self):
        """Thalamic gate should suppress inactive/dummy zero features completely."""
        encoder = SpikeEncoder(num_features=10, num_neurons=40, thalamic_gate=True)
        # All zeros
        features = np.zeros(10, dtype=np.float32)
        spikes = encoder.encode(features)
        assert np.count_nonzero(spikes) == 0, "Zero features should produce 0 spikes under Thalamic gating"

    def test_thalamic_gate_sparsity(self):
        """Thalamic gate should enforce controlled sparsity with adaptive multi-spike gating."""
        encoder = SpikeEncoder(num_features=119, num_neurons=512, thalamic_gate=True)
        # Typical sparse NSL-KDD vector: 30 non-zero continuous features, 89 zeros
        features = np.zeros(119, dtype=np.float32)
        features[:30] = np.random.uniform(0.1, 1.0, 30).astype(np.float32)
        spikes = encoder.encode(features)

        active_count = np.count_nonzero(spikes)
        sparsity = active_count / encoder.num_neurons
        top_k = encoder.effective_top_k
        expected = 30 * min(top_k, encoder.neurons_per_feature)
        assert active_count == expected, f"Expected {expected} active spikes (30 features × top-{top_k}), got {active_count}"
        assert sparsity <= 0.25, f"Expected sparsity <= 25%, got {sparsity:.2%}"


# ──────────────────────────────────────────────
# Prototype Consolidator
# ──────────────────────────────────────────────

class TestPrototypeConsolidator:
    """Test periodic synaptic consolidation."""

    def test_consolidation_runs_without_error(self):
        """Consolidation should complete without exceptions."""
        ensemble = HebbianEnsemble(M=2)

        # Run some events first to create learned patterns
        for _ in range(20):
            x = np.random.randint(0, 2, 512).astype(np.float32)
            ensemble.process_event(x)

        consolidator = PrototypeConsolidator(ensemble)
        result = consolidator.consolidate()

        assert "pruned_synapses" in result
        assert "recovered_neurons" in result
        assert "modules_cleaned" in result
        assert result["modules_cleaned"] == 2

    def test_weights_normalised_after_consolidation(self):
        """After consolidation, weights should be finite and non-negative."""
        ensemble = HebbianEnsemble(M=2)

        for _ in range(20):
            x = np.random.randint(0, 2, 512).astype(np.float32)
            ensemble.process_event(x)

        consolidator = PrototypeConsolidator(ensemble)
        consolidator.consolidate()

        for module in ensemble.modules:
            # Weights should be finite and non-negative after consolidation
            assert np.all(np.isfinite(module.W)), "Weights should be finite"
            assert np.all(module.W >= 0.0), "Weights should be non-negative"

    def test_dead_neurons_recovered(self):
        """Neurons with very low total weights should be detected for recovery."""
        ensemble = HebbianEnsemble(M=1)
        # Manually zero out some neurons to simulate dead neurons
        ensemble.modules[0].W[0, :] = 0.0
        ensemble.modules[0].W[1, :] = 0.0

        consolidator = PrototypeConsolidator(ensemble)
        result = consolidator.consolidate()

        # The consolidator should have detected and attempted recovery
        assert result["recovered_neurons"] >= 2, \
            f"Expected at least 2 recovered neurons, got {result['recovered_neurons']}"
