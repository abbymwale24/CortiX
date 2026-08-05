"""
CortiX — Integration Test: Full Pipeline End-to-End

Tests the complete flow from raw feature vector → spike encoding →
SNN ensemble detection → classifier inference → containment agent decision.
Validates that all modules work together correctly.
"""

import time
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for integration tests")

from cortix.preprocessor.encoder import SpikeEncoder
from cortix.preprocessor.features import FlowAggregator, FLOW_FEATURE_NAMES
from cortix.snn.ensemble import HebbianEnsemble
from cortix.snn.consolidator import PrototypeConsolidator
from cortix.classifier.inference import ClassifierInference
from cortix.containment.agent import ContainmentAgent


class TestFullPipeline:
    """End-to-end integration tests for the CortiX detection pipeline."""

    def setup_method(self):
        """Set up shared pipeline components."""
        self.encoder = SpikeEncoder(num_features=16, num_neurons=512)
        self.ensemble = HebbianEnsemble(M=3)
        self.classifier = ClassifierInference(model_path="/nonexistent/path")
        self.rl_agent = ContainmentAgent(agent_path="/nonexistent/path")

    def test_benign_traffic_pipeline(self):
        """Benign traffic should flow through without triggering containment."""
        # Simulate 100 benign flows to build baseline
        for _ in range(100):
            features = np.random.uniform(0.2, 0.4, 16).astype(np.float32)
            spikes = self.encoder.encode(features)
            result = self.ensemble.process_event(spikes)

        # After baseline, normal traffic should not be anomalous
        features = np.random.uniform(0.25, 0.35, 16).astype(np.float32)
        spikes = self.encoder.encode(features)
        snn_result = self.ensemble.process_event(spikes)

        # Classifier mock returns BENIGN
        feature_40 = np.zeros(40, dtype=np.float32)
        feature_40[:16] = features
        # Fill buffer first
        for _ in range(10):
            clf_result = self.classifier.predict_flow("10.0.0.1", feature_40)

        # RL agent should ALLOW benign
        action = self.rl_agent.select_action(
            z_score=snn_result["z_score"],
            confidence=clf_result["confidence"],
            predicted_class=clf_result["class"],
            rolling_fpr=0.01,
            reputation=0.1,
            volume_percentile=0.2,
            time_since_last_alert=1.0,
        )

        assert action == 0, "Benign traffic should result in ALLOW (0)"

    def test_attack_traffic_pipeline(self):
        """Attack-like traffic should trigger containment."""
        # Build baseline with benign traffic
        for _ in range(100):
            features = np.random.uniform(0.2, 0.4, 16).astype(np.float32)
            spikes = self.encoder.encode(features)
            self.ensemble.process_event(spikes)

        # Use the RL agent heuristic (no model loaded)
        action = self.rl_agent.select_action(
            z_score=12.0,
            confidence=0.95,
            predicted_class="DDoS",
            rolling_fpr=0.01,
            reputation=0.9,
            volume_percentile=0.95,
            time_since_last_alert=0.01,
        )

        assert action in [2, 3, 4, 5], f"Attack should trigger containment, got action {action}"

    def test_pipeline_latency_acceptable(self):
        """Full pipeline hot-path should complete under 50ms per event."""
        # Build baseline
        for _ in range(100):
            features = np.random.uniform(0.2, 0.4, 16).astype(np.float32)
            spikes = self.encoder.encode(features)
            self.ensemble.process_event(spikes)

        # Measure latency
        latencies = []
        for _ in range(50):
            features = np.random.uniform(0, 1, 16).astype(np.float32)

            t0 = time.perf_counter_ns()
            spikes = self.encoder.encode(features)
            snn_result = self.ensemble.process_event(spikes)
            latency_ms = (time.perf_counter_ns() - t0) / 1e6

            latencies.append(latency_ms)

        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)

        assert p50 < 50.0, f"p50 latency {p50:.2f}ms exceeds 50ms threshold"

    def test_consolidation_preserves_functionality(self):
        """After consolidation, the ensemble should still function correctly."""
        # Train for a while
        for _ in range(50):
            features = np.random.uniform(0.2, 0.4, 16).astype(np.float32)
            spikes = self.encoder.encode(features)
            self.ensemble.process_event(spikes)

        # Consolidate
        consolidator = PrototypeConsolidator(self.ensemble)
        result = consolidator.consolidate()

        assert result["modules_cleaned"] == 3

        # Ensemble should still work after consolidation
        features = np.random.uniform(0, 1, 16).astype(np.float32)
        spikes = self.encoder.encode(features)
        post_result = self.ensemble.process_event(spikes)

        assert "is_anomaly" in post_result
        assert "z_score" in post_result

    def test_spike_encoding_to_snn_compatibility(self):
        """Spike encoder output should be compatible with SNN input dimensions."""
        features = np.random.uniform(0, 1, 16).astype(np.float32)
        spikes = self.encoder.encode(features)

        assert spikes.shape[0] == self.ensemble.n_input, \
            f"Encoder output ({spikes.shape[0]}) != SNN input ({self.ensemble.n_input})"

        # Should process without error
        result = self.ensemble.process_event(spikes)
        assert result is not None


class TestFlowAggregatorIntegration:
    """Test the flow aggregator feature vector conversion."""

    def test_feature_vector_shape(self):
        """Feature vector should have 16 dimensions."""
        aggregator = FlowAggregator()

        mock_flow = {
            "src_ip_hash": 0.5,
            "dst_ip_hash": 0.3,
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,
            "packet_length": 500,
            "inter_packet_interval": 0.01,
            "flow_byte_count": 50000,
            "flow_packet_count": 100,
            "flow_duration": 5.0,
            "tcp_flags": 18,
            "payload_entropy": 4.5,
            "application_fingerprint": 1,
            "time_of_day_bucket": 14,
            "device_type_hint": 1,
            "subnet_id": 168,
        }

        vec = aggregator.to_feature_vector(mock_flow)
        assert vec.shape == (16,), f"Expected (16,), got {vec.shape}"
        assert vec.dtype == np.float32

    def test_feature_vector_normalised(self):
        """All feature values should be in [0, 1] range."""
        aggregator = FlowAggregator()

        mock_flow = {
            "src_ip_hash": 0.5,
            "dst_ip_hash": 0.3,
            "src_port": 65535,
            "dst_port": 443,
            "protocol": 17,
            "packet_length": 1500,
            "inter_packet_interval": 0.5,
            "flow_byte_count": 1000000,
            "flow_packet_count": 1000,
            "flow_duration": 120,
            "tcp_flags": 255,
            "payload_entropy": 8.0,
            "application_fingerprint": 15,
            "time_of_day_bucket": 23,
            "device_type_hint": 3,
            "subnet_id": 255,
        }

        vec = aggregator.to_feature_vector(mock_flow)
        assert np.all(vec >= 0.0), f"Found negative values: {vec}"
        assert np.all(vec <= 1.0), f"Found values > 1.0: {vec}"

    def test_feature_names_match_vector_length(self):
        """Feature names list should have 16 entries."""
        assert len(FLOW_FEATURE_NAMES) == 16


class TestMultiModuleCoherence:
    """Test that all modules produce coherent results when chained."""

    def test_ensemble_modules_independent(self):
        """Each module in the ensemble should have different weights."""
        ensemble = HebbianEnsemble(M=5)

        # Verify initial weights differ (random initialisation)
        for i in range(4):
            assert not np.array_equal(
                ensemble.modules[i].W,
                ensemble.modules[i + 1].W,
            ), f"Modules {i} and {i+1} should have different initial weights"

    def test_ensemble_adapts_over_time(self):
        """Ensemble weights should change after processing events."""
        ensemble = HebbianEnsemble(M=2)

        W_before = [m.W.copy() for m in ensemble.modules]

        for _ in range(50):
            x = np.random.randint(0, 2, 512).astype(np.float32)
            ensemble.process_event(x)

        for i, module in enumerate(ensemble.modules):
            assert not np.array_equal(W_before[i], module.W), \
                f"Module {i} weights should change after learning"
