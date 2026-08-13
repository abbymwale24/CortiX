"""
CortiX — Unit Tests for Containment RL Environment and Classifier

Tests the Gymnasium environment, reward shaping, DQN agent heuristic fallback,
and the classifier inference pipeline.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for classifier tests")

from cortix.containment.environment import CortixEnv
from cortix.containment.agent import ContainmentAgent
from cortix.classifier.inference import ClassifierInference
from cortix.classifier.dataset import clean_dataframe, SequenceFlowDataset, SELECTED_FEATURES
from cortix.config import config

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ──────────────────────────────────────────────
# Gymnasium Environment
# ──────────────────────────────────────────────

class TestCortixEnv:
    """Test the Gymnasium containment environment."""

    def test_env_creation(self):
        """Environment should initialise without errors."""
        env = CortixEnv()
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == config.NUM_ACTIONS
        assert env.observation_space.shape == (config.STATE_DIM,)

    def test_reset_returns_valid_state(self):
        """Reset should return a valid observation and info dict."""
        env = CortixEnv()
        state, info = env.reset()

        assert state.shape == (config.STATE_DIM,)
        assert isinstance(info, dict)
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

    def test_step_returns_correct_format(self):
        """Step should return (state, reward, terminated, truncated, info)."""
        env = CortixEnv()
        state, _ = env.reset()

        next_state, reward, terminated, truncated, info = env.step(0)

        assert next_state.shape == (config.STATE_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_correct_block_attack_positive_reward(self):
        """Blocking an attack should yield positive reward."""
        events = [
            {
                "is_attack": True, "class": "DoS", "confidence": 0.95,
                "z_score": 8.0, "reputation": 0.8, "volume": 0.7,
                "time_since_last": 0.1,
            }
        ]
        env = CortixEnv(events_dataset=events * 10)
        env.reset()

        # Action 4 = HARD_BLOCK for a DoS attack
        _, reward, _, _, _ = env.step(4)
        assert reward > 0, f"Blocking attack should be positive reward, got {reward}"

    def test_false_positive_block_negative_reward(self):
        """Blocking benign traffic should yield negative reward."""
        events = [
            {
                "is_attack": False, "class": "BENIGN", "confidence": 0.99,
                "z_score": 0.5, "reputation": 0.1, "volume": 0.3,
                "time_since_last": 0.5,
            }
        ]
        env = CortixEnv(events_dataset=events * 10)
        env.reset()

        # Action 4 = HARD_BLOCK on benign → false positive
        _, reward, _, _, _ = env.step(4)
        assert reward < 0, f"False positive should be negative reward, got {reward}"

    def test_correct_allow_benign_positive_reward(self):
        """Allowing benign traffic should yield positive reward."""
        events = [
            {
                "is_attack": False, "class": "BENIGN", "confidence": 0.99,
                "z_score": 0.3, "reputation": 0.05, "volume": 0.2,
                "time_since_last": 0.8,
            }
        ]
        env = CortixEnv(events_dataset=events * 10)
        env.reset()

        # Action 0 = ALLOW for benign traffic
        _, reward, _, _, _ = env.step(0)
        assert reward > 0, f"Correct allow should be positive reward, got {reward}"

    def test_missed_attack_negative_reward(self):
        """Allowing an attack should yield negative reward."""
        events = [
            {
                "is_attack": True, "class": "DDoS", "confidence": 0.92,
                "z_score": 10.0, "reputation": 0.9, "volume": 0.9,
                "time_since_last": 0.05,
            }
        ]
        env = CortixEnv(events_dataset=events * 10)
        env.reset()

        # Action 0 = ALLOW → missed attack
        _, reward, _, _, _ = env.step(0)
        assert reward < 0, f"Missing attack should be negative reward, got {reward}"

    def test_honeypot_redirect_infiltration(self):
        """Redirecting infiltration to honeypot should yield high reward."""
        events = [
            {
                "is_attack": True, "class": "Infiltration", "confidence": 0.88,
                "z_score": 5.0, "reputation": 0.7, "volume": 0.4,
                "time_since_last": 0.2,
            }
        ]
        env = CortixEnv(events_dataset=events * 10)
        env.reset()

        # Action 5 = HONEYPOT_REDIRECT for Infiltration
        _, reward, _, _, _ = env.step(5)
        assert reward == config.REWARD_HONEYPOT_CAPTURE

    def test_episode_terminates(self):
        """Episode should terminate when events are exhausted."""
        events = [
            {
                "is_attack": False, "class": "BENIGN", "confidence": 0.99,
                "z_score": 0.1, "reputation": 0.05, "volume": 0.1,
                "time_since_last": 1.0,
            }
        ]
        env = CortixEnv(events_dataset=events * 3)
        env.reset()

        for i in range(3):
            _, _, terminated, _, _ = env.step(0)

        # Should terminate after exhausting events
        assert terminated is True

    def test_fp_rate_tracking(self):
        """False positive rate should be tracked over episodes."""
        events = [
            {"is_attack": False, "class": "BENIGN", "confidence": 0.99,
             "z_score": 0.1, "reputation": 0.05, "volume": 0.1, "time_since_last": 1.0},
        ] * 10
        env = CortixEnv(events_dataset=events)
        env.reset()

        # Block all benign → all false positives
        for _ in range(5):
            env.step(4)  # HARD_BLOCK

        assert env.fp_alerts == 5
        assert env.rolling_fp_rate == pytest.approx(1.0)


# ──────────────────────────────────────────────
# Containment Agent (Heuristic Fallback)
# ──────────────────────────────────────────────

class TestContainmentAgent:
    """Test the DQN agent heuristic fallback rules."""

    def test_benign_returns_allow(self):
        """Benign traffic should get ALLOW action."""
        agent = ContainmentAgent(agent_path="/nonexistent/path")

        action = agent.select_action(
            z_score=0.5, confidence=0.99, predicted_class="BENIGN",
            rolling_fpr=0.01, reputation=0.1, volume_percentile=0.2,
            time_since_last_alert=1.0,
        )
        assert action == 0, "BENIGN should be ALLOW (0)"

    def test_ddos_returns_hard_block(self):
        """DDoS should trigger HARD_BLOCK."""
        agent = ContainmentAgent(agent_path="/nonexistent/path")

        action = agent.select_action(
            z_score=12.0, confidence=0.95, predicted_class="DDoS",
            rolling_fpr=0.01, reputation=0.9, volume_percentile=0.95,
            time_since_last_alert=0.01,
        )
        assert action == 4, "DDoS should be HARD_BLOCK (4)"

    def test_dos_returns_hard_block(self):
        """DoS should trigger HARD_BLOCK."""
        agent = ContainmentAgent(agent_path="/nonexistent/path")

        action = agent.select_action(
            z_score=10.0, confidence=0.92, predicted_class="DoS",
            rolling_fpr=0.01, reputation=0.8, volume_percentile=0.85,
            time_since_last_alert=0.05,
        )
        assert action == 4

    def test_portscan_returns_temp_block(self):
        """PortScan should trigger TEMP_BLOCK."""
        agent = ContainmentAgent(agent_path="/nonexistent/path")

        action = agent.select_action(
            z_score=5.0, confidence=0.87, predicted_class="PortScan",
            rolling_fpr=0.02, reputation=0.5, volume_percentile=0.6,
            time_since_last_alert=0.2,
        )
        assert action == 2, "PortScan should be TEMP_BLOCK (2)"

    def test_infiltration_returns_honeypot(self):
        """Infiltration should trigger HONEYPOT_REDIRECT."""
        agent = ContainmentAgent(agent_path="/nonexistent/path")

        action = agent.select_action(
            z_score=6.0, confidence=0.85, predicted_class="Infiltration",
            rolling_fpr=0.01, reputation=0.7, volume_percentile=0.4,
            time_since_last_alert=0.3,
        )
        assert action == 5, "Infiltration should be HONEYPOT_REDIRECT (5)"

    def test_unknown_class_returns_rate_limit(self):
        """Unknown attack classes should default to RATE_LIMIT."""
        agent = ContainmentAgent(agent_path="/nonexistent/path")

        action = agent.select_action(
            z_score=4.0, confidence=0.80, predicted_class="WebAttack",
            rolling_fpr=0.01, reputation=0.5, volume_percentile=0.5,
            time_since_last_alert=0.5,
        )
        assert action == 1, "Unknown attack should default to RATE_LIMIT (1)"


# ──────────────────────────────────────────────
# Classifier Inference
# ──────────────────────────────────────────────

class TestClassifierInference:
    """Test the LSTM-CNN inference wrapper."""

    def test_warmup_returns_benign(self):
        """Before sequence buffer is full, should return BENIGN."""
        classifier = ClassifierInference(model_path="/nonexistent/model.pt", seq_len=10)

        flow = np.random.uniform(0, 1, 40).astype(np.float32)
        result = classifier.predict_flow("192.168.1.1", flow)

        assert result["class"] == "BENIGN"
        assert result["status"] == "warming_up"
        assert result["is_threat"] is False

    def test_buffer_accumulation(self):
        """Buffer should accumulate flows per IP."""
        classifier = ClassifierInference(model_path="/nonexistent/model.pt", seq_len=10)

        for i in range(5):
            flow = np.random.uniform(0, 1, 40).astype(np.float32)
            classifier.predict_flow("10.0.0.1", flow)

        assert len(classifier._flow_buffers["10.0.0.1"]) == 5

    def test_different_ips_separate_buffers(self):
        """Different IPs should have independent buffers."""
        classifier = ClassifierInference(model_path="/nonexistent/model.pt", seq_len=10)

        for i in range(3):
            classifier.predict_flow("10.0.0.1", np.zeros(40, dtype=np.float32))
        for i in range(5):
            classifier.predict_flow("10.0.0.2", np.ones(40, dtype=np.float32))

        assert len(classifier._flow_buffers["10.0.0.1"]) == 3
        assert len(classifier._flow_buffers["10.0.0.2"]) == 5

    def test_clear_buffer(self):
        """Clearing buffer should remove IP history."""
        classifier = ClassifierInference(model_path="/nonexistent/model.pt", seq_len=10)

        for i in range(5):
            classifier.predict_flow("10.0.0.1", np.zeros(40, dtype=np.float32))

        classifier.clear_buffer("10.0.0.1")
        assert len(classifier._flow_buffers["10.0.0.1"]) == 0

    def test_mock_fallback_after_full_buffer(self):
        """Without a model, full buffer should still return BENIGN (mock)."""
        classifier = ClassifierInference(model_path="/nonexistent/model.pt")

        result: dict = {}
        # Fill buffer to seq_len
        for i in range(config.CLASSIFIER_SEQ_LEN):
            flow = np.random.uniform(0, 1, 40).astype(np.float32)
            result = classifier.predict_flow("10.0.0.1", flow)

        assert result.get("class") == "BENIGN"
        assert result.get("confidence") == 1.0
        assert result.get("status") == "active"


# ──────────────────────────────────────────────
# Dataset Utilities
# ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDatasetUtils:
    """Test dataset preprocessing utilities."""

    def test_clean_dataframe_handles_inf(self):
        """Cleaning should replace inf values with median."""
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0, -np.inf, 5.0], "b": [1, 2, 3, 4, 5]})
        cleaned = clean_dataframe(df)

        assert not np.any(np.isinf(cleaned["a"].values))
        assert not np.any(np.isnan(cleaned["a"].values))

    def test_clean_dataframe_handles_nan(self):
        """Cleaning should fill NaN with median."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        cleaned = clean_dataframe(df)

        assert not np.any(np.isnan(cleaned["x"].values))

    def test_clean_dataframe_strips_columns(self):
        """Column names should be stripped of whitespace."""
        df = pd.DataFrame({" col_a ": [1], " col_b ": [2]})
        cleaned = clean_dataframe(df)

        assert "col_a" in cleaned.columns
        assert "col_b" in cleaned.columns

    def test_sequence_dataset_length(self):
        """SequenceFlowDataset should have correct length."""
        X = np.random.uniform(0, 1, (100, 40)).astype(np.float32)
        y = np.random.randint(0, 9, 100)
        seq_len = 10

        dataset = SequenceFlowDataset(X, y, seq_len)

        # Valid sequences: 100 - 10 + 1 = 91
        assert len(dataset) == 91

    def test_sequence_dataset_shapes(self):
        """Each item should have correct tensor shapes."""
        X = np.random.uniform(0, 1, (50, 40)).astype(np.float32)
        y = np.random.randint(0, 9, 50)
        seq_len = 10

        dataset = SequenceFlowDataset(X, y, seq_len)
        x_seq, y_label = dataset[0]

        assert x_seq.shape == (10, 40)
        assert y_label.shape == ()

    def test_selected_features_count(self):
        """SELECTED_FEATURES should have exactly 40 features."""
        assert len(SELECTED_FEATURES) == 40
