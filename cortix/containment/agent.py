"""
CortiX Module 4 — DQN Containment Agent Wrapper

Wraps stable-baselines3 double-dueling DQN model for production 
inference action selection and observation loading.
"""

import os
import logging
import numpy as np

try:
    from stable_baselines3 import DQN
except ImportError:
    DQN = None

from cortix.config import config

logger = logging.getLogger("cortix.containment.agent")


class ContainmentAgent:
    """
    Production interface for selecting the optimal containment action
    given the current security telemetry.
    """

    def __init__(self, agent_path: str = None):
        self.agent_path = agent_path or config.AGENT_PATH
        self.agent = None

        self._load_agent()

    def _load_agent(self):
        """Load pretrained stable-baselines3 DQN agent."""
        if DQN is None:
            logger.warning("stable-baselines3 not installed. DQN containment disabled.")
            return

        if os.path.exists(self.agent_path):
            try:
                self.agent = DQN.load(self.agent_path)
                logger.info("Containment DQN agent loaded from: %s", self.agent_path)
            except Exception as exc:
                logger.error("Failed to load DQN agent weights: %s", exc)
        else:
            logger.warning("No DQN model weights found at %s. Falling back to rule-based fallback.", self.agent_path)

    def select_action(
        self,
        z_score: float,
        confidence: float,
        predicted_class: str,
        rolling_fpr: float,
        reputation: float,
        volume_percentile: float,
        time_since_last_alert: float,
        current_action: int = 0,
    ) -> int:
        """
        Determine optimal containment action using the DQN agent (or deterministic rules).
        
        Returns:
            action_id: integer in range 0..5
        """
        # Construct state vector matching Gymnasium definition (20 dimensions)
        state = np.zeros(config.STATE_DIM, dtype=np.float32)
        state[0] = min(z_score / 20.0, 1.0)
        state[1] = confidence

        # One-hot predicted class
        attack_classes = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration", "Botnet", "ZeroDay"]
        class_idx = attack_classes.index(predicted_class) if predicted_class in attack_classes else 0
        state[2 + class_idx] = 1.0

        state[11] = rolling_fpr
        state[12] = reputation
        state[13] = volume_percentile
        state[14] = time_since_last_alert
        
        # One-hot current containment action
        if 0 <= current_action < 6:
            state[15 + current_action] = 1.0

        # DQN inference
        if self.agent is not None:
            try:
                action, _ = self.agent.predict(state, deterministic=True)
                return int(action)
            except Exception as exc:
                logger.error("DQN action prediction failed: %s. Falling back to heuristic.", exc)

        # Heuristic Rule-Based Fallback if model is not trained yet
        if predicted_class == "BENIGN":
            return 0  # ALLOW
            
        if predicted_class in ["DDoS", "DoS"]:
            return 4  # HARD_BLOCK if critical flood
            
        if predicted_class == "Infiltration":
            return 5  # HONEYPOT_REDIRECT for high risk intrusions
            
        if predicted_class == "PortScan":
            return 2  # TEMP_BLOCK
            
        return 1  # RATE_LIMIT (default threshold)
