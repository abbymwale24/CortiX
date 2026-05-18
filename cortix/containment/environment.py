"""
CortiX Module 4 — Gymnasium Containment Environment

Simulates firewall decisions. The RL agent receives observations (anomaly z-score, 
classifier output, false positive rate, reputation) and selects from actions 
(Allow, Rate-Limit, Temp-Block, Quarantine, Hard-Block, Redirect to Honeypot).
"""

import logging
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cortix.config import config

logger = logging.getLogger("cortix.containment.environment")


class CortixEnv(gym.Env):
    """
    Gymnasium environment simulating network attack events for Double DQN containment training.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, events_dataset: list[dict] = None):
        super().__init__()
        # Load dataset of historical scenarios if provided; else generate synthetic
        self.events = events_dataset or self._generate_synthetic_events()
        
        # Action space: 6 discrete containment levels
        self.action_space = spaces.Discrete(config.NUM_ACTIONS)

        # Observation space: 20 dimensions
        # - z_score (normalised) [0]
        # - classifier confidence [1]
        # - classifier predicted class (one-hot, 9 dims) [2..10]
        # - recent false_positive_rate [11]
        # - source IP reputation [12]
        # - traffic volume percentile [13]
        # - time_since_last_alert [14]
        # - current_containment_action (one-hot, 6 dims) [15..20]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.STATE_DIM,),
            dtype=np.float32,
        )

        self.current_idx = 0
        self.rolling_fp_rate = 0.02
        self.total_alerts = 0
        self.fp_alerts = 0

    def _generate_synthetic_events(self, num_events: int = 5000) -> list[dict]:
        """Generate diverse benign and attack event dictionaries for training offline."""
        events = []
        for _ in range(num_events):
            is_attack = random.random() < 0.25  # 25% attacks
            
            if is_attack:
                attack_classes = ["DoS", "DDoS", "PortScan", "BruteForce", "Infiltration", "Botnet"]
                pred_class = random.choice(attack_classes)
                confidence = random.uniform(0.70, 0.99)
                z_score = random.uniform(3.0, 15.0)
                reputation = random.uniform(0.50, 1.0)
            else:
                pred_class = "BENIGN"
                confidence = random.uniform(0.90, 1.0)
                z_score = random.uniform(0.0, 2.5)
                reputation = random.uniform(0.0, 0.20)

            events.append({
                "is_attack": is_attack,
                "class": pred_class,
                "confidence": confidence,
                "z_score": z_score,
                "reputation": reputation,
                "volume": random.uniform(0.1, 0.9),
                "time_since_last": random.uniform(0.01, 1.0),
            })
        return events

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.rolling_fp_rate = 0.02
        self.total_alerts = 0
        self.fp_alerts = 0
        
        state = self._get_observation()
        info = {}
        return state, info

    def _get_observation(self) -> np.ndarray:
        event = self.events[self.current_idx]
        
        obs = np.zeros(config.STATE_DIM, dtype=np.float32)
        obs[0] = min(event["z_score"] / 20.0, 1.0)
        obs[1] = event["confidence"]

        # One-hot predicted class
        attack_classes = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration", "Botnet", "ZeroDay"]
        class_idx = attack_classes.index(event["class"]) if event["class"] in attack_classes else 0
        obs[2 + class_idx] = 1.0

        obs[11] = self.rolling_fp_rate
        obs[12] = event["reputation"]
        obs[13] = event["volume"]
        obs[14] = event["time_since_last"]

        # Current state containment action: default ALLOW (0) during new state
        obs[15] = 1.0  # Default ALLOW index 15
        
        return obs

    def step(self, action: int):
        event = self.events[self.current_idx]
        is_attack = event["is_attack"]
        class_name = event["class"]

        reward = 0.0
        info = {"status": "SUCCESS"}

        # Reward formulation based on selected action VS real ground-truth threat status
        if is_attack:
            if class_name == "Infiltration" and action == 5:
                # Redirect ransomware / high risk threat to Honeypot
                reward += config.REWARD_HONEYPOT_CAPTURE
            elif action in [2, 3, 4]:
                # Strong containment
                reward += config.REWARD_CORRECT_BLOCK
            elif action == 1:
                # Throttled/Rate-limited attack
                reward += config.REWARD_RATE_LIMIT
            else:
                # Let attack pass completely (Allow)
                reward += config.PENALTY_MISSED_ATTACK
        else:
            # Benign event
            if action == 0:
                reward += config.REWARD_CORRECT_ALLOW
            elif action in [2, 3, 4]:
                # False positive block
                reward += config.PENALTY_FALSE_POSITIVE
                self.fp_alerts += 1
            else:
                reward += config.PENALTY_UNNECESSARY_QUARANTINE

        # Update FP stats
        self.total_alerts += 1
        if self.total_alerts > 0:
            self.rolling_fp_rate = self.fp_alerts / self.total_alerts

        self.current_idx += 1
        terminated = self.current_idx >= len(self.events) - 1
        truncated = False

        next_state = self._get_observation() if not terminated else np.zeros((config.STATE_DIM,), dtype=np.float32)

        return next_state, reward, terminated, truncated, info

    def render(self):
        pass
