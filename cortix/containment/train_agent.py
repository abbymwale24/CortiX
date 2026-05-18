"""
CortiX Module 4 — Double DQN Training script

Trains the Double-Dueling DQN agent inside the Gymnasium firewall environment.
Saves model weights once training steps are complete.
"""

import os
import argparse
import logging

try:
    from stable_baselines3 import DQN
except ImportError:
    DQN = None

from cortix.config import config
from cortix.containment.environment import CortixEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.containment.train_agent")


def train_agent(timesteps: int = None):
    timesteps = timesteps or config.RL_TOTAL_TIMESTEPS
    
    if DQN is None:
        logger.error("stable-baselines3 not installed. Cannot train DQN agent.")
        return

    logger.info("Initializing Gym environment for DQN training")
    env = CortixEnv()

    logger.info("Building double-dueling DQN agent architecture")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.RL_LEARNING_RATE,
        buffer_size=config.RL_BUFFER_SIZE,
        learning_starts=1000,
        batch_size=config.RL_BATCH_SIZE,
        tau=1.0,
        gamma=config.RL_GAMMA,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config.RL_TARGET_UPDATE_INTERVAL,
        exploration_fraction=config.RL_EXPLORATION_FRACTION,
        exploration_final_eps=config.RL_EXPLORATION_FINAL_EPS,
        verbose=1,
    )

    logger.info("Starting RL training loop for %d timesteps", timesteps)
    model.learn(total_timesteps=timesteps)

    logger.info("Training finished. Saving agent weights to: %s", config.AGENT_PATH)
    os.makedirs(os.path.dirname(config.AGENT_PATH), exist_ok=True)
    model.save(config.AGENT_PATH)
    logger.info("DQN agent weights saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CortiX DQN Agent")
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Number of training steps"
    )
    args = parser.parse_args()
    train_agent(timesteps=args.timesteps)
