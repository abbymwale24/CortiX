import os
import pickle
import argparse
import logging
import numpy as np

from typing import TYPE_CHECKING

try:
    from stable_baselines3 import DQN
except ImportError:
    DQN = None

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback
else:
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        class BaseCallback:
            def __init__(self, verbose: int = 0):
                pass

from cortix.config import config
from cortix.containment.environment import CortixEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.containment.train_agent")


class RewardLoggingCallback(BaseCallback):
    """Callback for logging reward progress during DQN training."""
    def __init__(self, log_freq: int = 25000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.reward_history = []

    def _on_step(self) -> bool:
        # Check if episode finished
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if self.n_calls % self.log_freq == 0:
            rewards = self.logger.name_to_value.get("rollout/ep_rew_mean")
            if rewards is not None:
                logger.info("[Step %7d / %d] Mean Episode Reward: %+.2f", self.n_calls, self.model.num_timesteps, rewards)
                self.reward_history.append((self.n_calls, rewards))
            elif self.episode_rewards:
                recent_mean = np.mean(self.episode_rewards[-100:])
                logger.info("[Step %7d] Recent Mean Reward (last 100 eps): %+.2f", self.n_calls, recent_mean)
                self.reward_history.append((self.n_calls, recent_mean))
        return True


def train_agent(
    events_dataset: list[dict] | None = None,
    timesteps: int | None = None,
    real_dataset_path: str = "data/real_containment_events_kddtrain.pkl",
):
    timesteps = timesteps or config.RL_TOTAL_TIMESTEPS
    
    if DQN is None:
        logger.error("stable-baselines3 not installed. Cannot train DQN agent.")
        return None

    if events_dataset is None and os.path.exists(real_dataset_path):
        logger.info("Loading real containment events from %s...", real_dataset_path)
        with open(real_dataset_path, "rb") as f:
            events_dataset = pickle.load(f)

    if events_dataset is not None:
        logger.info("Initializing Gym environment with REAL events dataset (%d samples)...", len(events_dataset))
        env = CortixEnv(events_dataset=events_dataset)
    else:
        logger.warning("No real events dataset provided or found. Falling back to synthetic events.")
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
        verbose=0,
    )

    callback = RewardLoggingCallback(log_freq=25000)
    logger.info("Starting RL training loop for %d timesteps...", timesteps)
    model.learn(total_timesteps=timesteps, callback=callback)

    logger.info("Training finished. Saving agent weights to: %s", config.AGENT_PATH)
    os.makedirs(os.path.dirname(config.AGENT_PATH), exist_ok=True)
    model.save(config.AGENT_PATH)
    logger.info("DQN agent weights saved.")
    
    return callback.reward_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CortiX DQN Agent")
    parser.add_argument(
        "--timesteps", type=int, default=500000, help="Number of training steps"
    )
    args = parser.parse_args()
    train_agent(timesteps=args.timesteps)
