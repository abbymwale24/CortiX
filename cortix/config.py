"""
CortiX — Centralised Configuration

All tunable parameters, API keys, thresholds, and system paths.
Use environment variables to override sensitive values in production.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CortixConfig:
    """Master configuration for all CortiX modules."""

    # ──────────────────────────────────────────────
    # Network Capture
    # ──────────────────────────────────────────────
    CAPTURE_INTERFACE: str = os.getenv("CORTIX_INTERFACE", "eth0")
    BPF_FILTER: str = os.getenv("CORTIX_BPF_FILTER", "")
    FLOW_WINDOW_SECONDS: float = 1.0

    # ──────────────────────────────────────────────
    # Spike Encoding / Thalamic Gating
    # ──────────────────────────────────────────────
    NUM_INPUT_NEURONS: int = 512
    RECEPTIVE_FIELD_CENTERS: int = 20  # Gaussian receptive fields per feature
    SPIKE_RATE_WINDOW_MS: float = 50.0
    THALAMIC_GATE_ENABLED: bool = True
    THALAMIC_ZERO_SUPPRESSION_EPS: float = 1e-4

    # ──────────────────────────────────────────────
    # SNN / Hebbian Engine & Neuromodulation
    # ──────────────────────────────────────────────
    HEBBIAN_MODULES: int = 5
    NEURONS_PER_MODULE: int = 512
    HIDDEN_NEURONS: int = 256
    KWTA_SPARSITY: float = 0.10  # Fraction of hidden neurons that win
    STDP_A_PLUS: float = 0.03
    STDP_A_MINUS: float = 0.035
    STDP_TAU_PLUS: float = 20e-3  # 20 ms
    STDP_TAU_MINUS: float = 20e-3
    HEBBIAN_LR: float = 0.001
    METAPLASTICITY_ALPHA: float = 10.0
    NEUROMODULATION_ENABLED: bool = True
    NEUROMODULATION_BASELINE: float = 0.1
    ANOMALY_Z_THRESHOLD: float = 3.5
    SLIDING_WINDOW_SIZE: int = 1000
    # ── Reproducibility ──
    # Default seed for HebbianEnsemble weight initialisation.
    # - Read from CORTIX_SEED env var if set, so judges can watch you
    #   change it live at the terminal without touching code.
    # - None means "no seed" -> falls back to nondeterministic init,
    #   matching the original (buggy) behaviour, for comparison purposes.
    RANDOM_SEED: int | None = (
        int(os.environ["CORTIX_SEED"]) if os.environ.get("CORTIX_SEED") else 42
    )
    CONSOLIDATION_INTERVAL_SEC: int = 600  # 10 minutes

    # ──────────────────────────────────────────────
    # LSTM-CNN Classifier
    # ──────────────────────────────────────────────
    MODEL_PATH: str = "models/lstm_cnn_cicids2017.pt"
    MODEL_PATH_CICIDS2017: str = "models/lstm_cnn_cicids2017.pt"
    MODEL_PATH_NSLKDD: str = "models/lstm_cnn_nslkdd.pt"
    CLASSIFIER_NUM_CLASSES: int = 9
    CLASSIFIER_SEQ_LEN: int = 10
    CLASSIFIER_NUM_FEATURES: int = 40
    CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.85
    CLASSIFIER_BATCH_SIZE: int = 256
    CLASSIFIER_EPOCHS: int = 50
    CLASSIFIER_LR: float = 1e-3
    CLASSIFIER_WEIGHT_DECAY: float = 1e-4
    CLASSIFIER_EARLY_STOPPING_PATIENCE: int = 10

    # ──────────────────────────────────────────────
    # Deep RL Containment Agent
    # ──────────────────────────────────────────────
    AGENT_PATH: str = "models/dqn_containment.zip"
    RL_LEARNING_RATE: float = 1e-4
    RL_BUFFER_SIZE: int = 100_000
    RL_BATCH_SIZE: int = 64
    RL_GAMMA: float = 0.99
    RL_TARGET_UPDATE_INTERVAL: int = 1000
    RL_EXPLORATION_FRACTION: float = 0.2
    RL_EXPLORATION_FINAL_EPS: float = 0.02
    RL_TOTAL_TIMESTEPS: int = 500_000
    BLOCK_DURATION_SECONDS: int = 60
    NUM_ACTIONS: int = 6
    STATE_DIM: int = 20

    # Reward shaping
    REWARD_CORRECT_BLOCK: float = 10.0
    REWARD_RATE_LIMIT: float = 5.0
    REWARD_CORRECT_ALLOW: float = 1.0
    PENALTY_FALSE_POSITIVE: float = -20.0
    PENALTY_MISSED_ATTACK: float = -5.0
    PENALTY_UNNECESSARY_QUARANTINE: float = -2.0
    REWARD_HONEYPOT_CAPTURE: float = 15.0

    # ──────────────────────────────────────────────
    # Ransomware Honeypot
    # ──────────────────────────────────────────────
    HONEYPOT_WATCH_DIRS: list = field(
        default_factory=lambda: ["/home", "/var", "/data"]
    )
    HONEYPOT_RENAME_THRESHOLD: int = 50  # per second
    HONEYPOT_CONTAINER_NAME: str = "cortix-honeypot"
    HONEYPOT_TIMEOUT_SECONDS: int = 60

    # ──────────────────────────────────────────────
    # OSINT Attribution API Keys (free tiers)
    # ──────────────────────────────────────────────
    ABUSEIPDB_API_KEY: str = os.getenv("ABUSEIPDB_API_KEY", "")
    VIRUSTOTAL_API_KEY: str = os.getenv("VIRUSTOTAL_API_KEY", "")
    SHODAN_API_KEY: str = os.getenv("SHODAN_API_KEY", "")
    MAXMIND_DB_PATH: str = os.getenv(
        "MAXMIND_DB_PATH", "data/GeoLite2-City.mmdb"
    )

    # ──────────────────────────────────────────────
    # Alerting
    # ──────────────────────────────────────────────
    ADMIN_EMAIL: str = os.getenv("CORTIX_ADMIN_EMAIL", "admin@your-org.com")
    SMTP_SERVER: str = os.getenv("CORTIX_SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("CORTIX_SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("CORTIX_SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("CORTIX_SMTP_PASSWORD", "")

    # ──────────────────────────────────────────────
    # Database & Message Bus
    # ──────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "CORTIX_DATABASE_URL", "sqlite:///cortix.db"
    )
    REDIS_URL: str = os.getenv("CORTIX_REDIS_URL", "redis://localhost:6379/0")

    # ──────────────────────────────────────────────
    # Dashboard / API
    # ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DASHBOARD_PORT: int = 3000
    CORS_ORIGINS: list = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # ──────────────────────────────────────────────
    # Logging & Profiling
    # ──────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("CORTIX_LOG_LEVEL", "INFO")
    LATENCY_LOG_INTERVAL: int = 100  # Log p50/p99 every N events


# Singleton instance
config = CortixConfig()
