"""
CortiX — Redis Pub/Sub Message Bus

All inter-module communication flows through Redis channels.
Provides publish/subscribe helpers and channel constants.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import redis

from cortix.config import config

logger = logging.getLogger("cortix.redis_bus")

# ──────────────────────────────────────────────
# Channel Definitions
# ──────────────────────────────────────────────
CHANNEL_THREAT_DETECTED = "cortix:threat_detected"
CHANNEL_CLASSIFICATION = "cortix:classification"
CHANNEL_CONTAINMENT_ACTION = "cortix:containment_action"
CHANNEL_ATTACKER_PROFILE = "cortix:attacker_profile"
CHANNEL_HONEYPOT_EVENT = "cortix:honeypot_event"
CHANNEL_SYSTEM_METRICS = "cortix:system_metrics"
CHANNEL_LIVE_EVENTS = "cortix:live_events"  # Dashboard WebSocket feed


class RedisBus:
    """
    Wrapper around Redis pub/sub for CortiX inter-module communication.

    Usage:
        bus = RedisBus()
        bus.publish(CHANNEL_THREAT_DETECTED, {"src_ip": "1.2.3.4", "score": 4.2})
        bus.subscribe(CHANNEL_THREAT_DETECTED, handler_fn)
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._url = redis_url or config.REDIS_URL
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._handlers: Dict[str, list] = {}

    # ── Connection ──────────────────────────────

    def connect(self) -> "RedisBus":
        """Establish Redis connection."""
        try:
            self._client = redis.Redis.from_url(
                self._url, decode_responses=True
            )
            self._client.ping()
            self._pubsub = self._client.pubsub()
            logger.info("Redis bus connected: %s", self._url)
        except redis.ConnectionError:
            logger.warning(
                "Redis unavailable at %s — running in offline mode",
                self._url,
            )
            self._client = None
            self._pubsub = None
        return self

    def disconnect(self):
        """Cleanly shut down the bus."""
        self._running = False
        if self._pubsub:
            self._pubsub.close()
        if self._client:
            self._client.close()
        logger.info("Redis bus disconnected")

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    # ── Publish ─────────────────────────────────

    def publish(self, channel: str, data: Dict[str, Any]):
        """
        Publish a JSON message to a Redis channel.

        Falls back to direct handler invocation if Redis is offline.
        """
        message = json.dumps(data, default=str)

        if self._client:
            try:
                self._client.publish(channel, message)
                logger.debug("Published to %s: %s", channel, message[:120])
                return
            except redis.ConnectionError:
                logger.warning("Redis publish failed — falling back to local")

        # Offline fallback: invoke local handlers directly
        for handler in self._handlers.get(channel, []):
            try:
                handler(data)
            except Exception as exc:
                logger.error("Handler error on %s: %s", channel, exc)

    # ── Subscribe ───────────────────────────────

    def subscribe(self, channel: str, handler: Callable[[Dict], None]):
        """
        Register a handler for messages on a channel.

        Args:
            channel: Redis channel name
            handler: Callable that receives the decoded dict payload
        """
        self._handlers.setdefault(channel, []).append(handler)

        if self._pubsub:
            self._pubsub.subscribe(
                **{channel: self._make_redis_handler(handler)}
            )
            logger.info("Subscribed to %s", channel)

    def _make_redis_handler(self, handler: Callable):
        """Wrap user handler to decode Redis message."""

        def wrapper(message):
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    handler(data)
                except json.JSONDecodeError:
                    logger.error("Bad JSON on Redis: %s", message["data"][:100])
                except Exception as exc:
                    logger.error("Handler exception: %s", exc)

        return wrapper

    # ── Listener Thread ─────────────────────────

    def start_listening(self):
        """Start background thread that processes incoming messages."""
        if not self._pubsub:
            logger.warning("No pubsub — listener not started")
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="redis-listener"
        )
        self._listener_thread.start()
        logger.info("Redis listener thread started")

    def _listen_loop(self):
        """Main listener loop — runs in background thread."""
        while self._running:
            try:
                message = self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is None:
                    continue
            except redis.ConnectionError:
                logger.warning("Redis connection lost — retrying in 5s")
                time.sleep(5)
                try:
                    self.connect()
                except Exception:
                    pass
            except Exception as exc:
                logger.error("Listener error: %s", exc)
                time.sleep(1)

    # ── Utility ─────────────────────────────────

    def get_client(self) -> Optional[redis.Redis]:
        """Return raw Redis client for advanced operations."""
        return self._client


# ── Module-level convenience instance ────────────
_bus: Optional[RedisBus] = None


def get_bus() -> RedisBus:
    """Get or create the singleton RedisBus instance."""
    global _bus
    if _bus is None:
        _bus = RedisBus()
        _bus.connect()
    return _bus
