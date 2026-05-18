"""
CortiX — Database Models & Session Management

SQLAlchemy ORM models for threats, attacker profiles, metrics, and containment actions.
Uses SQLite for development, PostgreSQL for production.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

from cortix.config import config

logger = logging.getLogger("cortix.database")


# ──────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class Threat(Base):
    """A detected threat event."""

    __tablename__ = "threats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    src_ip = Column(String(45), index=True)
    dst_ip = Column(String(45))
    src_port = Column(Integer)
    dst_port = Column(Integer)
    protocol = Column(String(10))
    attack_class = Column(String(50), index=True)
    confidence = Column(Float)
    z_score = Column(Float)
    action_taken = Column(String(30))
    resolved = Column(Boolean, default=False)
    attacker_profile_id = Column(
        Integer, ForeignKey("attacker_profiles.id"), nullable=True
    )
    flow_features_json = Column(Text)  # JSON blob of raw flow features

    # Relationships
    attacker_profile = relationship(
        "AttackerProfile", back_populates="threats"
    )
    containment_actions = relationship(
        "ContainmentAction", back_populates="threat"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "attack_class": self.attack_class,
            "confidence": self.confidence,
            "z_score": self.z_score,
            "action_taken": self.action_taken,
            "resolved": self.resolved,
            "attacker_profile_id": self.attacker_profile_id,
            "flow_features": (
                json.loads(self.flow_features_json)
                if self.flow_features_json
                else None
            ),
        }


class AttackerProfile(Base):
    """Passive OSINT-derived attacker profile."""

    __tablename__ = "attacker_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ip = Column(String(45), unique=True, index=True)
    country = Column(String(5))
    city = Column(String(100))
    lat = Column(Float)
    lon = Column(Float)
    isp = Column(String(200))
    asn = Column(String(20))
    hostname = Column(String(255))
    abuse_score = Column(Integer)
    known_malicious = Column(Boolean, default=False)
    attack_categories = Column(Text)  # JSON array
    vt_malicious = Column(Integer, default=0)
    shodan_data_json = Column(Text)
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    threat_level = Column(String(20), default="UNKNOWN")  # LOW, MEDIUM, HIGH, CRITICAL

    # Relationships
    threats = relationship("Threat", back_populates="attacker_profile")

    def to_dict(self):
        return {
            "id": self.id,
            "ip": self.ip,
            "country": self.country,
            "city": self.city,
            "lat": self.lat,
            "lon": self.lon,
            "isp": self.isp,
            "asn": self.asn,
            "hostname": self.hostname,
            "abuse_score": self.abuse_score,
            "known_malicious": self.known_malicious,
            "attack_categories": (
                json.loads(self.attack_categories)
                if self.attack_categories
                else []
            ),
            "vt_malicious": self.vt_malicious,
            "shodan_data": (
                json.loads(self.shodan_data_json)
                if self.shodan_data_json
                else None
            ),
            "first_seen": (
                self.first_seen.isoformat() if self.first_seen else None
            ),
            "last_seen": (
                self.last_seen.isoformat() if self.last_seen else None
            ),
            "threat_level": self.threat_level,
        }


class SystemMetric(Base):
    """Periodic system performance metrics."""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    event_count = Column(Integer, default=0)
    tp_count = Column(Integer, default=0)
    fp_count = Column(Integer, default=0)
    fn_count = Column(Integer, default=0)
    latency_p50_ms = Column(Float)
    latency_p99_ms = Column(Float)
    throughput_pps = Column(Float)  # packets per second

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_count": self.event_count,
            "tp_count": self.tp_count,
            "fp_count": self.fp_count,
            "fn_count": self.fn_count,
            "fpr": (
                self.fp_count / max(self.fp_count + (self.event_count - self.tp_count - self.fp_count - self.fn_count), 1)
                if self.event_count > 0
                else 0.0
            ),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_pps": self.throughput_pps,
        }


class ContainmentAction(Base):
    """Log of containment actions taken by the RL agent or admin."""

    __tablename__ = "containment_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    src_ip = Column(String(45), index=True)
    action = Column(String(30))  # ALLOW, RATE_LIMIT, TEMP_BLOCK, QUARANTINE, HARD_BLOCK, HONEYPOT_REDIRECT
    triggered_by = Column(String(30))  # RL_AGENT, ADMIN, HONEYPOT
    duration_seconds = Column(Integer)
    resolved = Column(Boolean, default=False)
    threat_id = Column(Integer, ForeignKey("threats.id"), nullable=True)

    # Relationships
    threat = relationship("Threat", back_populates="containment_actions")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "src_ip": self.src_ip,
            "action": self.action,
            "triggered_by": self.triggered_by,
            "duration_seconds": self.duration_seconds,
            "resolved": self.resolved,
            "threat_id": self.threat_id,
        }


# ──────────────────────────────────────────────
# Session Factory
# ──────────────────────────────────────────────
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            config.DATABASE_URL,
            echo=False,
            connect_args=(
                {"check_same_thread": False}
                if "sqlite" in config.DATABASE_URL
                else {}
            ),
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


def get_session() -> Session:
    """Create a new database session."""
    factory = get_session_factory()
    return factory()


def init_db():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised: %s", config.DATABASE_URL)
