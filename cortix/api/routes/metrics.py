"""
CortiX Dashboard — System Metrics Router
"""

import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from cortix.database import get_session, SystemMetric

logger = logging.getLogger("cortix.api.routes.metrics")

router = APIRouter(prefix="/metrics", tags=["metrics"])


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_metrics(limit: int = 100, db: Session = Depends(get_db)):
    """Get historical system performance metrics (latency, FPR, FNR, throughput)."""
    metrics = db.query(SystemMetric).order_by(SystemMetric.timestamp.desc()).limit(limit).all()
    return [m.to_dict() for m in reversed(metrics)]


@router.get("/summary")
def get_metrics_summary(db: Session = Depends(get_db)):
    """Aggregate high level health stats for the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    metrics = db.query(SystemMetric).filter(SystemMetric.timestamp >= since).all()
    
    if not metrics:
        return {
            "p50_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "avg_fpr": 0.0,
            "avg_throughput_pps": 0.0,
            "total_events_processed": 0,
        }
        
    avg_p50 = sum(m.latency_p50_ms for m in metrics) / len(metrics)
    avg_p99 = sum(m.latency_p99_ms for m in metrics) / len(metrics)
    avg_tps = sum(m.throughput_pps for m in metrics) / len(metrics)
    total_events = sum(m.event_count for m in metrics)
    
    total_fp = sum(m.fp_count for m in metrics)
    total_tp = sum(m.tp_count for m in metrics)
    total_fn = sum(m.fn_count for m in metrics)
    
    # Calculate global FPR
    fpr = total_fp / max(total_fp + (total_events - total_tp - total_fp - total_fn), 1)
    
    return {
        "p50_latency_ms": float(avg_p50),
        "p99_latency_ms": float(avg_p99),
        "avg_fpr": float(fpr),
        "avg_throughput_pps": float(avg_tps),
        "total_events_processed": int(total_events),
    }
