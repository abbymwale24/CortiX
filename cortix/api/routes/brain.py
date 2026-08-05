"""
CortiX Dashboard — Brain / Synaptic Weight Visualization API

Exposes the SNN Hebbian ensemble's internal weight distributions,
heatmaps, and metaplasticity state for live dashboard rendering.
"""

import logging
import time
import numpy as np
from collections import deque
from fastapi import APIRouter, Query

logger = logging.getLogger("cortix.api.routes.brain")

router = APIRouter(prefix="/brain", tags=["brain"])

# ──────────────────────────────────────────────
# Shared SNN Ensemble reference (set on startup)
# ──────────────────────────────────────────────
_ensemble = None
_weight_history = deque(maxlen=100)
_last_snapshot_time = 0


def set_ensemble(ensemble):
    """Register the SNN ensemble instance for API access."""
    global _ensemble
    _ensemble = ensemble
    logger.info("Brain API: SNN ensemble registered")


def _get_ensemble():
    """Get the ensemble, creating a default one if none registered."""
    global _ensemble
    if _ensemble is None:
        from cortix.snn.ensemble import HebbianEnsemble
        _ensemble = HebbianEnsemble()
        logger.info("Brain API: Created default HebbianEnsemble for visualization")
    return _ensemble


def _snapshot_weights():
    """Take a periodic snapshot of weight statistics for history."""
    global _last_snapshot_time
    now = time.time()
    if now - _last_snapshot_time < 5.0:  # Max one snapshot per 5 seconds
        return
    _last_snapshot_time = now

    ensemble = _get_ensemble()
    snapshot = {
        "timestamp": now,
        "modules": [],
    }
    for i, module in enumerate(ensemble.modules):
        W = module.W
        snapshot["modules"].append({
            "module_id": i,
            "weight_mean": float(np.mean(W)),
            "weight_std": float(np.std(W)),
            "sparsity": float(np.mean(np.abs(W) < 1e-4)),
        })
    _weight_history.append(snapshot)


@router.get("/weights")
def get_brain_weights():
    """
    Return per-module weight statistics and histograms.
    """
    ensemble = _get_ensemble()
    _snapshot_weights()

    modules = []
    for i, module in enumerate(ensemble.modules):
        W = module.W
        flat = W.flatten()

        # 20-bin histogram
        hist_counts, hist_edges = np.histogram(flat, bins=20)

        # Compute statistics
        w_mean = float(np.mean(flat))
        w_std = float(np.std(flat))
        threshold_high = w_mean + 2 * w_std
        top_synapses = int(np.sum(flat > threshold_high))
        dead_synapses = int(np.sum(np.abs(flat) < 1e-6))
        sparsity = float(np.mean(np.abs(flat) < 1e-4))

        modules.append({
            "module_id": i,
            "weight_mean": w_mean,
            "weight_std": w_std,
            "weight_min": float(np.min(flat)),
            "weight_max": float(np.max(flat)),
            "histogram": hist_counts.tolist(),
            "histogram_bins": hist_edges.tolist(),
            "top_synapses": top_synapses,
            "dead_synapses": dead_synapses,
            "sparsity": sparsity,
        })

    # Metaplasticity state
    meta = ensemble.metaplasticity
    meta_state = {
        "current_eta": float(meta.eta),
        "base_eta": float(meta.eta_0),
        "adaptation_ratio": float(meta.eta / meta.eta_0) if meta.eta_0 > 0 else 1.0,
    }

    return {
        "modules": modules,
        "metaplasticity": meta_state,
        "total_events_processed": ensemble.total_processed,
    }


@router.get("/weights/heatmap")
def get_weight_heatmap(module: int = Query(0, ge=0, le=4)):
    """
    Return a downsampled weight matrix (32×32) for heatmap rendering.
    """
    ensemble = _get_ensemble()

    if module >= len(ensemble.modules):
        return {"error": f"Module {module} does not exist"}

    W = ensemble.modules[module].W  # shape: (n_hidden, n_input)
    h, w = W.shape

    # Downsample to 32×32 using block averaging
    target_h, target_w = 32, 32
    block_h = max(1, h // target_h)
    block_w = max(1, w // target_w)

    # Trim to exact multiples
    trimmed_h = block_h * target_h
    trimmed_w = block_w * target_w
    W_trimmed = W[:trimmed_h, :trimmed_w]

    # Reshape and average
    downsampled = W_trimmed.reshape(target_h, block_h, target_w, block_w).mean(axis=(1, 3))

    # Normalize to [0, 1] for color mapping
    d_min = float(downsampled.min())
    d_max = float(downsampled.max())
    d_range = d_max - d_min if d_max > d_min else 1.0
    normalized = ((downsampled - d_min) / d_range).tolist()

    return {
        "module_id": module,
        "size": target_h,
        "data": normalized,
        "weight_min": d_min,
        "weight_max": d_max,
    }


@router.get("/weights/history")
def get_weight_history():
    """
    Return rolling weight statistics over time (last 100 snapshots).
    """
    _snapshot_weights()
    return list(_weight_history)
