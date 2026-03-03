# AetherForge v1.0 — src/modules/watchtower/graph.py
# ─────────────────────────────────────────────────────────────────
# WatchTower: Real-time anomaly detection and system monitoring.
# Uses statistical Z-score + IQR methods for anomaly detection.
# No external dependencies — pure NumPy for edge-device compat.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger("aetherforge.watchtower")

# Sliding window for each metric stream (last 1000 samples)
_METRIC_WINDOWS: dict[str, deque[float]] = {}
_WINDOW_SIZE = 1000
# Z-score threshold for anomaly detection (3σ = 99.7% confidence)
_Z_THRESHOLD = 3.0

# ── Calibration Baseline ──────────────────────────────────────────
# Store the FIRST 30 readings as the "calm baseline" for each metric.
# Z-scores are always compared against this baseline, not the current
# window mean. This prevents baseline drift when the system runs hot
# continuously (e.g. memory at 97% for hours → mean=97 → Z=0 for spikes).
_CALIBRATION_BASELINE: dict[str, dict[str, float]] = {}  # {metric: {mean, std}}
_CALIBRATION_SAMPLES = 30


def build_watchtower_graph() -> dict[str, Any]:
    """Build WatchTower module descriptor."""
    return {
        "module_id": "watchtower",
        "run": run_watchtower,
        "ingest_metric": ingest_metric,
        "get_anomalies": get_recent_anomalies,
    }


def ingest_metric(metric_name: str, value: float) -> dict[str, Any]:
    """
    Ingest a single metric value and check for anomalies.

    Uses a two-phase strategy:
      Phase 1 (first 30 samples): Build a calm baseline.
      Phase 2 (subsequent): Z-score against that fixed baseline mean/std.

    This prevents baseline-drift (the "hot system" problem where sustained
    high values make the Z-score permanently 0).
    """
    if metric_name not in _METRIC_WINDOWS:
        _METRIC_WINDOWS[metric_name] = deque(maxlen=_WINDOW_SIZE)

    window = _METRIC_WINDOWS[metric_name]
    is_anomaly = False
    z_score = 0.0

    if len(window) >= _CALIBRATION_SAMPLES and metric_name not in _CALIBRATION_BASELINE:
        # Lock in the calm baseline from the first N samples
        arr = np.array(list(window)[:_CALIBRATION_SAMPLES])
        _CALIBRATION_BASELINE[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
        logger.info("WatchTower baseline calibrated for '%s': mean=%.2f std=%.2f",
                    metric_name, _CALIBRATION_BASELINE[metric_name]["mean"],
                    _CALIBRATION_BASELINE[metric_name]["std"])

    if metric_name in _CALIBRATION_BASELINE:
        baseline = _CALIBRATION_BASELINE[metric_name]
        std = baseline["std"]
        # If std is too small (very stable metric), use a minimum of 2.0 so spikes register
        effective_std = max(std, 2.0)
        z_score = abs((value - baseline["mean"]) / effective_std)
        is_anomaly = z_score > _Z_THRESHOLD
    elif len(window) >= 10:
        # Pre-calibration fallback: relative Z-score
        arr = np.array(window)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std > 1e-8:
            z_score = abs((value - mean) / std)

    window.append(value)

    result = {
        "metric": metric_name,
        "value": value,
        "is_anomaly": is_anomaly,
        "z_score": round(z_score, 3),
        "window_size": len(window),
        "timestamp": time.time(),
        "baseline_locked": metric_name in _CALIBRATION_BASELINE,
    }

    if is_anomaly:
        logger.warning("ANOMALY detected: %s=%.4f z=%.2f", metric_name, value, z_score)
        _recent_anomalies.append(result)

    return result


def run_watchtower(
    query: str,
    metrics: dict[str, list[float]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Analyze a batch of metrics and return anomaly report.

    Args:
      query:   Natural language query about the metrics
      metrics: Dict of {metric_name: [values...]} to analyze

    Returns a structured anomaly report.
    """
    if not metrics:
        return {"anomalies": [], "summary": "No metrics provided", "query": query}

    anomalies = []
    for metric_name, values in metrics.items():
        for v in values:
            result = ingest_metric(metric_name, float(v))
            if result["is_anomaly"]:
                anomalies.append(result)

    summary = (
        f"Analyzed {sum(len(v) for v in metrics.values())} data points "
        f"across {len(metrics)} metrics. "
        f"Found {len(anomalies)} anomalies."
    )
    logger.info("WatchTower analysis: %s", summary)

    return {
        "anomalies": anomalies,
        "summary": summary,
        "query": query,
        "metrics_analyzed": list(metrics.keys()),
    }


_recent_anomalies: list[dict[str, Any]] = []


def get_recent_anomalies(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent anomaly detections."""
    return _recent_anomalies[-limit:]
