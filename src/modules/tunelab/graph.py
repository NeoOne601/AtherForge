# AetherForge v1.0 — src/modules/tunelab/graph.py
# ─────────────────────────────────────────────────────────────────
# TuneLab: Interactive model fine-tuning UI backend module.
# Exposes training job management, OPLoRA visualization,
# and checkpoint browser for the frontend TuneLab panel.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("aetherforge.tunelab")

# Active training jobs: job_id → status dict
_ACTIVE_JOBS: dict[str, dict[str, Any]] = {}


def build_tunelab_graph() -> dict[str, Any]:
    """Build TuneLab module descriptor."""
    return {
        "module_id": "tunelab",
        "run": run_tunelab,
        "list_jobs": list_training_jobs,
        "get_job": get_job_status,
    }


def run_tunelab(
    query: str,
    action: str = "status",
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    TuneLab controller. Accepts natural language queries about fine-tuning
    and dispatches to the appropriate action handler.

    Actions: status | list_jobs | trigger | explain_oplora | capacity
    """
    action_lower = action.lower().strip()
    p = params or {}

    if action_lower == "status":
        from src.config import get_settings
        from src.learning.oploRA_manager import OPLoRAManager
        settings = get_settings()
        manager = OPLoRAManager(settings)
        manager.load_checkpoints()
        capacity = manager.estimate_capacity()
        summary = manager.get_subspace_summary()
        return {
            "status": "ready",
            "learning_capacity_pct": round(capacity * 100, 1),
            "subspace_summary": summary,
            "query": query,
        }

    elif action_lower == "explain_oplora":
        return {
            "explanation": _OPLORA_EXPLANATION,
            "query": query,
        }

    elif action_lower == "list_jobs":
        return {"jobs": list(_ACTIVE_JOBS.values()), "query": query}

    elif action_lower == "trigger":
        job_id = f"manual_{int(time.time())}"
        _ACTIVE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "started_at": time.time(),
            "samples": 0,
        }
        return {"job_id": job_id, "status": "queued", "message": "OPLoRA job queued"}

    return {"query": query, "action": action, "message": "Unknown action"}


def list_training_jobs() -> list[dict[str, Any]]:
    return list(_ACTIVE_JOBS.values())


def get_job_status(job_id: str) -> dict[str, Any] | None:
    return _ACTIVE_JOBS.get(job_id)


_OPLORA_EXPLANATION = """
OPLoRA (Orthogonal Projection LoRA) prevents catastrophic forgetting:

1. After each task T_k, SVD-decompose the LoRA weight update:
   ΔW = U Σ Vᵀ

2. Extract top-k singular vectors:
   U_k = U[:, :k]   (left singular vectors)
   V_k = V[:, :k]   (right singular vectors)

3. Build orthogonal projectors:
   P_L = I - U_k @ U_kᵀ
   P_R = I - V_k @ V_kᵀ

4. Project new task updates into the safe subspace:
   ΔW_safe = P_L @ ΔW_new @ P_R

This guarantees ΔW_safe ⊥ ΔW (T_k), so past knowledge is preserved.
Learning capacity decreases gracefully as more tasks are registered.
"""
