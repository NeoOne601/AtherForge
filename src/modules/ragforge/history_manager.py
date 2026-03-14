# AetherForge v1.0 — src/modules/ragforge/history_manager.py
# ─────────────────────────────────────────────────────────────────
# RAG performance history persistence.
#
# Stores ThinkingTrace objects as a time-series JSON log.
# This data powers the 'Hill-Climbing' evaluation in the evolution loop.
# ─────────────────────────────────────────────────────────────────

import json
import time
from typing import Any

import structlog

from src.config import AetherForgeSettings

logger = structlog.get_logger("aetherforge.rag_history")


class RAGHistoryManager:
    """
    Manages persistence of RAG performance traces.
    Helps track grounding scores and latencies over time.
    """

    def __init__(self, settings: AetherForgeSettings) -> None:
        self.settings = settings
        self._history_file = settings.data_dir / "rag_performance_history.jsonl"
        self._history_file.parent.mkdir(parents=True, exist_ok=True)

    def record_trace(self, query: str, trace: Any) -> None:
        """
        Record a RAG thinking trace to history.
        `trace` is expected to be a ThinkingTrace object.
        """
        try:
            entry = {
                "timestamp": time.time(),
                "query_preview": query[:100],
                "query_type": trace.query_type,
                "grounding_score": trace.grounding_score,
                "latency_ms": trace.latency_ms,
                "evidence_chunks": trace.evidence_chunks,
                "verification_passed": trace.verification_passed,
                "retrieval_rounds": trace.retrieval_rounds,
            }

            with open(self._history_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            logger.info(
                "Recorded RAG trace: score=%.1f latency=%.0fms",
                trace.grounding_score,
                trace.latency_ms,
            )
        except Exception as e:
            logger.error("Failed to record RAG history: %s", e)

    def get_recent_metrics(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent RAG performance entries."""
        if not self._history_file.exists():
            return []

        results = []
        try:
            with open(self._history_file) as f:
                for line in f:
                    results.append(json.loads(line))
            return results[-limit:]
        except Exception as e:
            logger.error("Failed to read RAG history: %s", e)
            return []

    def get_average_grounding(self, window: int = 20) -> float:
        """Compute average grounding score over a specific window of recent queries."""
        recent = self.get_recent_metrics(limit=window)
        if not recent:
            return 1.0
        scores = [r["grounding_score"] for r in recent]
        return sum(scores) / len(scores)
