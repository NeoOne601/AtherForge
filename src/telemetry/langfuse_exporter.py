# AetherForge v1.0 — src/telemetry/langfuse_exporter.py
# ─────────────────────────────────────────────────────────────────
# Local Langfuse telemetry exporter. Sends traces to self-hosted
# Langfuse instance (localhost:3000 via docker-compose).
#
# Design: This is entirely optional and off by default.
# Set LANGFUSE_ENABLED=true in .env to activate.
# All data stays on-device — the Langfuse Docker container is local.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import structlog
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

logger = structlog.get_logger("aetherforge.telemetry")


class LangfuseExporter:
    """
    Thin wrapper around the Langfuse Python SDK for local telemetry.
    Gracefully disabled when Langfuse is not configured or reachable.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._client: Any = None
        self._enabled = settings.langfuse_enabled

    def initialize(self) -> None:
        """Initialize Langfuse client if enabled."""
        if not self._enabled:
            logger.info("Langfuse telemetry disabled (LANGFUSE_ENABLED=false)")
            return
        try:
            from langfuse import Langfuse  # type: ignore[import]
            self._client = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
            logger.info("Langfuse telemetry active: %s", self.settings.langfuse_host)
        except ImportError:
            logger.warning("langfuse package not installed — telemetry disabled")
            self._enabled = False
        except Exception as exc:
            logger.warning("Langfuse init failed: %s — telemetry disabled", exc)
            self._enabled = False

    def trace_interaction(
        self,
        session_id: str,
        module: str,
        prompt: str,
        response: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Record a single interaction trace in Langfuse.
        Returns trace_id or None if telemetry is disabled.
        """
        if not self._enabled or self._client is None:
            return None

        try:
            trace_id = str(uuid.uuid4())
            trace = self._client.trace(
                id=trace_id,
                name=f"aetherforge.{module}",
                user_id=session_id,
                session_id=session_id,
                metadata={
                    "module": module,
                    "latency_ms": latency_ms,
                    **(metadata or {}),
                },
                tags=["aetherforge", f"module:{module}"],
            )
            trace.generation(
                name="llm_generation",
                model="BitNet-1.58b",
                prompt=prompt[:2000],
                completion=response[:2000],
                usage={
                    "promptTokens": len(prompt.split()),
                    "completionTokens": len(response.split()),
                },
                metadata={"local_model": True, "cloud": False},
            )
            logger.debug("Langfuse trace recorded: %s", trace_id)
            return trace_id
        except Exception as exc:
            logger.debug("Langfuse trace failed (non-fatal): %s", exc)
            return None

    def trace_policy_decision(
        self,
        session_id: str,
        allowed: bool,
        reason: str,
        latency_ms: float,
    ) -> None:
        """Record a Silicon Colosseum policy decision."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.event(
                name="silicon_colosseum.decision",
                user_id=session_id,
                metadata={
                    "allowed": allowed,
                    "reason": reason,
                    "latency_ms": latency_ms,
                },
                level="DEFAULT" if allowed else "WARNING",
            )
        except Exception:
            pass  # Non-fatal

    def flush(self) -> None:
        """Flush all pending traces to Langfuse server."""
        if self._enabled and self._client:
            try:
                self._client.flush()
            except Exception as exc:
                logger.debug("Langfuse flush failed: %s", exc)

    @property
    def is_active(self) -> bool:
        return self._enabled and self._client is not None
