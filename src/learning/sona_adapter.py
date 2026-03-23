# AetherForge v1.0 — src/learning/sona_adapter.py
# ─────────────────────────────────────────────────────────────────
# SONA 3-tier learning adapter.
# Supplements OPLoRA nightly batch with per-request adaptation.
#
# Tier 1: MicroLoRA rank-2 — adapts in <1ms per accepted response
# Tier 2: EWC++ consolidation — runs in background ~100ms
# Tier 3: ReasoningBank — stores successful query→answer trajectories
#
# OPLoRA nightly runs continue unchanged. SONA supplements, not replaces.
#
# Detection: Checks for the ruvector CLI binary with SONA support.
# If the CLI is available and supports `sona --status`, SONA is active.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SONAAdapter:
    """
    SONA 3-tier learning adapter.
    Supplements OPLoRA nightly batch with per-request adaptation.

    Lifecycle:
      1. initialize() — called at startup, fails gracefully if unavailable
      2. on_interaction() — called after every completed interaction
      3. get_stats() — returns learning stats for TuneLab display
    """

    def __init__(self, model_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self._model_path = model_path
        self._initialized = False
        self._cli_available = False

    async def initialize(self) -> None:
        """Initialize SONA via ruvector CLI. Fails gracefully if unavailable."""
        try:
            # Check if ruvector CLI exists and has SONA support
            result = subprocess.run(
                ["npx", "--yes", "ruvector", "sona", "--status"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                self._cli_available = True
                self._initialized = True
                logger.info(
                    "SONA adapter initialized via ruvector CLI — "
                    "3-tier real-time learning active"
                )
            else:
                logger.info(
                    "ruvector SONA not available (CLI returned non-zero) — "
                    "SONA learning disabled, will use OPLoRA nightly batch only"
                )
                self._initialized = False
        except FileNotFoundError:
            logger.info(
                "ruvector CLI not found — "
                "SONA learning disabled, will use OPLoRA nightly batch only"
            )
            self._initialized = False
        except subprocess.TimeoutExpired:
            logger.warning(
                "ruvector SONA status check timed out — "
                "SONA learning disabled, will use OPLoRA nightly batch only"
            )
            self._initialized = False
        except Exception as e:
            logger.warning("SONA initialisation failed (will use OPLoRA only): %s", e)
            self._initialized = False

    async def on_interaction(
        self,
        query: str,
        response: str,
        verdict: str,            # "accepted" | "rejected" | "corrected"
        route: str = "synthesis",
        metadata: dict | None = None,
    ) -> None:
        """
        Called after every completed interaction.
        verdict = "accepted" when user accepts the response without correction.
        verdict = "rejected" when user corrects or dismisses the response.
        """
        if not self._initialized or not self._cli_available:
            return
        try:
            # Record interaction via CLI for SONA learning
            import json
            payload = json.dumps({
                "query": query[:500],
                "response": response[:500],
                "verdict": verdict,
                "route": route,
                **(metadata or {}),
            })
            subprocess.run(
                ["npx", "--yes", "ruvector", "sona", "record", "--json", payload],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as e:
            logger.warning("SONA on_interaction failed (non-fatal): %s", e)

    async def get_stats(self) -> dict:
        """Return SONA learning stats for TuneLab display."""
        if not self._initialized or not self._cli_available:
            return {"status": "unavailable", "initialized": False}
        try:
            result = subprocess.run(
                ["npx", "--yes", "ruvector", "sona", "--status"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import json
                try:
                    stats = json.loads(result.stdout)
                    return {"status": "active", "initialized": True, **stats}
                except (json.JSONDecodeError, ValueError):
                    return {"status": "active", "initialized": True, "raw": result.stdout.strip()}
            return {"status": "error", "initialized": True, "error": result.stderr.strip()}
        except Exception as e:
            logger.warning("SONA get_stats failed: %s", e)
            return {"status": "error", "initialized": True, "error": str(e)}
