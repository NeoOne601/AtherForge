# AetherForge v1.0 — src/rca/root_cause_agent.py
# ─────────────────────────────────────────────────────────────────
# Root Cause Analysis agent. Given a reported issue or anomaly,
# uses a structured causal reasoning chain to identify probable
# root causes and generates a prioritized remediation plan.
#
# Methodology: 5-Whys iterative LLM chain with evidence collection.
# All reasoning steps are visible in X-Ray mode.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("aetherforge.rca")


@dataclass
class RCAResult:
    """Structured RCA output with causal chain and remediation plan."""
    issue: str
    root_causes: list[str] = field(default_factory=list)
    causal_chain: list[dict[str, str]] = field(default_factory=list)  # [{"why": "...", "because": "..."}]
    evidence: list[str] = field(default_factory=list)
    remediation_steps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class RootCauseAgent:
    """
    5-Whys iterative RCA agent.

    Algorithm:
      1. Start with the reported issue
      2. For each iteration (up to max_depth):
         a. Generate "why did X happen?" prompt
         b. Collect LLM answer as the next level of causation
         c. Stop when no further cause can be identified
      3. Synthesize root causes and remediation steps
    """

    def __init__(self, llm_fn: Any = None, max_depth: int = 5) -> None:
        """
        Args:
          llm_fn:    Callable(prompt: str) -> str. If None, uses heuristic.
          max_depth: Maximum depth of 5-Whys chain (default 5)
        """
        self.llm_fn = llm_fn
        self.max_depth = max_depth

    def analyze(
        self,
        issue: str,
        context: dict[str, Any] | None = None,
        anomalies: list[dict[str, Any]] | None = None,
    ) -> RCAResult:
        """
        Perform root cause analysis on the given issue.

        Args:
          issue:     Description of the observed problem
          context:   Additional context (system state, metrics, etc.)
          anomalies: WatchTower anomaly records to incorporate

        Returns a fully structured RCAResult.
        """
        t0 = time.perf_counter()
        ctx = context or {}
        anoms = anomalies or []

        causal_chain: list[dict[str, str]] = []
        current_issue = issue
        evidence: list[str] = []

        # ── 5-Whys Chain ──────────────────────────────────────────
        for depth in range(self.max_depth):
            why_prompt = self._build_why_prompt(current_issue, depth, ctx, anoms)
            because = self._ask_llm(why_prompt) or f"Insufficient data at depth {depth + 1}"

            causal_chain.append({"depth": depth + 1, "why": current_issue, "because": because})
            evidence.append(f"Depth {depth + 1}: {because}")

            # Stop if LLM indicates root cause found
            if any(marker in because.lower() for marker in [
                "root cause", "fundamental", "primary cause", "no further", "base cause"
            ]):
                logger.info("RCA chain terminated at depth %d", depth + 1)
                break

            current_issue = because

        # ── Extract root causes (last 1-2 chain entries) ──────────
        root_causes = [
            chain_item["because"]
            for chain_item in causal_chain[-2:]
        ]

        # ── Generate remediation plan ─────────────────────────────
        remediation = self._generate_remediation(issue, root_causes, ctx)

        # ── Confidence estimation ─────────────────────────────────
        # Higher confidence when more anomaly evidence correlates
        confidence = min(0.95, 0.6 + 0.05 * len(causal_chain) + 0.05 * len(anoms))

        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info("RCA complete: depth=%d root_causes=%d duration=%.1fms",
                    len(causal_chain), len(root_causes), duration_ms)

        return RCAResult(
            issue=issue,
            root_causes=root_causes,
            causal_chain=causal_chain,
            evidence=evidence,
            remediation_steps=remediation,
            confidence=round(confidence, 3),
            duration_ms=round(duration_ms, 2),
        )

    def _build_why_prompt(
        self,
        issue: str,
        depth: int,
        context: dict[str, Any],
        anomalies: list[dict[str, Any]],
    ) -> str:
        anomaly_text = ""
        if anomalies:
            anomaly_text = f"\nRelated anomalies: {[a.get('metric', '') for a in anomalies[:3]]}"

        return (
            f"Root Cause Analysis — Step {depth + 1}\n"
            f"Observed: {issue}{anomaly_text}\n"
            f"Context: {context}\n\n"
            "Why did this happen? Identify the most probable direct cause. "
            "If this IS the root cause, say 'Root cause: ...' and stop."
        )

    def _generate_remediation(
        self,
        issue: str,
        root_causes: list[str],
        context: dict[str, Any],
    ) -> list[str]:
        prompt = (
            f"Issue: {issue}\n"
            f"Root causes identified: {root_causes}\n\n"
            "List 3-5 specific, actionable remediation steps in order of priority."
        )
        response = self._ask_llm(prompt) or ""
        # Parse numbered list from LLM response
        steps = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 5
        ][:5]
        if not steps:
            steps = ["Investigate root causes further", "Review system logs", "Monitor affected metrics"]
        return steps

    def _ask_llm(self, prompt: str) -> str:
        """Call the configured LLM function or return a heuristic response."""
        if not self.llm_fn:
            # Heuristic fallback when no LLM is configured
            return "Insufficient diagnostic context — please provide more system state."
        try:
            return str(self.llm_fn(prompt))
        except Exception as exc:
            logger.exception("RCA LLM call failed: %s", exc)
            return ""
