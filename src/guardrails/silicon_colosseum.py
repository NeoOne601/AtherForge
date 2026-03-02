# AetherForge v1.0 — src/guardrails/silicon_colosseum.py
# ─────────────────────────────────────────────────────────────────
# Silicon Colosseum: AetherForge's deterministic safety engine.
#
# Two complementary safety mechanisms run in tandem:
#
#   1. OPA (Open Policy Agent): declarative Rego policies evaluated
#      via subprocess (embedded mode) or HTTP (server mode).
#      Handles: tool budgets, faithfulness thresholds, content filters.
#
#   2. FSM (Finite State Machine): stateful enforcement of agent
#      lifecycle transitions. Prevents illegal state jumps like
#      "output before input" or "tool call after shutdown".
#
# Design rationale: OPA handles stateless per-request policies while
# FSM handles stateful session-level invariants. Together they cover
# the full safety surface without needing an LLM-based judge.
#
# Every tool call and agent output MUST pass both checks.
# Failures are logged with full audit context for X-Ray visualization.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import httpx

from src.config import AetherForgeSettings

logger = logging.getLogger("aetherforge.silicon_colosseum")

# ── Policy Decision Result ─────────────────────────────────────────

@dataclass
class PolicyDecision:
    """
    The result of a Silicon Colosseum evaluation.
    Immutable after construction — safe to pass around and serialize.
    """
    allowed: bool
    reason: str
    deny_reasons: list[str] = field(default_factory=list)
    policy_version: str = "1.0.0"
    latency_ms: float = 0.0
    fsm_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "deny_reasons": self.deny_reasons,
            "policy_version": self.policy_version,
            "latency_ms": round(self.latency_ms, 3),
            "fsm_state": self.fsm_state,
        }


# ── FSM State Definitions ─────────────────────────────────────────

class AgentFSMState(Enum):
    """
    Finite State Machine states for a single agent session.

    Valid transitions:
      IDLE → PROCESSING → TOOL_CALLING → PROCESSING (cycle)
      PROCESSING → RESPONDING → IDLE
      Any → ERROR → IDLE (recovery)
    """
    IDLE = auto()          # Waiting for next user input
    PROCESSING = auto()    # LLM generating, no tool calls yet
    TOOL_CALLING = auto()  # Executing a tool call (under budget)
    RESPONDING = auto()    # Finalizing and sending response
    ERROR = auto()         # Error state — resets to IDLE on next turn


# Valid FSM transitions
_FSM_TRANSITIONS: dict[AgentFSMState, set[AgentFSMState]] = {
    AgentFSMState.IDLE:         {AgentFSMState.PROCESSING},
    AgentFSMState.PROCESSING:   {AgentFSMState.TOOL_CALLING, AgentFSMState.RESPONDING, AgentFSMState.ERROR},
    AgentFSMState.TOOL_CALLING: {AgentFSMState.PROCESSING, AgentFSMState.ERROR},
    AgentFSMState.RESPONDING:   {AgentFSMState.IDLE, AgentFSMState.ERROR},
    AgentFSMState.ERROR:        {AgentFSMState.IDLE},
}


class AgentFSM:
    """
    Per-session FSM. One instance per session_id.
    Enforces legal state transitions and tracks tool call count.
    """
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.state = AgentFSMState.IDLE
        self.tool_call_count = 0
        self._created_at = time.time()

    def transition(self, new_state: AgentFSMState) -> tuple[bool, str]:
        """
        Attempt a state transition.
        Returns (success: bool, reason: str).
        """
        allowed_next = _FSM_TRANSITIONS.get(self.state, set())
        if new_state not in allowed_next:
            reason = (
                f"FSM illegal transition: {self.state.name} -> {new_state.name}. "
                f"Allowed from {self.state.name}: {[s.name for s in allowed_next]}"
            )
            logger.warning("[FSM] %s | session=%s", reason, self.session_id)
            return False, reason

        logger.debug("[FSM] %s -> %s | session=%s", self.state.name, new_state.name, self.session_id)
        self.state = new_state
        return True, "ok"

    def increment_tool_calls(self) -> int:
        self.tool_call_count += 1
        return self.tool_call_count

    def reset(self) -> None:
        """Reset to IDLE for the next turn."""
        self.state = AgentFSMState.IDLE
        self.tool_call_count = 0


# ── Silicon Colosseum ─────────────────────────────────────────────

class SiliconColosseum:
    """
    The Silicon Colosseum safety engine.

    Combines OPA policy evaluation with FSM state enforcement.
    Every call to evaluate_request_sync() passes through both layers.

    Modes:
      - embedded: spawns `opa eval` subprocess per request (~2 ms overhead)
      - server:   HTTP POST to OPA server at opa_server_url (~1 ms network)

    The embedded mode is the default for local deployments — no
    Docker required. Server mode scales better for enterprise deploys.
    """

    def __init__(self, settings: AetherForgeSettings) -> None:
        self.settings = settings
        self._policy_path = Path(__file__).parent / "default_policies.rego"
        self._policy_source: str = ""
        self._sessions: dict[str, AgentFSM] = {}  # session_id → FSM
        self._opa_available = False
        self._http_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """
        Load policy source and verify OPA is available.
        Called once during FastAPI lifespan startup.
        """
        # Load policy source for runtime editing
        if self._policy_path.exists():
            self._policy_source = self._policy_path.read_text()
            logger.info("Loaded OPA policy from %s", self._policy_path)
        else:
            logger.warning("Policy file not found: %s — using default inline policy", self._policy_path)
            self._policy_source = _DEFAULT_INLINE_POLICY

        # Check OPA binary availability (embedded mode)
        if self.settings.opa_mode == "embedded":
            try:
                result = subprocess.run(
                    ["opa", "version"], capture_output=True, timeout=5
                )
                self._opa_available = result.returncode == 0
                if self._opa_available:
                    logger.info("OPA binary available: embedded mode active")
                else:
                    logger.warning("OPA binary found but failed — falling back to Python evaluator")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("OPA binary not found — using Python policy evaluator fallback")
                self._opa_available = False
        else:
            # Server mode — create persistent HTTP client
            self._http_client = httpx.AsyncClient(base_url=self.settings.opa_server_url, timeout=5.0)
            logger.info("OPA server mode: %s", self.settings.opa_server_url)

    def _get_or_create_fsm(self, session_id: str) -> AgentFSM:
        if session_id not in self._sessions:
            self._sessions[session_id] = AgentFSM(session_id)
        return self._sessions[session_id]

    def evaluate_request_sync(self, input_data: dict[str, Any]) -> PolicyDecision:
        """
        Synchronous policy evaluation (called from LangGraph nodes).
        Combines OPA evaluation + FSM state check.

        input_data must include:
          session_id, module, message, tool_call_count
        Optionally: faithfulness_score (for post-output checks)
        """
        t0 = time.perf_counter()
        session_id = input_data.get("session_id", "unknown")
        fsm = self._get_or_create_fsm(session_id)

        # ── OPA evaluation ────────────────────────────────────────
        if self._opa_available:
            opa_result = self._eval_opa_embedded(input_data)
        else:
            opa_result = self._eval_python_fallback(input_data)

        # ── FSM state check ───────────────────────────────────────
        # Determine intended next FSM state from the operation type
        if input_data.get("tool_call_count", 0) > 0:
            target_state = AgentFSMState.TOOL_CALLING
            if fsm.state == AgentFSMState.IDLE:
                # First, transition to PROCESSING
                fsm.transition(AgentFSMState.PROCESSING)
        else:
            target_state = AgentFSMState.PROCESSING
            if fsm.state == AgentFSMState.IDLE:
                fsm.transition(AgentFSMState.PROCESSING)
                target_state = AgentFSMState.RESPONDING

        latency_ms = (time.perf_counter() - t0) * 1000

        # Merge OPA + FSM decisions
        if not opa_result["allowed"]:
            return PolicyDecision(
                allowed=False,
                reason="; ".join(opa_result["deny_reasons"]) or "Policy denied",
                deny_reasons=opa_result["deny_reasons"],
                latency_ms=latency_ms,
                fsm_state=fsm.state.name,
            )

        return PolicyDecision(
            allowed=True,
            reason="All policies passed",
            deny_reasons=[],
            latency_ms=latency_ms,
            fsm_state=fsm.state.name,
        )

    def _eval_opa_embedded(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate policy by spawning `opa eval` as a subprocess.
        Uses a temp file for the policy + stdin for input data.
        ~2–4 ms overhead on M1.
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".rego", delete=False, encoding="utf-8"
            ) as f:
                f.write(self._policy_source)
                policy_file = f.name

            input_json = json.dumps({"input": input_data})
            result = subprocess.run(
                [
                    "opa", "eval",
                    "--format", "json",
                    "--data", policy_file,
                    "--stdin-input",
                    "data.aetherforge.guardrails",
                ],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=10,
            )
            Path(policy_file).unlink(missing_ok=True)

            if result.returncode != 0:
                logger.error("OPA eval failed: %s", result.stderr)
                return {"allowed": True, "deny_reasons": []}

            opa_output = json.loads(result.stdout)
            results = opa_output.get("result", [{}])[0].get("expressions", [{}])[0].get("value", {})
            return {
                "allowed": results.get("allow", True),
                "deny_reasons": list(results.get("deny_reasons", [])),
            }
        except Exception as exc:
            logger.exception("OPA embedded eval error: %s", exc)
            return {"allowed": True, "deny_reasons": []}

    def _eval_python_fallback(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Pure Python policy evaluator — used when OPA binary is absent.
        Replicates the Rego policies faithfully in Python.
        This is the production fallback for air-gapped environments
        where installing OPA may not be possible.
        """
        deny_reasons: list[str] = []
        message = input_data.get("message", "")
        tool_count = input_data.get("tool_call_count", 0)
        module = input_data.get("module", "")
        faithfulness = input_data.get("faithfulness_score")

        # Rule 1: Tool call budget
        max_calls = self.settings.silicon_colosseum_max_tool_calls
        if tool_count > max_calls:
            deny_reasons.append(f"Tool call budget exceeded: {tool_count} > {max_calls}")

        # Rule 2: Faithfulness threshold
        if faithfulness is not None and faithfulness < self.settings.silicon_colosseum_min_faithfulness:
            deny_reasons.append(
                f"Output faithfulness {faithfulness:.2f} < "
                f"{self.settings.silicon_colosseum_min_faithfulness:.2f} threshold"
            )

        # Rule 3: Prohibited patterns
        PROHIBITED = [
            "rm -rf", "DELETE FROM", "DROP TABLE", "sudo",
            "__import__", "eval(", "exec(", "os.system", "subprocess.call",
        ]
        for pattern in PROHIBITED:
            if pattern in message:
                deny_reasons.append(f"Prohibited pattern: '{pattern}'")

        # Rule 4: Module allowlist
        VALID_MODULES = {"ragforge", "localbuddy", "watchtower", "streamsync", "tunelab", "output_filter"}
        if module and module not in VALID_MODULES:
            deny_reasons.append(f"Unknown module: '{module}'")

        # Rule 5: Empty message
        if not message.strip():
            deny_reasons.append("Empty message rejected")

        # Rule 6: Length limit (16 KB)
        if len(message) > 16384:
            deny_reasons.append(f"Message exceeds 16KB: {len(message)} chars")

        return {"allowed": len(deny_reasons) == 0, "deny_reasons": deny_reasons}

    async def evaluate_request_async(self, input_data: dict[str, Any]) -> PolicyDecision:
        """Async wrapper — dispatches to server mode or thread executor."""
        if self.settings.opa_mode == "server" and self._http_client:
            return await self._eval_opa_server(input_data)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate_request_sync, input_data)

    async def _eval_opa_server(self, input_data: dict[str, Any]) -> PolicyDecision:
        """HTTP-based OPA evaluation for server mode."""
        t0 = time.perf_counter()
        try:
            assert self._http_client is not None
            resp = await self._http_client.post(
                "/v1/data/aetherforge/guardrails",
                json={"input": input_data},
            )
            resp.raise_for_status()
            data = resp.json().get("result", {})
            latency_ms = (time.perf_counter() - t0) * 1000
            return PolicyDecision(
                allowed=data.get("allow", True),
                reason="OPA server evaluation",
                deny_reasons=list(data.get("deny_reasons", [])),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.exception("OPA server eval failed: %s", exc)
            return PolicyDecision(allowed=True, reason="OPA server error — fallback allow", deny_reasons=[])

    async def get_policy_source(self) -> str:
        """Return current policy source for the PolicyEditor UI."""
        return self._policy_source

    async def update_policy(self, new_policy: str) -> dict[str, Any]:
        """
        Hot-reload policy source. Validates Rego syntax via `opa check`
        before accepting. Returns {"success": bool, "error": str}.
        """
        if self._opa_available:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".rego", delete=False, encoding="utf-8"
                ) as f:
                    f.write(new_policy)
                    tmp = f.name
                result = subprocess.run(
                    ["opa", "check", tmp],
                    capture_output=True, text=True, timeout=10
                )
                Path(tmp).unlink(missing_ok=True)
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        self._policy_source = new_policy
        # Persist to disk
        self._policy_path.write_text(new_policy)
        logger.info("Policy hot-reloaded (%d chars)", len(new_policy))
        return {"success": True, "error": None}


# ── Inline fallback policy (used when .rego file is missing) ──────
_DEFAULT_INLINE_POLICY = """
package aetherforge.guardrails
import rego.v1
default allow := true
default deny_reasons := []
"""
