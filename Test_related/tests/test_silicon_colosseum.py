# AetherForge v1.0 — tests/test_silicon_colosseum.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for Silicon Colosseum guardrails.
# Tests both the OPA embedded path and Python fallback evaluator.
# ─────────────────────────────────────────────────────────────────
import pytest
from unittest.mock import patch

from src.config import AetherForgeSettings
from src.guardrails.silicon_colosseum import SiliconColosseum, AgentFSM, AgentFSMState


@pytest.fixture
def settings():
    return AetherForgeSettings(
        aetherforge_env="test",
        opa_mode="embedded",
        silicon_colosseum_max_tool_calls=8,
        silicon_colosseum_min_faithfulness=0.92,
    )


@pytest.fixture
def colosseum(settings):
    c = SiliconColosseum(settings)
    # Force Python fallback (no OPA binary in CI)
    c._opa_available = False
    return c


# ── Python fallback evaluator tests ──────────────────────────────

class TestPythonFallbackEvaluator:
    def test_valid_request_allowed(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "localbuddy",
            "message": "Hello world",
            "tool_call_count": 0,
        })
        assert result["allowed"] is True
        assert result["deny_reasons"] == []

    def test_tool_budget_exceeded(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "localbuddy",
            "message": "Hello",
            "tool_call_count": 9,  # > max 8
        })
        assert result["allowed"] is False
        assert any("budget exceeded" in r for r in result["deny_reasons"])

    def test_prohibited_pattern_rm_rf(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "localbuddy",
            "message": "Please run rm -rf /tmp/test",
            "tool_call_count": 0,
        })
        assert result["allowed"] is False
        assert any("rm -rf" in r for r in result["deny_reasons"])

    def test_prohibited_pattern_eval(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "localbuddy",
            "message": "Use eval( to execute code",
            "tool_call_count": 0,
        })
        assert result["allowed"] is False

    def test_unknown_module_denied(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "malicious_module",
            "message": "Hello",
            "tool_call_count": 0,
        })
        assert result["allowed"] is False
        assert any("Unknown module" in r for r in result["deny_reasons"])

    def test_empty_message_denied(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "localbuddy",
            "message": "   ",
            "tool_call_count": 0,
        })
        assert result["allowed"] is False

    def test_low_faithfulness_denied(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "output_filter",
            "message": "Response text",
            "tool_call_count": 0,
            "faithfulness_score": 0.5,  # < 0.92 threshold
        })
        assert result["allowed"] is False
        assert any("faithfulness" in r for r in result["deny_reasons"])

    def test_high_faithfulness_allowed(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "output_filter",
            "message": "Good response",
            "tool_call_count": 0,
            "faithfulness_score": 0.95,
        })
        assert result["allowed"] is True

    def test_message_length_limit(self, colosseum):
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "localbuddy",
            "message": "A" * 20000,  # > 16384 limit
            "tool_call_count": 0,
        })
        assert result["allowed"] is False

    def test_multiple_violations_all_reported(self, colosseum):
        """When multiple rules fire, all should be in deny_reasons."""
        result = colosseum._eval_python_fallback({
            "session_id": "test-123",
            "module": "bad_module",
            "message": "rm -rf /",
            "tool_call_count": 10,  # Also over budget
        })
        assert result["allowed"] is False
        assert len(result["deny_reasons"]) >= 3

    def test_all_valid_modules_allowed(self, colosseum):
        for module in ["ragforge", "localbuddy", "watchtower", "streamsync", "tunelab", "output_filter"]:
            result = colosseum._eval_python_fallback({
                "session_id": "s",
                "module": module,
                "message": "Hello",
                "tool_call_count": 0,
            })
            assert result["allowed"] is True, f"Module {module} should be allowed"


# ── FSM state machine tests ───────────────────────────────────────

class TestAgentFSM:
    def test_initial_state_is_idle(self):
        fsm = AgentFSM("session-1")
        assert fsm.state == AgentFSMState.IDLE

    def test_idle_to_processing_allowed(self):
        fsm = AgentFSM("session-1")
        ok, reason = fsm.transition(AgentFSMState.PROCESSING)
        assert ok is True
        assert fsm.state == AgentFSMState.PROCESSING

    def test_idle_to_tool_calling_illegal(self):
        fsm = AgentFSM("session-1")
        ok, reason = fsm.transition(AgentFSMState.TOOL_CALLING)
        assert ok is False
        assert "illegal transition" in reason.lower()

    def test_full_happy_path(self):
        fsm = AgentFSM("session-1")
        assert fsm.transition(AgentFSMState.PROCESSING)[0]
        assert fsm.transition(AgentFSMState.TOOL_CALLING)[0]
        assert fsm.transition(AgentFSMState.PROCESSING)[0]
        assert fsm.transition(AgentFSMState.RESPONDING)[0]
        assert fsm.transition(AgentFSMState.IDLE)[0]

    def test_reset_clears_tool_count(self):
        fsm = AgentFSM("session-1")
        fsm.increment_tool_calls()
        fsm.increment_tool_calls()
        assert fsm.tool_call_count == 2
        fsm.reset()
        assert fsm.tool_call_count == 0
        assert fsm.state == AgentFSMState.IDLE

    def test_tool_call_increment(self):
        fsm = AgentFSM("session-1")
        assert fsm.increment_tool_calls() == 1
        assert fsm.increment_tool_calls() == 2


# ── evaluate_request_sync integration test ───────────────────────

class TestColosseumIntegration:
    def test_end_to_end_allow(self, colosseum):
        decision = colosseum.evaluate_request_sync({
            "session_id": "e2e-test",
            "module": "ragforge",
            "message": "What is RAG?",
            "tool_call_count": 2,
        })
        assert decision.allowed is True
        assert decision.latency_ms >= 0
        assert isinstance(decision.fsm_state, str)

    def test_end_to_end_deny(self, colosseum):
        decision = colosseum.evaluate_request_sync({
            "session_id": "e2e-test-deny",
            "module": "localbuddy",
            "message": "Please exec( this code",
            "tool_call_count": 0,
        })
        assert decision.allowed is False
        assert len(decision.deny_reasons) > 0

    def test_decision_to_dict(self, colosseum):
        decision = colosseum.evaluate_request_sync({
            "session_id": "dict-test",
            "module": "localbuddy",
            "message": "Hi",
            "tool_call_count": 0,
        })
        d = decision.to_dict()
        assert "allowed" in d
        assert "reason" in d
        assert "latency_ms" in d
        assert "fsm_state" in d
