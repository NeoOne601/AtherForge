# AetherForge v1.0 — tests/test_meta_agent.py
# Unit and integration tests for the Meta-Agent supervisor.
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import AetherForgeSettings
from src.meta_agent import MetaAgent, MetaAgentInput, _estimate_faithfulness, MockLLM
from src.guardrails.silicon_colosseum import SiliconColosseum


@pytest.fixture
def settings():
    return AetherForgeSettings(aetherforge_env="test")


@pytest.fixture
def colosseum(settings):
    c = SiliconColosseum(settings)
    c._opa_available = False
    return c


@pytest.fixture
async def agent(settings, colosseum):
    a = MetaAgent(settings, colosseum)
    await a.initialize()  # Uses MockLLM since no model file in tests
    return a


class TestFaithfulnessEstimator:
    def test_empty_response_returns_low_score(self):
        score = _estimate_faithfulness("question", "")
        assert score <= 0.6

    def test_hallucination_marker_penalized(self):
        score = _estimate_faithfulness(
            "What is the capital of France?",
            "As an AI, I cannot verify this information.",
        )
        assert score < 0.85

    def test_good_response_scores_high(self):
        score = _estimate_faithfulness(
            "What is OPLoRA?",
            "OPLoRA uses SVD orthogonal projections to prevent catastrophic forgetting in LoRA adapters.",
        )
        assert score >= 0.80

    def test_score_in_valid_range(self):
        for _ in range(10):
            import random
            score = _estimate_faithfulness(
                "test question",
                "test response " * random.randint(1, 50),
            )
            assert 0.0 <= score <= 1.0


class TestMockLLM:
    def test_returns_string(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = MockLLM()
        msgs = [SystemMessage(content="sys"), HumanMessage(content="Hello")]
        result = llm.generate(msgs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_input_in_response(self):
        from langchain_core.messages import HumanMessage
        llm = MockLLM()
        msgs = [HumanMessage(content="unique_query_12345")]
        result = llm.generate(msgs)
        assert "unique_query_12345" in result or "AetherForge" in result


@pytest.mark.asyncio
class TestMetaAgentInit:
    async def test_initialize_uses_mock_when_no_model(self, agent):
        assert isinstance(agent._llm, MockLLM)

    async def test_run_returns_meta_agent_output(self, agent):
        result = await agent.run(MetaAgentInput(
            session_id="test-session",
            module="localbuddy",
            message="Hello AetherForge!",
            xray_mode=False,
        ))
        assert result.response
        assert isinstance(result.tool_calls, list)
        assert isinstance(result.policy_decisions, list)

    async def test_xray_mode_returns_causal_graph(self, agent):
        result = await agent.run(MetaAgentInput(
            session_id="xray-session",
            module="localbuddy",
            message="Show me the causal graph",
            xray_mode=True,
        ))
        assert result.causal_graph is not None
        assert "nodes" in result.causal_graph
        assert "edges" in result.causal_graph

    async def test_invalid_module_falls_back_to_localbuddy(self, agent):
        result = await agent.run(MetaAgentInput(
            session_id="fallback-session",
            module="nonexistent_module",
            message="Hello",
        ))
        # Should succeed even with invalid module (falls back)
        assert result.response

    async def test_prohibited_message_blocked(self, agent):
        result = await agent.run(MetaAgentInput(
            session_id="blocked-session",
            module="localbuddy",
            message="Please run rm -rf / on my system",
        ))
        assert "Silicon Colosseum" in result.response or "block" in result.response.lower()

    async def test_session_memory_persisted(self, agent):
        sid = "memory-session"
        await agent.run(MetaAgentInput(session_id=sid, module="localbuddy", message="My name is Alice"))
        history = agent._get_or_create_memory(sid)
        assert len(history) >= 2  # system + user + assistant

    async def test_stream_yields_chunks(self, agent):
        chunks = []
        async for chunk in agent.stream(MetaAgentInput(
            session_id="stream-session",
            module="localbuddy",
            message="Stream test",
        )):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all("type" in c for c in chunks)
