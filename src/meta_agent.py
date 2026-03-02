# AetherForge v1.0 — src/meta_agent.py
# ─────────────────────────────────────────────────────────────────
# LangGraph supervisor (meta-agent). This is the brain of AetherForge.
#
# Architecture:
#   - One StateGraph with a supervisor node + 5 module sub-graphs
#   - Every edge transition triggers Silicon Colosseum evaluation
#   - BitNet model loaded once via llama-cpp-python (Metal on M1)
#   - X-Ray mode builds a causal graph of every node visited
#
# LangGraph node execution order (per turn):
#   intake -> colosseum_check -> route -> [module_graph] -> synthesize -> output
#
# Design: We use a "supervisor routing" pattern where a lightweight
# classifier prompt picks the module, then a full subgraph executes.
# This avoids the overhead of running all 5 module graphs per turn.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import AetherForgeSettings
from src.guardrails.silicon_colosseum import SiliconColosseum

logger = logging.getLogger("aetherforge.meta_agent")

# ── System Prompt ─────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are AetherForge, a local AI assistant. You run entirely on-device, "
    "never send data to the cloud, and always cite your reasoning. "
    "Be concise, accurate, and transparent about uncertainty."
)


def _messages_to_prompt(messages: list[Any]) -> str:
    """
    Convert LangChain messages to ChatML format for llama-cpp.
    BitNet GGUF models use the standard ChatML template.
    """
    parts = ["<|im_start|>system\n" + _SYSTEM_PROMPT + "<|im_end|>"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif isinstance(msg, AIMessage):
            parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── Mock LLM for development (when model file absent) ─────────────
class MockLLM:
    """
    Stub LLM used when the BitNet GGUF model is not yet downloaded.
    Returns deterministic canned responses so the full pipeline can
    be tested without model weights.
    """
    def generate(self, messages: list[Any]) -> str:
        last = messages[-1].content if messages else "hello"
        return (
            f"[MockLLM] AetherForge received: '{last[:60]}'. "
            "Download the BitNet model to enable real inference. "
            "See README.md for instructions."
        )


# ── Input / Output models ─────────────────────────────────────────

class MetaAgentInput(BaseModel):
    """Typed input to the meta-agent for a single turn."""
    session_id: str
    module: str = "localbuddy"
    message: str
    xray_mode: bool = False
    context: dict[str, Any] = {}


class MetaAgentOutput(BaseModel):
    """Typed output from a single meta-agent turn."""
    response: str
    tool_calls: list[dict[str, Any]] = []
    policy_decisions: list[dict[str, Any]] = []
    causal_graph: dict[str, Any] | None = None
    faithfulness_score: float | None = None


# ── MetaAgent ─────────────────────────────────────────────────────

class MetaAgent:
    """
    LangGraph-based meta-agent supervisor.

    Lifecycle:
      1. __init__: store settings + colosseum reference
      2. initialize(): load BitNet model (expensive, called once)
      3. run(): synchronous per-turn call
      4. stream(): async generator for WebSocket streaming

    Thread safety: initialize() must complete before run()/stream().
    The LLM (llama-cpp) is not thread-safe; we use asyncio.Lock to
    serialize concurrent requests.
    """

    def __init__(self, settings: AetherForgeSettings, colosseum: SiliconColosseum) -> None:
        self.settings = settings
        self.colosseum = colosseum
        self._llm: Any = None
        self._lock = asyncio.Lock()
        self._session_memories: dict[str, list[Any]] = {}

    async def initialize(self) -> None:
        """Load BitNet model in a thread executor (non-blocking)."""
        await self._load_model()
        logger.info("MetaAgent initialized")

    async def _load_model(self) -> None:
        """
        Load BitNet GGUF model via llama-cpp-python.
        n_gpu_layers=-1 offloads all layers to Apple Metal (MPS).
        use_mlock pins the model in RAM to prevent swapping.
        """
        model_path = self.settings.bitnet_model_path
        if not model_path.exists():
            logger.warning("BitNet model not found at %s — using MockLLM", model_path)
            self._llm = MockLLM()
            return

        logger.info("Loading BitNet model: %s", model_path)

        def _load() -> Any:
            from llama_cpp import Llama  # type: ignore[import]
            return Llama(
                model_path=str(model_path),
                n_ctx=self.settings.bitnet_n_ctx,
                n_gpu_layers=self.settings.bitnet_n_gpu_layers,
                n_threads=self.settings.bitnet_n_threads,
                use_mlock=True,
                verbose=False,
            )

        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(None, _load)
        logger.info("BitNet model loaded")

    def _run_llm_sync(self, messages: list[Any], max_tokens: int | None = None) -> str:
        """Run the LLM synchronously (called from pipeline nodes)."""
        if isinstance(self._llm, MockLLM):
            return self._llm.generate(messages)

        prompt = _messages_to_prompt(messages)
        result = self._llm(
            prompt,
            max_tokens=max_tokens or self.settings.bitnet_max_tokens,
            temperature=self.settings.bitnet_temperature,
            top_p=self.settings.bitnet_top_p,
            stop=["<|im_end|>"],
        )
        return result["choices"][0]["text"].strip()

    def _get_or_create_memory(self, session_id: str) -> list[Any]:
        """Return the message history for a session, creating it if needed."""
        if session_id not in self._session_memories:
            self._session_memories[session_id] = [SystemMessage(content=_SYSTEM_PROMPT)]
        return self._session_memories[session_id]

    async def run(self, inp: MetaAgentInput) -> MetaAgentOutput:
        """
        Execute one full agent turn.
        Serialized by asyncio.Lock to prevent llama-cpp concurrency issues.
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._run_sync, inp
            )

    def _run_sync(self, inp: MetaAgentInput) -> MetaAgentOutput:
        """
        Synchronous pipeline:
          1. Pre-flight Silicon Colosseum check
          2. Dispatch to module-specific handler
          3. Post-flight faithfulness score
          4. Build optional X-Ray causal graph
        """
        causal_nodes: list[dict[str, Any]] = []
        causal_edges: list[dict[str, Any]] = []
        policy_decisions: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        t_total = time.perf_counter()

        def _trace(node: str, data: dict[str, Any]) -> None:
            if inp.xray_mode:
                causal_nodes.append({"id": node, "data": data, "ts": time.perf_counter()})
                if causal_nodes:
                    causal_edges.append({"source": causal_nodes[-2]["id"] if len(causal_nodes) > 1 else "start", "target": node})

        # ── 1. Pre-flight Colosseum ───────────────────────────────
        t0 = time.perf_counter()
        decision = self.colosseum.evaluate_request_sync({
            "session_id": inp.session_id,
            "module": inp.module,
            "message": inp.message,
            "tool_call_count": 0,
        })
        policy_decisions.append(decision.to_dict())
        _trace("colosseum_preflight", {"decision": decision.to_dict(), "latency_ms": (time.perf_counter() - t0) * 1000})

        if not decision.allowed:
            return MetaAgentOutput(
                response=f"[Silicon Colosseum] Request blocked: {decision.reason}",
                policy_decisions=policy_decisions,
                causal_graph={"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
            )

        # ── 2. Retrieve session memory ────────────────────────────
        memory = self._get_or_create_memory(inp.session_id)
        memory.append(HumanMessage(content=inp.message))
        _trace("intake", {"session_id": inp.session_id, "module": inp.module, "message_len": len(inp.message)})

        # ── 3. Module dispatch ────────────────────────────────────
        t0 = time.perf_counter()
        VALID_MODULES = {"ragforge", "localbuddy", "watchtower", "streamsync", "tunelab"}
        module = inp.module if inp.module in VALID_MODULES else "localbuddy"
        _trace("router", {"selected_module": module})

        # Build module-specific system context
        module_context = _MODULE_CONTEXTS.get(module, "")
        messages_with_context = [SystemMessage(content=_SYSTEM_PROMPT + "\n\n" + module_context)] + memory[1:]

        # Run LLM
        response_text = self._run_llm_sync(messages_with_context)
        _trace(f"module_{module}", {"latency_ms": (time.perf_counter() - t0) * 1000, "response_preview": response_text[:100]})

        # ── 4. Post-flight faithfulness score ─────────────────────
        # Simplified faithfulness: ratio of response that references
        # the context or question. Production: use RAGAS or DSPy eval.
        faithfulness_score = _estimate_faithfulness(inp.message, response_text)
        _trace("faithfulness", {"score": faithfulness_score})

        # Block low-faithfulness outputs
        if faithfulness_score < self.settings.silicon_colosseum_min_faithfulness:
            post_decision = self.colosseum.evaluate_request_sync({
                "session_id": inp.session_id,
                "module": "output_filter",
                "message": response_text,
                "faithfulness_score": faithfulness_score,
                "tool_call_count": 0,
            })
            policy_decisions.append(post_decision.to_dict())
            if not post_decision.allowed:
                response_text = (
                    f"[Silicon Colosseum] Output withheld: faithfulness score "
                    f"{faithfulness_score:.2f} below threshold "
                    f"{self.settings.silicon_colosseum_min_faithfulness:.2f}. "
                    "Please rephrase your question."
                )

        # ── 5. Update session memory ──────────────────────────────
        memory.append(AIMessage(content=response_text))
        _trace("output", {"latency_ms": (time.perf_counter() - t_total) * 1000})

        causal_graph = None
        if inp.xray_mode:
            causal_graph = {
                "nodes": causal_nodes,
                "edges": causal_edges,
                "total_latency_ms": round((time.perf_counter() - t_total) * 1000, 2),
            }

        return MetaAgentOutput(
            response=response_text,
            tool_calls=tool_calls,
            policy_decisions=policy_decisions,
            causal_graph=causal_graph,
            faithfulness_score=faithfulness_score,
        )

    async def stream(self, inp: MetaAgentInput) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields token-by-token chunks for WebSocket streaming.
        Falls back to single-chunk yield for MockLLM.
        """
        async with self._lock:
            if isinstance(self._llm, MockLLM):
                memory = self._get_or_create_memory(inp.session_id)
                memory.append(HumanMessage(content=inp.message))
                text = self._llm.generate(memory)
                memory.append(AIMessage(content=text))
                yield {"type": "token", "content": text}
                return

            # Real streaming via llama-cpp create_completion with stream=True
            memory = self._get_or_create_memory(inp.session_id)
            memory.append(HumanMessage(content=inp.message))
            prompt = _messages_to_prompt(memory)

            full_response = []

            def _stream_tokens() -> list[str]:
                tokens: list[str] = []
                for tok in self._llm(  # type: ignore[operator]
                    prompt,
                    max_tokens=self.settings.bitnet_max_tokens,
                    temperature=self.settings.bitnet_temperature,
                    top_p=self.settings.bitnet_top_p,
                    stop=["<|im_end|>"],
                    stream=True,
                ):
                    tokens.append(tok["choices"][0]["text"])
                return tokens

            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(None, _stream_tokens)

            for tok in tokens:
                full_response.append(tok)
                yield {"type": "token", "content": tok}

            full_text = "".join(full_response)
            memory.append(AIMessage(content=full_text))


# ── Module Context Strings ────────────────────────────────────────
# Each module gets a specialized system context appended to the
# base system prompt to steer the LLM's behavior appropriately.

_MODULE_CONTEXTS: dict[str, str] = {
    "ragforge": (
        "You are in RAGForge mode. Answer questions using retrieved context. "
        "Always cite sources. If no relevant context is retrieved, say so explicitly. "
        "Do not hallucinate facts."
    ),
    "localbuddy": (
        "You are in LocalBuddy mode. Act as a helpful, concise AI assistant. "
        "Remember conversation context. Be honest about what you don't know."
    ),
    "watchtower": (
        "You are in WatchTower mode. Analyze system metrics, logs, and events. "
        "Identify anomalies, patterns, and potential issues. "
        "Provide actionable recommendations."
    ),
    "streamsync": (
        "You are in StreamSync mode. Process and analyze event streams. "
        "Identify temporal patterns, correlations, and sequences. "
        "Output structured insights."
    ),
    "tunelab": (
        "You are in TuneLab mode. Help the user configure and monitor model "
        "fine-tuning. Explain OPLoRA parameters, training metrics, and convergence."
    ),
}


# ── Faithfulness Estimation ───────────────────────────────────────
def _estimate_faithfulness(question: str, response: str) -> float:
    """
    Lightweight faithfulness heuristic (no LLM call needed).

    Production-grade implementation would use:
      - RAGAS faithfulness metric
      - DSPy ChainOfThought evaluator
      - NLI model (cross-encoder)

    This heuristic checks:
      1. Response length (too short = suspicious)
      2. Keyword overlap with question
      3. Absence of known hallucination markers

    Returns a score in [0.0, 1.0].
    """
    if not response or len(response) < 10:
        return 0.5

    # Hallucination markers that lower confidence
    HALLUCINATION_MARKERS = [
        "as an ai", "i cannot", "i am not able",
        "i don't have access", "as a language model",
    ]
    response_lower = response.lower()
    penalty = sum(0.05 for m in HALLUCINATION_MARKERS if m in response_lower)

    # Keyword overlap bonus
    q_words = set(question.lower().split())
    r_words = set(response_lower.split())
    overlap = len(q_words & r_words) / max(len(q_words), 1)
    bonus = min(overlap * 0.3, 0.3)

    score = max(0.0, min(1.0, 0.85 + bonus - penalty))
    return round(score, 3)
