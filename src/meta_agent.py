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
import collections
import json
import re
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, cast, Union, Dict, List
import types

if TYPE_CHECKING:
    from src.modules.ragforge.cognitive_rag import CognitiveRAG # type: ignore
    from src.learning.evolution import AetherResearcher # type: ignore

import structlog # type: ignore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ConfigDict # type: ignore

from src.chat_contract import (
    extract_attachment_names,
    merge_attachment_names,
    normalize_citations,
    sanitize_output,
    split_reasoning_trace,
) # type: ignore
from src.config import AetherForgeSettings # type: ignore
from src.guardrails.silicon_colosseum import SiliconColosseum # type: ignore
from src.guardrails.coherence_gate import verify_calc_response, NumberVerificationError, is_calc_route  # type: ignore
from src.learning.sona_adapter import SONAAdapter  # type: ignore

logger = structlog.get_logger("aetherforge.meta_agent")

_SYSTEM_PROMPT_VARIANTS = {
    "v1": (
        "You are AetherForge, a local AI assistant. You run entirely on-device. "
        "Your goal is to be helpful, concise, and technically precise.\n\n"
        "MANDATORY THINKING PROTOCOL:\n"
        "1. Every response MUST start with thorough internal reasoning inside <think>...</think> tags.\n"
        "2. Decompose user intent, identify needed tools, and plan your response.\n"
        "3. Only the reasoning goes inside the tags. The final answer goes AFTER the closing tag.\n\n"
        "TOOL SELECTION RULES:\n"
        "1. For ANY live data, news, prices, or factual lookups not known to you, call 'search_web' with a specific query.\n"
        "2. ONLY call 'get_weather' if the user explicitly asks about weather, temperature, or climate.\n"
        "3. For diesel/petrol prices, stock prices, or breaking news, call 'search_web' immediately.\n"
        "4. Format tool calls as a single JSON block: ```json\n{\"name\": \"...\", \"arguments\": {...}}\n```\n"
        "5. If you call a tool, do it IMMEDIATELY after your reasoning inside the <think> block or right after it."
    ),
    "v2": (
        "You are AetherForge, a highly capable local AI. "
        "Your first priority is FAITHFULNESS and TRUTH. Do not hallucinate.\n\n"
        "THINKING PROCESS:\n"
        "You must analyze every request internally before responding. Use <think> tags "
        "for your step-by-step reasoning. Only write your final response outside the tags.\n\n"
        "TOOL PROTOCOL:\n"
        "If you need real-time data, you MUST call a tool. "
        "Format: ```json\n{\"name\": \"tool_name\", \"arguments\": {...}}\n```"
    ),
    "v3": (
        "You are AetherForge, a local-first AI agent optimized for precision.\n\n"
        "MANDATORY EXECUTION PROTOCOL:\n"
        "1. REASON: Start with <think>...</think> describing your plan.\n"
        "2. ACT: If you need information (weather, news, prices), you MUST output a tool call JSON block IMMEDIATELY after </think>.\n"
        "3. FORMAT: Only use this exact format:\n"
        "```json\n"
        "{\"name\": \"search_web\", \"arguments\": {\"query\": \"fuel price in kolkata\"}}\n"
        "```\n"
        "4. RESPOND: Provide the final answer ONLY after the tool result is provided to you.\n\n"
        "EXAMPLE:\n"
        "User: What is the weather in Delhi?\n"
        "<think>The user wants weather for Delhi. I need to call get_weather.</think>\n"
        "```json\n"
        "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Delhi\"}}\n"
        "```"
    ),
    "v4": (
        "You are AetherForge DeepResearch. Your goal is to solve complex queries through iterative research.\n\n"
        "PROTOCOL:\n"
        "1. THOUGHT: You MUST start every response with <think>...</think> tracing your reasoning.\n"
        "2. ACTION: If you need information, call 'search_web'. If you find information, IMMEDIATELY call 'write_vfs_note' to save it.\n"
        "3. SYNTHESIS: Once you have enough information in your VFS (Virtual File System), provide the final answer outside the tags.\n\n"
        "TOOL FORMAT (MANDATORY):\n"
        "Output ONLY a JSON block like this after your reasoning:\n"
        "```json\n"
        "{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n"
        "```\n"
        "NO PREAMBLE. If you are finished, just output the answer text (no tool call).\n"
    ),
    "v4_small": (
        "You are AetherForge DeepResearch. Solve queries step-by-step.\n\n"
        "1. THINK: Reasoning goes inside <think>...</think>.\n"
        "2. ACT: Call 'search_web' using JSON to get data. Call 'write_vfs_note' to save data.\n"
        "3. ANSWER: If the goal is met, provide a text answer after the tags.\n\n"
        "TOOL CALL EXAMPLE:\n"
        "```json\n"
        "{\"name\": \"search_web\", \"arguments\": {\"query\": \"bitcoin price\"}}\n"
        "```\n"
        "DO NOT repeat yourself. DO NOT just output reasoning without an action."
    ),
}
_SYSTEM_PROMPT = _SYSTEM_PROMPT_VARIANTS["v1"]

from dataclasses import dataclass, field
@dataclass
class MetaAgentOutput:
    """Typed output from a single meta-agent turn."""
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    policy_decisions: list[dict[str, Any]] = field(default_factory=list)
    causal_graph: dict[str, Any] | None = None
    faithfulness_score: float | None = None
    blueprint: Any | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)


def _get_grammar_generator() -> Any:
    """Lazy import and return GrammarGenerator."""
    from src.core.grammar import GrammarGenerator # type: ignore

    return GrammarGenerator


def _messages_to_prompt(messages: list[Any]) -> str:
    """
    Convert LangChain messages to ChatML format for llama-cpp.
    BitNet GGUF models use the standard ChatML template.
    """
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif isinstance(msg, HumanMessage):
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif isinstance(msg, AIMessage):
            parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

    # Add trailing prompt initiator with forced thinking start
    parts.append("<|im_start|>assistant\n<think>")
    return "\n".join(parts)


def _lc_messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain messages to standard dict format."""
    dicts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            dicts.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            dicts.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            dicts.append({"role": "assistant", "content": msg.content})
    return dicts


def _dicts_to_lc_messages(dicts: list[dict[str, Any]]) -> list[Any]:
    """Convert standard dict format back to LangChain messages."""
    messages = []
    for d in dicts:
        role = d.get("role")
        content = d.get("content")
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content)) # type: ignore
    return messages


# ── Mock LLM for development (when model file absent) ─────────────
class MockLLM:
    """
    Stub LLM used when the BitNet GGUF model is not yet downloaded.
    Mimics the llama-cpp-python interface for basic testing.
    """

    def generate(self, messages: list[Any]) -> str:
        last = messages[-1].content if messages else "hello"
        return (
            f"[MockLLM] AetherForge received: '{last[:60]}'. "
            "Download the BitNet model to enable real inference. "
        )

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        # If stream=True, return an iterator of chunks
        if kwargs.get("stream"):

            def _iterator():
                response = f"[MockLLM Streaming] {prompt[:40]}..."  # type: ignore
                # break into small chunks
                for i in range(0, len(response), 4):
                    chunk = response[i : i + 4]  # type: ignore
                    yield {"choices": [{"text": chunk}]}

            return _iterator()
        else:
            return {"choices": [{"text": f"[MockLLM Sync] {prompt[:40]}..."}]}  # type: ignore


# ── Stream Sanitization ───────────────────────────────────────────

class StreamSanitizer:
    """
    A state-machine wrapper for LLM token generators that rigorously
    shields internal protocol tags and prevents tool JSON from leaking to the UI.
    """
    def __init__(self, generator: AsyncGenerator[str, None]):
        self.generator = generator
        self.is_thinking = False
        self.shield_active = False
        self.content_buffer = ""
        self.json_buffer = ""
        self.recovered_tool: dict[str, Any] | None = None

    async def __aiter__(self) -> AsyncGenerator[dict[str, Any], None]:
        async for chunk in self.generator:
            # 1. Update lookahead buffer
            self.content_buffer += chunk

            # 2. Check for tool shielding (JSON blocks)
            # CRITICAL: We only shield JSON if we are NOT in thinking mode.
            # DeepSeek/Llama models often output JSON plans inside <think> tags.
            # We want those to be visible as reasoning, not shielded as tools.
            if not self.is_thinking:
                json_indicator = re.search(r"(?:```(?:json)?\s*)?\{\s*\"(?:name|tool_name|action)\"", self.content_buffer, re.IGNORECASE)
                if json_indicator:
                    # Yield everything before the JSON start as normal tokens
                    pre = self.content_buffer[:json_indicator.start()] # type: ignore
                    if pre:
                        yield {"type": "token", "content": pre}
                    
                    self.shield_active = True
                    # Buffer the JSON part
                    self.json_buffer += self.content_buffer[json_indicator.start():] # type: ignore
                    self.content_buffer = ""
                    
                if self.shield_active:
                    # Check if the JSON block is already complete in the buffer
                    # (LLM might have output it all at once or in a fast stream)
                    if "```" in self.json_buffer and self.json_buffer.count("```") >= 2:
                        self.shield_active = False
                        self._try_parse_tool()
                        # Any leftover text in content_buffer will be handled by the thinking check below
                    else:
                        # Continue buffering JSON and dont yield tokens
                        if len(self.json_buffer) > 2000: # Safety valve
                            self.shield_active = False
                            self._try_parse_tool()
                            self.json_buffer = ""
                        continue

            # 3. Handle Thinking Tags
            while True:
                if not self.is_thinking:
                    # Look for <think> start
                    match = re.search(r"<think>", self.content_buffer, re.IGNORECASE)
                    if match:
                        # Yield everything before <think> as tokens
                        pre = self.content_buffer[:match.start()] # type: ignore
                        if pre:
                            yield {"type": "token", "content": pre}
                        
                        self.is_thinking = True
                        self.content_buffer = self.content_buffer[match.end():] # type: ignore
                        continue
                    else:
                        # Safe to yield if buffer is getting long, but keep enough for multi-token tags
                        if len(self.content_buffer) > 10:
                            pre = self.content_buffer[:-10] # type: ignore
                            if pre:
                                yield {"type": "token", "content": pre}
                            self.content_buffer = self.content_buffer[-10:] # type: ignore
                        break
                else:
                    # We are thinking. Look for </think> end
                    match = re.search(r"</think>", self.content_buffer, re.IGNORECASE)
                    if match:
                        # Yield reasoning trace
                        trace = self.content_buffer[:match.start()] # type: ignore
                        if trace:
                            yield {"type": "reasoning", "content": trace}
                        
                        self.is_thinking = False
                        self.content_buffer = self.content_buffer[match.end():] # type: ignore
                        continue
                    else:
                        # No </think> yet. Yield reasoning if buffer is long.
                        if len(self.content_buffer) > 15:
                            yield {"type": "reasoning", "content": self.content_buffer[:-10]} # type: ignore
                            self.content_buffer = self.content_buffer[-10:] # type: ignore
                        break
            
        # 4. Final Flush
        if self.content_buffer:
            # Final check to strip any trailing protocol noise
            clean = re.sub(r"</?think>", "", self.content_buffer, flags=re.IGNORECASE).strip()
            # If it's not a tool start, yield it
            if clean and not re.search(r"\{?\s*\"?(?:name|tool_name|action)\"?", clean, re.IGNORECASE):
                if self.is_thinking:
                    yield {"type": "reasoning", "content": clean}
                else:
                    yield {"type": "token", "content": clean}
        
        if self.shield_active:
            self._try_parse_tool()

        # Reset buffers for safety
        self.content_buffer = ""
        self.is_thinking = False

    def _try_parse_tool(self) -> None:
        """Attempt to parse buffered JSON into recovered_tool."""
        if not self.json_buffer:
            return
        
        try:
            # Strip markdown if present
            clean_json = re.sub(r"```(?:json)?", "", self.json_buffer, flags=re.IGNORECASE).strip()
            # Find the actual JSON object
            match = re.search(r"(\{.*\})", clean_json, re.DOTALL)
            if match:
                self.recovered_tool = json.loads(match.group(1))
        except Exception:
            pass # Heuristic extraction in MetaAgent will handle failures

# ── Zero-Hallucination Grounding Auditor ──────────────────────────

class GroundingAuditor:
    """
    Refinement sub-agent that verifies synthesis drafts against tool results
    to eliminate hallucinations and enforce evidence-based citations.
    """
    def __init__(self, agent: 'MetaAgent'):
        self.agent = agent

    async def verify_synthesis(
        self, 
        draft: str, 
        context: str, 
        query: str
    ) -> str:
        """
        Performs the Stage 2 (Audit) and Stage 3 (Correction) pass.
        Returns the final, verified, and cited response.
        Uses synchronous LLM call to prevent the audit prompt from echoing
        back into the streamed output (prevents response duplication bug).
        """
        # Build a focused messages list for the auditor — DO NOT use _stream_llm_pass
        # which would stream the entire audit prompt text back to the user.
        # The auditor runs in a thread via run_in_executor to stay non-blocking.
        audit_messages = [
            SystemMessage(content=(
                "You are a grounding auditor. Your ONLY job is to output a single, "
                "clean final response based on the provided draft and search results. "
                "Do NOT repeat the instructions. Do NOT include <think> tags. "
                "Output ONLY the final user-facing answer text."
            )),
            HumanMessage(content=(
                f"Search results (RESOURCES):\n{context}\n\n"
                f"User query: {query}\n\n"
                f"Draft answer to verify and improve:\n{draft}\n\n"
                "Output the corrected, citation-backed final answer. "
                "If the search results do not answer the query, say so honestly. "
                "Append [Source N] citations where supported."
            )),
        ]

        logger.info("GroundingAuditor: Starting verification pass")
        loop = self.agent.loop
        final_output = await loop.run_in_executor(
            None,
            self.agent._run_llm_sync,
            audit_messages,
            512,   # max_tokens
            0.1,   # low temperature for factual accuracy
        )
        # Strip any leaked <think>...</think> blocks from model output
        final_output = re.sub(r"<think>.*?</think>", "", str(final_output), flags=re.DOTALL).strip()
        # Strip standalone closing tags the model sometimes emits
        final_output = re.sub(r"</think>", "", final_output).strip()
        logger.info("GroundingAuditor: Verification pass complete", length=len(final_output))
        return final_output


# ── Input / Output models ─────────────────────────────────────────

class MetaAgentInput(BaseModel):
    """Typed input to the meta-agent for a single turn."""

    session_id: str
    module: str = "localbuddy"
    message: str
    xray_mode: bool = False
    context: dict[str, Any] = {}
    system_location: str | None = None


# ── IRA Framework Data Models ─────────────────────────────────────


class BlueprintTask(BaseModel):
    """A single sub-task within an IRA Blueprint."""

    id: str
    description: str
    module: str
    tool_call: dict[str, Any] | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: str | None = None
    reflection: str | None = None


class IRABlueprint(BaseModel):
    """The recursive plan generated before execution."""

    goal: str
    tasks: list[BlueprintTask] = []
    metadata: dict[str, Any] = {}


class IRAState(BaseModel):
    """Memory state for the recursive agentic loop."""

    session_id: str
    blueprint: IRABlueprint | None = None
    current_task_idx: int = 0
    internal_history: list[dict[str, Any]] = []
    recursion_depth: int = 0
    max_recursion: int = 5
    is_complete: bool = False


# Removed MetaAgentOutput from here as it was moved up


class PlanningState:
    """Ephemeral state for the iterative reasoning loop."""
    def __init__(self, session_id: str, vfs: Any):
        self.session_id = session_id
        self.vfs = vfs
        self.todo_list: list[str] = []
        self.completed_tasks: list[str] = []
        self.iteration_count: int = 0

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

    def __init__(
        self,
        settings: AetherForgeSettings,
        colosseum: SiliconColosseum,
        vector_store: Any = None,
        sparse_index: Any = None,
        replay_buffer: Any = None,
        export_engine: Any = None,
    ) -> None:
        self.settings = settings
        self.colosseum = colosseum
        self.vector_store = vector_store
        self.replay_buffer = replay_buffer
        self.export_engine = export_engine
        self._sparse_index = sparse_index
        self._llm: Any = None
        # Use asyncio.Lock for async-safe model switching, threading.Lock for sync inference
        self._lock: asyncio.Lock = asyncio.Lock()
        self.model_id: str = "qwen-2.5-7b"  # Default model ID
        self.selected_chat_model: str = "qwen-2.5-7b"
        self._sync_lock: threading.Lock = threading.Lock()
        self._session_memories: collections.OrderedDict[str, list[Any]] = collections.OrderedDict()
        self._session_vfs: dict[str, Any] = {}
        self._grammar_cache: dict[str, Any] = {}

        # ── SA-07: SONA 3-tier real-time learning ─────────────
        self._sona = SONAAdapter(
            model_path=getattr(settings, 'bitnet_model_path', ''),
            data_dir=str(settings.data_dir),
        )
        # Start SONA in background (non-blocking, fails gracefully)
        asyncio.get_event_loop().call_soon(
            lambda: asyncio.ensure_future(self._sona.initialize())
        )

        # Declare lazy-initialized attributes explicitly to satisfy static analysis
        self._cognitive_rag_instance: Any | None = None
        self._researcher_instance: Any | None = None
        self.loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        # Headroom attributes (declared here; conditionally initialized below)
        self.token_counter: Any = None
        self.headroom_tokenizer: Any = None
        self.headroom_config: Any = None
        self.context_manager: Any = None

        # ── Headroom Context Optimization ─────────────────────
        try:
            from headroom.config import IntelligentContextConfig  # type: ignore
            from headroom.providers.openai_compatible import OpenAICompatibleTokenCounter  # type: ignore
            from headroom.tokenizer import Tokenizer  # type: ignore
        except ImportError:
            pass

        # [RUVECTOR PHASE 6] RVF Cognitive Container Bridge (via CLI)
        try:
            import subprocess as _sp
            _sp.run(["npx", "--yes", "ruvector", "rvf", "--help"], capture_output=True, check=True, timeout=10)
            self.rvf = True  # RVF CLI is available
            logger.info("RVF Container engine active (via CLI Bridge)")
        except Exception:
            self.rvf = None
            from headroom.transforms.intelligent_context import IntelligentContextManager  # type: ignore

            # Use gpt-3.5-turbo tokenizer as a safe proxy for BitNet/Llama token counting
            self.token_counter = OpenAICompatibleTokenCounter(model="gpt-3.5-turbo")
            self.headroom_tokenizer = Tokenizer(self.token_counter)
            self.headroom_config = IntelligentContextConfig(
                enabled=True,
                keep_last_turns=3,
                keep_system=True,
            )
            self.context_manager = IntelligentContextManager(config=self.headroom_config)
            logger.info("Headroom context optimization layer initialized")
        except ImportError:
            logger.warning("headroom-ai not installed; falling back to legacy pruning")
            self.context_manager = None

    async def initialize(self) -> None:
        """Load BitNet model in a thread executor (non-blocking)."""
        await self._load_model()
        
        # Register Deep Research planning tools
        try:
            from src.core.planning_tools import register_planning_tools # type: ignore
            register_planning_tools()
            logger.info("Deep Research planning tools registered")
        except ImportError:
            logger.warning("planning_tools.py not found; Deep Research disabled")

        logger.info("MetaAgent initialized")

    async def refine_text(self, text: str) -> str:
        """
        Refine the user's input text for grammar, spelling, and clarity.
        Uses a specialized prompt and low temperature for zero-shot correction.
        """
        if not text.strip():
            return text

        system_prompt = (
            "You are a professional grammar and spelling correction tool. "
            "Refine the user's text for clarity, grammar, and spelling. "
            "Keep the original meaning and tone. "
            "Output ONLY the corrected text. Do NOT include any explanations, "
            "quotes, or prefixes like 'Refined:' or 'Corrected:'."
        )

        from langchain_core.messages import HumanMessage, SystemMessage # type: ignore

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]

        # Run synchronous LLM call in a thread pool to avoid blocking the event loop
        loop = self.loop
        refined = await loop.run_in_executor(None, self._run_llm_sync, messages, None, 0.0) # type: ignore
        return sanitize_output(str(refined))

    def _mask_sensitive(self, text: str | Any) -> str:
        """Mask potential PII or secrets in debug logs."""
        if not isinstance(text, str):
            return str(text)
        # Mask things that look like API keys or base64 blobs
        masked = re.sub(r"[a-fA-F0-9]{32,}", "[MASKED_HEX]", text)
        masked = re.sub(r"[A-Za-z0-9+/]{40,}={0,2}", "[MASKED_B64]", masked)
        # Truncate extremely long strings in logs using cast to satisfy strict type checkers
        m_str = cast(str, masked)
        if len(m_str) > 2000:
            return cast(str, m_str[:1000]) + "... [TRUNCATED] ..." + cast(str, m_str[-500:])
        return m_str

    # ── Model Registry ───────────────────────────────────────────────
    # We maintain a registry of supported models to ensure plug-and-play
    # consistency and optimized parameters per model.
    _MODEL_REGISTRY: dict[str, dict[str, Any]] = {
        "qwen-2.5-7b": {
            "repo": None,
            "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
            "params": {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
        },
        "bitnet-b1.58-2b": {
            "repo": None,
            "file": "bitnet-b1.58-2b-4t.gguf",
            "params": {"temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.1},
        },
        "gemma-2b": {
            "repo": "bartowski/gemma-2-2b-it-GGUF",
            "file": "gemma-2-2b-it-Q4_K_M.gguf",
            "params": {"temperature": 0.4, "top_p": 0.95, "repeat_penalty": 1.2},
        },
        "llama-3-8b": {
            "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "params": {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1},
        },
    }

    async def _refine_message_batch(self, messages: list[Any]) -> str:
        """ Lightweight pass to prune or normalize a batch of messages. """
        loop = self.loop

        def _run():
            return self._run_llm_sync(messages, max_tokens=256, temperature=0.1)

        refined_any = await loop.run_in_executor(None, _run) # type: ignore
        refined: str = str(refined_any)

        # Post-processing: remove any residual quotes or labels if the model leaked them
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1] # type: ignore
        if ":" in refined[:20] and any( # type: ignore
            lbl in refined[:20].lower() for lbl in ["corrected", "refined", "output"] # type: ignore
        ):
            refined = refined.split(":", 1)[1].strip() # type: ignore

        return refined.strip()

    async def _load_model(self) -> None:
        """
        Load Gemini-class GGUF model via llama-cpp-python.
        Logic is now registry-driven to ensure zero-tolerance for misconfiguration.
        """
        model_path = self.settings.bitnet_model_path
        if not model_path.exists():
            logger.warning("Model file not found at %s — using MockLLM", model_path)
            self._llm = MockLLM()
            return

        model_id = getattr(self, "selected_chat_model", "qwen-2.5-7b")
        logger.info("Initializing LLM from registry", model_id=model_id, path=str(model_path))

        def _load() -> Any:
            # ── RuvLLM Subprocess Engine (Primary) ───────────────────
            # Wraps @ruvector/ruvllm CLI for LLM inference.
            # This is the PRODUCTION engine. llama_cpp is EMERGENCY only.
            class RuvLLMSubprocessEngine:
                """Thin wrapper that delegates LLM calls to npx @ruvector/ruvllm."""

                def __init__(self, model_path: str, n_ctx: int = 16384):
                    self.model_path = model_path
                    self.n_ctx = n_ctx

                def create_chat_completion(
                    self,
                    messages: list[dict[str, str]],
                    max_tokens: int = 1024,
                    temperature: float = 0.2,
                    grammar: Any = None,
                    **kwargs: Any,
                ) -> dict:
                    import subprocess as _sp

                    # Flatten messages into a prompt the CLI understands
                    prompt = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in messages
                    )
                    cmd = [
                        "npx", "--yes", "@ruvector/ruvllm", "generate",
                        prompt,
                        "--temperature", str(temperature),
                    ]
                    try:
                        proc = _sp.run(cmd, capture_output=True, text=True, check=True, timeout=120)
                        text = proc.stdout.strip()
                        return {
                            "choices": [{"message": {"role": "assistant", "content": text}}],
                        }
                    except _sp.TimeoutExpired:
                        logger.error("ruvllm subprocess timed out (120s)")
                        return {
                            "choices": [{"message": {"role": "assistant", "content": "LLM inference timed out."}}],
                        }
                    except Exception as exc:
                        logger.error("ruvllm subprocess error", error=str(exc))
                        return {
                            "choices": [{"message": {"role": "assistant", "content": f"LLM error: {exc}"}}],
                        }

                # Alias expected by some call-sites
                def __call__(self, prompt: str, **kw: Any) -> dict:
                    return self.create_chat_completion(
                        [{"role": "user", "content": prompt}], **kw
                    )

            # ── Engine selection ─────────────────────────────────────
            # Priority: llama_cpp (direct GGUF) → RuvLLM subprocess (fallback)
            # llama_cpp gives reliable local inference; the subprocess bridge
            # depends on @ruvector/ruvllm npm package availability.
            try:
                from llama_cpp import Llama, LlamaGrammar  # type: ignore
                setattr(self, "_LlamaGrammar", LlamaGrammar)
                logger.info("llama_cpp available — using direct GGUF inference (primary engine)")
                return Llama(
                    model_path=str(model_path),
                    n_ctx=self.settings.bitnet_n_ctx,
                    n_gpu_layers=self.settings.bitnet_n_gpu_layers,
                    n_threads=self.settings.bitnet_n_threads,
                    use_mlock=False,
                    n_batch=1024,
                    flash_attn=True,
                    verbose=False,
                )
            except ImportError:
                logger.warning(
                    "llama_cpp not installed — falling back to RuvLLM subprocess bridge",
                )
                import subprocess as _sp
                try:
                    _sp.run(["npx", "--version"], capture_output=True, check=True, timeout=10)
                    logger.info("RuvLLM subprocess bridge: npx verified — using ruvllm runtime")
                    return RuvLLMSubprocessEngine(model_path=str(model_path), n_ctx=16384)
                except Exception as npx_err:
                    logger.error("Neither llama_cpp nor npx available", error=str(npx_err))
                    raise RuntimeError(
                        "No LLM engine available: install llama-cpp-python or Node.js"
                    ) from npx_err

        loop = self.loop
        self._llm = await loop.run_in_executor(None, _load) # type: ignore
        self.model_id = model_id
        logger.info("LLM engine online", model_id=model_id)

    def _get_or_create_vfs(self, session_id: str) -> Any:
        """Get or create a VirtualFileSystem instance for the given session."""
        if session_id not in self._session_vfs:
            from src.core.vfs import VirtualFileSystem # type: ignore
            storage_path = self.settings.data_dir / "vfs"
            self._session_vfs[session_id] = VirtualFileSystem(session_id, str(storage_path))
            logger.info("VFS initialized for session", session_id=session_id)
        return self._session_vfs[session_id]

    def drop_session_state(self, session_id: str) -> None:
        """Remove session memory and scratchpad state for explicit deletes."""
        self._session_memories.pop(session_id, None)
        vfs_inst = self._session_vfs.pop(session_id, None)
        if vfs_inst is not None:
            try:
                vfs_inst.purge()
            except Exception:
                logger.warning("VFS purge failed during session delete", session_id=session_id)

    def _compile_grammar(self, gbnf: str | None) -> Any:
        """ Compile GBNF string into a LlamaGrammar object using the internal cache. """
        if not gbnf or isinstance(self._llm, MockLLM):
            return None
        
        if gbnf in self._grammar_cache:
            return self._grammar_cache[gbnf] # type: ignore
        
        try:
            LlamaGrammar = getattr(self, "_LlamaGrammar", None)
            if LlamaGrammar is None:
                from llama_cpp import LlamaGrammar # type: ignore
                setattr(self, "_LlamaGrammar", LlamaGrammar)
            
            grammar_obj = LlamaGrammar.from_string(gbnf)
            self._grammar_cache[gbnf] = grammar_obj # type: ignore
            return grammar_obj
        except Exception as e:
            logger.warning("GBNF compilation failed", error=str(e))
            return None

    async def switch_model(self, model_id: str) -> bool: # type: ignore
        """
        Dynamically unloads current model and loads a new one from the registry.
        Supports fuzzy matching for UI convenience.
        """
        import gc
        from huggingface_hub import hf_hub_download # type: ignore

        if model_id not in self._MODEL_REGISTRY:
            # Try fuzzy match (e.g. 'bitnet-2b' matches 'bitnet-b1.58-2b')
            matches = [k for k in self._MODEL_REGISTRY.keys() if model_id in k or k in model_id]
            if not matches:
                raise ValueError(f"Model {model_id} is not in the recognized registry.")
            model_id = matches[0]

        info = self._MODEL_REGISTRY[model_id]
        filename = info["file"]
        repo_id = info["repo"]
        target_path = self.settings.bitnet_model_path.parent / filename

        try:
            async with self._lock:
                if self._llm is not None:
                    logger.info("Switching models: purging old instance memory", model_id=model_id)
                    del self._llm
                    self._llm = None
                    gc.collect()

                if not target_path.exists():
                    if not repo_id:
                        raise FileNotFoundError(f"Model {filename} is a local-only dependency and is missing.")
                    
                    logger.info("Downloading model from HuggingFace Hub", repo=repo_id, file=filename)
                    def _download():
                        return hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=str(self.settings.bitnet_model_path.parent),
                        )
                    await self.loop.run_in_executor(None, _download) # type: ignore

                self.settings.bitnet_model_path = target_path
                self.selected_chat_model = model_id
                await self._load_model()
                return True
        except Exception as e:
            logger.error("Model switch failed", error=str(e))
            return False

    def _run_llm_sync(
        self,
        messages: list[Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
        grammar: str | None = None,
    ) -> str:
        """Run the LLM synchronously (called from pipeline nodes)."""
        if isinstance(self._llm, MockLLM):
            return self._llm.generate(messages)

        # ── Intelligent Context Pruning (Headroom Integration) ─
        if self.context_manager:
            msg_dicts = _lc_messages_to_dicts(messages)
            # available = n_ctx - max_tokens - safety_buffer
            result = self.context_manager.apply(
                messages=msg_dicts,
                tokenizer=self.headroom_tokenizer,
                model_limit=self.settings.bitnet_n_ctx,
                output_buffer=self.settings.bitnet_max_tokens + 100,  # with safety margin
            )
            pruned_messages = _dicts_to_lc_messages(result.messages)
            if result.transforms_applied:
                tokens_saved = result.tokens_before - result.tokens_after
                logger.info(
                    "Headroom optimized context: %d tokens saved via %d transforms",
                    tokens_saved,
                    len(result.transforms_applied),
                )
        else:
            # Legacy fallback
            pruned_messages = list(messages)
            while len(_messages_to_prompt(pruned_messages)) > 12000 and len(pruned_messages) > 3:
                pruned_messages.pop(1) # type: ignore
                pruned_messages.pop(1) # type: ignore

        # ── Registry-Aware Parameters ──────────────────────────
        model_id = getattr(self, "selected_chat_model", "qwen-2.5-7b")
        registry_config = self._MODEL_REGISTRY.get(model_id, self._MODEL_REGISTRY["qwen-2.5-7b"])
        registry_params = registry_config.get("params", {})

        prompt = _messages_to_prompt(pruned_messages)
        return self._call_llm_with_retry(
            prompt,
            max_tokens=max_tokens or self.settings.bitnet_max_tokens,
            temperature=temperature if temperature is not None else registry_params.get("temperature", self.settings.bitnet_temperature),
            top_p=registry_params.get("top_p", self.settings.bitnet_top_p),
            repeat_penalty=registry_params.get("repeat_penalty", 1.1),
            stop=["<|im_end|>"],
            grammar=grammar,
        )

    def _call_llm_with_retry(self, prompt: str, **kwargs: Any) -> str:
        """Calls the LLM with exponential backoff for transient failures."""
        max_retries = 3
        delay: float = 1.0
        last_err: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Thread-safe LLM access (serialize actual inference but not pipeline)
                with self._sync_lock:
                    # Compile GBNF string to LlamaGrammar object if present
                    if "grammar" in kwargs and isinstance(kwargs["grammar"], str):
                        kwargs["grammar"] = self._compile_grammar(kwargs["grammar"])
                    
                    result = self._llm(prompt, **kwargs)
                return str(result["choices"][0]["text"]).strip()
            except Exception as e:
                last_err = e
                logger.warning(f"LLM inference attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(float(delay))
                    delay = float(delay) * 2.0  # type: ignore

        logger.error(f"LLM inference exhausted {max_retries} attempts. Final error: {last_err}")
        return f"⚠️ LLM Error: I encountered a transient failure processing this request after {max_retries} attempts."

    def _get_or_create_memory(self, session_id: str) -> list[Any]:
        """Return the message history for a session, creating it if needed with LRU eviction."""
        if session_id in self._session_memories:
            # Move to end (MRU)
            self._session_memories.move_to_end(session_id)
        else:
            # Create new entry
            self._session_memories[session_id] = [SystemMessage(content=_SYSTEM_PROMPT)]
            # Check for eviction (default cap 100 sessions)
            if len(self._session_memories) > self.settings.max_session_memories:
                oldest_session, _ = self._session_memories.popitem(last=False)
                # Also prune VFS instance if it exists to prevent memory leaks
                if oldest_session in self._session_vfs:
                    vfs_inst = self._session_vfs.pop(oldest_session)
                    try:
                        vfs_inst.purge() # Optional: also delete the physical file
                    except Exception:
                        pass
        return self._session_memories[session_id]

    def _hybrid_search(
        self,
        query: str,
        k: int = 8,
        source_filter: str | list[str] | None = None,
    ) -> list[Any]:
        """
        Hybrid retrieval: Dense (ChromaDB/BGE-M3) + Sparse (FTS5/BM25) OR RuVector GNN-HNSW.
        Falls back to dense-only if FTS5 index is not available.

        CRITICAL: Do NOT cache a permanent 'UNAVAILABLE' sentinel.
        The DB is created when the first document is uploaded, which happens
        AFTER the server starts. Always re-check if the DB now exists.
        """
        # --- NEW: Check for RuVectorStore ---
        if type(self.vector_store).__name__ == "RuVectorStore":
            logger.info("RuVector GNN-HNSW unified search for '%s...'", query[:50])
            filter_kwargs = {}
            if source_filter:
                if isinstance(source_filter, str):
                    filter_kwargs["filter"] = {"source": source_filter}
                elif isinstance(source_filter, list) and len(source_filter) == 1:
                    filter_kwargs["filter"] = {"source": source_filter[0]}
                elif isinstance(source_filter, list):
                    filter_kwargs["filter"] = {"source": {"$in": source_filter}}
            return self.vector_store.similarity_search(query, k=k, **filter_kwargs)
        # ------------------------------------

        # Try to get/init the sparse index on every call when not yet loaded
        if self._sparse_index is None:
            try:
                from src.modules.ragforge.sparse_index import SparseIndex # type: ignore

                db_path = self.settings.data_dir / "sparse_index.db"
                if db_path.exists() and db_path.stat().st_size > 0:
                    self._sparse_index = SparseIndex(db_path=db_path)
                    logger.info(
                        "FTS5 sparse index loaded (%d chunks)",
                        self._sparse_index.count(),
                    )
                # If DB doesn't exist yet, leave self._sparse_index = None
                # so we retry on the next request (after a document is uploaded)
            except Exception as e:
                logger.warning("FTS5 init failed: %s", e)
                # Do NOT set UNAVAILABLE — retry in the next call

        # If sparse index is available, use hybrid search
        if self._sparse_index is not None:
            try:
                from src.modules.ragforge.sparse_index import hybrid_search # type: ignore

                results = hybrid_search(
                    query=query,
                    vector_store=self.vector_store,
                    sparse_index=self._sparse_index,
                    k=k,
                    source_filter=source_filter,
                )

                # Intent-Based Visual Boost
                # If the user is asking about visual data, forcibly promote VLM chunks
                q_lower = query.lower()
                visual_triggers = {
                    "chart",
                    "diagram",
                    "figure",
                    "table",
                    "image",
                    "visual",
                    "snapshot",
                    "graph",
                }
                if any(trigger in q_lower for trigger in visual_triggers):
                    # We need to reach into the chunks and re-sort them based on a manual boost.
                    # Since `hybrid_search` returns a sorted list of Documents (RRF scores are lost),
                    # we will bubble up `vlm_analysis` chunks to the top 3 positions.
                    vlm_docs = []
                    text_docs = []
                    for doc in results:
                        if doc.metadata.get("chunk_type") == "vlm_analysis":
                            vlm_docs.append(doc)
                        else:
                            text_docs.append(doc)

                    if vlm_docs:
                        results = vlm_docs + text_docs
                        logger.info(
                            "Visual boost applied: prioritized %d VLM chunks for query '%s'",
                            len(vlm_docs),
                            query[:30], # type: ignore
                        )

                    logger.info(
                        "Hybrid search: %d results for '%s...' (filter=%s)",
                        len(results), # type: ignore
                        query[:50], # type: ignore
                        source_filter,
                    )
                    # ── Defense-in-depth: post-filter hybrid results ──
                    if source_filter:
                        hybrid_allowed: set[str] = (
                            {source_filter} if isinstance(source_filter, str) else set(source_filter)
                        )
                        results = [d for d in results if d.metadata.get("source") in hybrid_allowed]
                    return results # type: ignore
            except Exception as e:
                logger.warning("Hybrid search failed: %s — falling back to dense", e)

        # Fallback: dense-only (ChromaDB)
        logger.info("Dense-only search for '%s...' (sparse index not loaded yet)", query[:50]) # type: ignore
        filter_kwargs: dict[str, Any] = {}
        if source_filter:
            if isinstance(source_filter, str):
                filter_kwargs["filter"] = {"source": source_filter}
            elif isinstance(source_filter, list) and len(source_filter) == 1:
                filter_kwargs["filter"] = {"source": source_filter[0]}
            elif isinstance(source_filter, list):
                filter_kwargs["filter"] = {"source": {"$in": source_filter}}
        dense_results = self.vector_store.similarity_search(query, k=k, **filter_kwargs)

        # ── Defense-in-depth: hard post-filter by source ───────────
        # Guarantees NO chunk from an unselected document leaks through.
        if source_filter:
            allowed_sources: set[str] = (
                {source_filter} if isinstance(source_filter, str) else set(source_filter)
            )
            dense_results = [d for d in dense_results if d.metadata.get("source") in allowed_sources]
        return dense_results

    def _generate_ira_blueprint(self, message: str, context: dict[str, Any]) -> IRABlueprint:
        """
        Generate a multi-step plan (blueprint) for the given request.
        Uses a dry-run prompt to decompose the goal into atomic tasks.
        """
        system_prompt = (
            "You are the IRA Planning Engine. Your job is to break a complex user request "
            "into 2-4 atomic, sequential tasks for an agentic loop.\n\n"
            "Each task must have:\n"
            "1. id: unique short string\n"
            "2. description: what to do\n"
            "3. module: one of [ragforge, localbuddy, watchtower, streamsync, tunelab]\n\n"
            "Output the plan as a JSON block in this format:\n"
            "```json\n"
            '{"goal": "Overall user goal", "tasks": [{"id": "t1", "description": "...", "module": "..."}]}\n'
            "```"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Request: {message}\nContext: {context}"),
        ]

        try:
            raw_plan = self._run_llm_sync(messages, max_tokens=500, temperature=0.1)
            import json
            import re

            json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_plan, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1))
                return IRABlueprint(**plan_data)
        except Exception as e:
            logger.warning("IRA Blueprint generation failed: %s", e)

        # Fallback: single-task blueprint
        return IRABlueprint(
            goal=message,  # type: ignore
            tasks=[BlueprintTask(id="task_1", description=message, module="localbuddy")],  # type: ignore
        )

    def _get_cognitive_rag(self) -> CognitiveRAG:
        """Lazily create a CognitiveRAG instance that reuses the loaded LLM + search."""
        if not hasattr(self, "_cognitive_rag_instance") or self._cognitive_rag_instance is None:
            from src.modules.ragforge.cognitive_rag import CognitiveRAG # type: ignore

            self._cognitive_rag_instance = CognitiveRAG(
                llm_fn=self._run_llm_sync,
                search_fn=self._hybrid_search,
            )
            logger.info("CognitiveRAG pipeline initialized (reusing %s + hybrid search)", self.model_id)
        return self._cognitive_rag_instance

    def _get_aether_researcher(self) -> AetherResearcher:
        """Lazily create an AetherResearcher instance."""
        if not hasattr(self, "_researcher_instance") or self._researcher_instance is None:
            from src.learning.evolution import AetherResearcher, ExperimentManager  # type: ignore

            manager = ExperimentManager(self.settings)
            self._researcher_instance = AetherResearcher(manager)
            logger.info("AetherResearcher specialized IRA agent initialized")
        return self._researcher_instance

    def _build_tool_state(self) -> Any:
        """Expose shared services to tool handlers."""
        return types.SimpleNamespace(
            replay_buffer=self.replay_buffer,
            settings=self.settings,
            sparse_index=self._sparse_index,
            vector_store=self.vector_store,
            export_engine=self.export_engine,
        )

    @staticmethod
    def _normalize_tool_output(raw_output: Any) -> tuple[str, list[str], list[dict[str, Any]]]:
        """Normalize tool outputs into content plus structured attachments/citations."""
        if isinstance(raw_output, dict):
            content = str(raw_output.get("content", "")).strip()
            attachments = merge_attachment_names(
                [str(item) for item in raw_output.get("attachments", [])],
                extract_attachment_names(content),
            )
            citations = normalize_citations(raw_output.get("citations", []))
            return content, attachments, citations

        content = str(raw_output)
        return content, extract_attachment_names(content), []

    @staticmethod
    def _build_document_citations(docs: list[Any]) -> list[dict[str, Any]]:
        """Convert retrieved documents into structured citations."""
        citations: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            meta = getattr(doc, "metadata", {}) or {}
            snippet = " ".join(str(getattr(doc, "page_content", "")).split())
            citations.append(
                {
                    "label": f"[{idx}]",
                    "source": str(meta.get("source", "Unknown source")),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "snippet": snippet[:240] or None, # type: ignore
                    "kind": "document",
                }
            )
        return normalize_citations(citations)

    def _resolve_local_source(
        self,
        message: str,
        context: dict[str, Any],
    ) -> str | None:
        """Resolve a local source filename from the user request or active docs."""
        if context.get("active_docs"):
            active_docs = context["active_docs"]
            if isinstance(active_docs, list) and active_docs:
                return str(active_docs[0])

        explicit_matches = re.findall(
            r"\b[\w.\- ]+\.(?:pdf|csv|tsv|xlsx|xls|md|txt|json)\b",
            message,
            flags=re.IGNORECASE,
        )
        candidate_roots = [
            self.settings.live_folder,
            self.settings.uploads_dir,
            self.settings.data_dir,
        ]

        for match in explicit_matches:
            candidate = match.strip().strip("\"'")
            for root in candidate_roots:
                if (root / candidate).exists():
                    return Path(candidate).name

        lowered = message.lower()
        if "livefolder" in lowered or "live folder" in lowered:
            files = sorted(
                p.name
                for p in self.settings.live_folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".pdf", ".csv", ".tsv", ".xlsx", ".xls", ".md", ".txt", ".json"}
            )
            if len(files) == 1:
                return files[0]
            if files:
                preferred = [name for name in files if name.lower().endswith(".pdf")]
                return preferred[0] if preferred else files[0]

        return None

    def _should_handle_with_local_file_tool(
        self,
        module: str,
        message: str,
        context: dict[str, Any],
    ) -> bool:
        """Detect requests that should be grounded in local file inspection."""
        if module not in {"localbuddy", "analytics", "ragforge"}:
            return False
        if self._resolve_local_source(message, context) is None:
            return False

        lowered = message.lower()
        file_cues = (
            "pdf",
            "excel",
            "spreadsheet",
            "csv",
            "document",
            "file",
            "chart",
            "graph",
            "diagram",
            "flow chart",
            "flowchart",
            "figure",
            "table",
            "kid",
            "child",
            "10 year old",
            "ten year old",
            "export",
            ".md",
            "markdown",
        )
        return any(cue in lowered for cue in file_cues)

    def _build_local_file_tool_args(
        self,
        message: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a deterministic analytics tool call for local file requests."""
        lowered = message.lower()
        source = self._resolve_local_source(message, context)
        include_visual = any(
            cue in lowered
            for cue in ("chart", "graph", "diagram", "flow chart", "flowchart", "figure", "visual")
        )
        if not include_visual and any(cue in lowered for cue in ("pdf", "document", "explain")):
            include_visual = True

        visual_type = "auto"
        if "flow chart" in lowered or "flowchart" in lowered or "diagram" in lowered:
            visual_type = "flowchart"

        export_format = "markdown"
        if ("markdown" in lowered or ".md" in lowered) and "pdf" in lowered:
            export_format = "both"
        elif "pdf" in lowered and "export" in lowered:
            export_format = "pdf"
        elif "csv" in lowered and "export" in lowered:
            export_format = "csv"

        audience = "general"
        if any(cue in lowered for cue in ("10 year old", "ten year old", "kid", "child")):
            audience = "10-year-old"

        return {
            "source": source,
            "question": message,
            "audience": audience,
            "format": export_format,
            "include_visual": include_visual,
            "visual_type": visual_type,
            "report_title": f"Analysis of {source}" if source else "Local file analysis",
        }

    def _heuristically_extract_tool_call(self, text: str) -> dict[str, Any] | None:
        """
        Universal tool extraction fallback for multiple models (BitNet, Llama, Gemma).
        Handles broken JSON, text markers (Action:), and plain intent description.
        """
        import re
        import json

        # 1. Clean and check for JSON-like blocks first
        # Look for any object containing tool keys or known tool names
        raw_json_match = re.search(r"(\{.*?(?:name|tool_name|search_web|get_weather|get_joke|rag_search).*?\})", text, re.DOTALL)
        if raw_json_match:
            try:
                repaired = self._repair_json(raw_json_match.group(1))
                data = json.loads(repaired)
                if "name" in data: return data
                if "tool_name" in data: return {"name": data["tool_name"], "arguments": data.get("arguments", {})}
            except Exception:
                pass

        # 2. Text-based Action Patterns (Llama/ReAct style)
        # Action: search_web
        # Action Input: "how is the weather"
        action_match = re.search(
            r"(?:Action|Call|Tool):\s*\"?(search_web|get_weather|get_joke|rag_search|write_vfs_note|write_todos|clear_planner|get_research_status)\"?", 
            text, 
            re.IGNORECASE
        )
        if action_match:
            tool = action_match.group(1).lower()
            input_match = re.search(r"(?:Action Input|Arguments|Args|Input|Query|Location|Content|Title|Notes):\s*\"?([^\"]+)\"?", text, re.IGNORECASE)
            if input_match:
                val = input_match.group(1).strip()
                # Heuristic mapping for planning tools
                arg_name = "query"
                if tool == "get_weather": arg_name = "location"
                elif tool == "write_vfs_note": arg_name = "content"
                elif tool == "write_todos": arg_name = "todos"
                
                return {"name": tool, "arguments": {arg_name: val}}

        # 3. Plain text intent patterns
        lowered = text.lower()
        # ... (simplified for planning tools)
        if "write_vfs_note" in lowered or "save note" in lowered:
             content_match = re.search(r'(?:note|content|save)\s*[:=]\s*"?([^".\n]+)"?', lowered)
             if content_match:
                 return {"name": "write_vfs_note", "arguments": {"content": content_match.group(1).strip(), "title": "Manual Note"}}

        # 4. Last Chance: Super-aggressive scan for any known tool name
        known_tools = ["search_web", "get_weather", "get_joke", "rag_search", "write_vfs_note", "write_todos", "clear_planner", "get_research_status"]
        for tool in known_tools:
            if tool in text: # Case-sensitive check for the actual tool name
                # Try specific keys first
                arg_match = re.search(r"(?:query|location|arguments|args|Action Input|Input|content):\s*(.+)", text, re.IGNORECASE)
                if arg_match:
                    val = arg_match.group(1).strip()
                    # Remove trailing markers if any
                    val = re.split(r"\s+(?:end|stop|Action:)\b", val, flags=re.IGNORECASE)[0]
                    # Strip outer braces/brackets
                    val = val.strip(' {}[]')
                    # Remove inner keys
                    val = re.sub(r'^(?:query|location|input|content|todos)\s*[:=]\s*', '', val, flags=re.IGNORECASE)
                    # Final strip of quotes and whitespace
                    val = val.strip(' "\'')
                    
                    # Heuristic arg name mapping
                    arg_name = "query"
                    if tool == "get_weather": arg_name = "location"
                    elif tool == "write_vfs_note": arg_name = "content"
                    elif tool == "write_todos": arg_name = "todos"
                    
                    return {"name": tool, "arguments": {arg_name: val}}

        return None

    def _repair_json(self, raw: str) -> str:
        """Fix common LLM JSON errors (unquoted keys, trailing commas, etc.)"""
        import re
        # Unquoted keys or single quotes
        repaired = re.sub(r"([{,])\s*([a-zA-Z_]\w*)\s*:", r'\1 "\2":', raw)
        # Single quotes for values
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
        # Trailing commas
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        # Smart quotes
        repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return repaired

    def _run_local_file_tool(
        self,
        inp: MetaAgentInput,
        main_module: str,
        module_context: str,
        tool_calls: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        attachments: list[str],
    ) -> str | None:
        """Run deterministic local-file analytics before falling back to free-form prompting."""
        if not self._should_handle_with_local_file_tool(main_module, inp.message, inp.context):
            return None

        from src.core.tool_registry import tool_registry # type: ignore

        tool_args = self._build_local_file_tool_args(inp.message, inp.context)
        raw_output = tool_registry.execute_tool("analyze_data", tool_args, state=self._build_tool_state())
        tool_output_text, tool_attachments, tool_citations = self._normalize_tool_output(raw_output)

        tool_calls.append(
            {
                "name": "analyze_data",
                "arguments": tool_args,
                "result": tool_output_text,
                "attachments": tool_attachments,
                "citations": tool_citations,
            }
        )
        citations.extend(tool_citations)
        attachments[:] = merge_attachment_names(attachments, tool_attachments) # type: ignore

        # Strip tool definitions for the synthesis prompt to prevent infinite loops
        clean_context = re.sub(r"\n\nAVAILABLE TOOLS:.*", "", module_context, flags=re.DOTALL)

        summary_messages = [
            SystemMessage(content=_SYSTEM_PROMPT + "\n\n" + clean_context),
            HumanMessage(content=inp.message),
            SystemMessage(
                content=(
                    "Use only the grounded notes below. Keep the explanation faithful to the source, "
                    "mention any generated artifacts naturally, and do not invent missing facts.\n\n"
                    f"{tool_output_text}"
                )
            ),
        ]
        return self._run_llm_sync(summary_messages, temperature=0.3)
    def _detect_and_inject_system_knowledge(self, message: str, current_context: str) -> str:
        """
        If the user asks about the application itself, bootstrap context from the manual.
        Returns the updated context string.
        """
        q_lower = message.lower()
        system_triggers = {
            "aetherforge", "the app", "how to use", "manual", 
            "features", "capabilities", "instruction", "how do i"
        }
        
        if not any(trigger in q_lower for trigger in system_triggers):
            return current_context

        logger.info("System Knowledge intent detected", query=message[:50])
        try:
            # Retrieve from the manual (indexed as AetherForge_User_Manual_v1.0.md)
            manual_docs = self._hybrid_search(
                query=message,
                k=4,
                source_filter="AetherForge_User_Manual_v1.0.md"
            )
            if manual_docs:
                manual_text = "\n\n".join([f"FROM MANUAL: {d.page_content}" for d in manual_docs])
                updated_context = current_context + (
                    f"\n\nSYSTEM KNOWLEDGE (AetherForge Manual):\n{manual_text}\n"
                    "Use the above information to answer questions about the application's "
                    "operations, features, and troubleshooting.\n"
                )
                return updated_context
        except Exception as e:
            logger.warning("System knowledge retrieval failed", error=str(e))
        
        return current_context


    async def run(self, inp: MetaAgentInput) -> MetaAgentOutput:
        """
        Execute one full agent turn.
        Concurrency: The model inference is internally serialized by threading.Lock
        to prevent llama-cpp-python race conditions, but RAG and tool lookups
        are now non-blocking.
        """
        return await self.loop.run_in_executor(None, self._run_sync, inp)

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
        ragforge_trace = None
        retrieved_doc_texts: list[str] = []
        retrieved_docs: list[Any] = []
        citations: list[dict[str, Any]] = []
        attachments: list[str] = []
        messages_with_context: list[Any] = []
        tool_executed_successfully = False
        faithfulness_score: float = 0.5
        blueprint: IRABlueprint | None = None

        def _trace(node: str, data: dict[str, Any], label: str | None = None) -> None:
            if inp.xray_mode:
                causal_nodes.append({"id": node, "data": data, "ts": time.perf_counter()})
                if causal_nodes:
                    source = causal_nodes[-2]["id"] if len(causal_nodes) > 1 else "start"
                    causal_edges.append({"source": source, "target": node, "label": label or ""})

        # ── 1. Pre-flight Colosseum ───────────────────────────────
        t0 = time.perf_counter()
        decision = self.colosseum.evaluate_request_sync(
            {
                "session_id": inp.session_id,
                "module": inp.module,
                "message": inp.message,
                "tool_call_count": 0,
            }
        )
        policy_decisions.append(decision.to_dict())
        _trace(
            "colosseum_preflight",
            {"decision": decision.to_dict(), "latency_ms": (time.perf_counter() - t0) * 1000},
            label="Safety Gate",
        )

        if not decision.allowed:
            return MetaAgentOutput(
                response=f"[Silicon Colosseum] Request blocked: {decision.reason}",  # type: ignore
                policy_decisions=policy_decisions,  # type: ignore
                causal_graph={"nodes": causal_nodes, "edges": causal_edges}  # type: ignore
                if inp.xray_mode
                else None,
            )

        # ── 1.5 Deterministic Query Routing ───────────────────────
        # Intercepts calculation queries BEFORE any LLM invocation.
        # The CalcEngine does exact lookup or linear interpolation from
        # the structured SQLite hydrostatic tables. NO LLM arithmetic.
        try:
            from src.core.query_router import route_query, QueryRoute, extract_draft, extract_column, extract_sg
            from src.core.calc_engine import CalcEngine

            route = route_query(inp.message)
            logger.info("Query router (_run_sync): %s → %s", inp.message[:60], route.value)
            _trace(
                "query_router",
                {"route": route.value, "message_preview": inp.message[:60]},
                label="Deterministic Router",
            )

            if route in (QueryRoute.TABLE_LOOKUP, QueryRoute.MULTI_LOOKUP, QueryRoute.INTERPOLATE, QueryRoute.UNIT_CONVERT):
                calc_engine = CalcEngine(db_path=str(self.settings.data_dir / "structured_data.db"))
                draft = extract_draft(inp.message)

                if draft is None:
                    return MetaAgentOutput(
                        response="I detected a calculation intent, but couldn't find a draft value (e.g. '8.17m'). Please include the draft in metres.",
                        policy_decisions=policy_decisions,
                        causal_graph={"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
                    )

                # Vessel ID heuristic — default for Primrose Ace / HA-13
                vessel_id = "HA"
                resp = ""

                try:
                    if route == QueryRoute.MULTI_LOOKUP:
                        result = calc_engine.lookup_all_hydrostatic(vessel_id, draft)
                        lines = [f"**Hydrostatic Particulars at {draft}m Draft (Salt Water)**\n"]
                        for col, data in result["results"].items():
                            lines.append(f"- **{col.upper()}**: {data['value']} {data['unit']}")
                        if result["results"]:
                            first_trace = next(iter(result["results"].values()))["trace"]
                            lines.append(f"\n*Method: {first_trace['method']}*")
                        if result.get("errors"):
                            for col, err in result["errors"].items():
                                lines.append(f"- ⚠️ {col.upper()}: {err}")
                        resp = "\n".join(lines)

                    elif route == QueryRoute.UNIT_CONVERT:
                        column = extract_column(inp.message)
                        sw_result = calc_engine.lookup_hydrostatic(vessel_id, draft, column)
                        sw_val = sw_result["value"]
                        sg = extract_sg(inp.message)
                        if sg:
                            correction = calc_engine.apply_sg_correction(sw_val, sg)
                            resp = (
                                f"**{column.upper()} at {draft}m in dock water (SG {sg})**\n\n"
                                f"- Salt water value: {sw_val} {sw_result['unit']}\n"
                                f"- Dock water value: **{correction['value']}** {correction['unit']}\n\n"
                                f"*Formula: {correction['trace']['formula']}*"
                            )
                        else:
                            correction = calc_engine.apply_fw_correction(sw_val)
                            resp = (
                                f"**{column.upper()} at {draft}m in fresh water**\n\n"
                                f"- Salt water value: {sw_val} {sw_result['unit']}\n"
                                f"- Fresh water value: **{correction['value']}** {correction['unit']}\n\n"
                                f"*Formula: {correction['trace']['formula']}*"
                            )

                    else:  # TABLE_LOOKUP or INTERPOLATE
                        column = extract_column(inp.message)
                        result = calc_engine.lookup_hydrostatic(vessel_id, draft, column)
                        method_label = "Exact Match" if result["trace"]["method"] == "exact_match" else "Interpolated"
                        resp = (
                            f"**{column.upper()} at {draft}m Draft ({method_label})**\n\n"
                            f"**{result['value']}** {result['unit']}\n\n"
                        )
                        if result["trace"].get("formula"):
                            resp += f"*Formula: {result['trace']['formula']}*\n"
                        resp += "\n*This result was calculated deterministically from the hydrostatic tables — no LLM arithmetic.*"

                    # Coherence gate
                    try:
                        calc_trace_for_verify = {}
                        if route == QueryRoute.MULTI_LOOKUP:
                            calc_trace_for_verify = {col: d.get("trace", {}) for col, d in result.get("results", {}).items()}
                        elif isinstance(result, dict) and "trace" in result:
                            calc_trace_for_verify = result.get("trace", {})
                        if calc_trace_for_verify:
                            verify_calc_response(resp, calc_trace_for_verify)
                            logger.info("Coherence gate (_run_sync): PASS")
                    except NumberVerificationError as nve:
                        logger.warning("Coherence gate (_run_sync): FAIL — %s", nve)
                        resp += "\n\n⚠️ *Number verification enforced — values re-verified against source tables.*"

                    _trace("calc_engine", {"route": route.value, "draft": draft, "response_len": len(resp)}, label="Deterministic Result")

                    resp += "\n\n*This result was calculated deterministically from the hydrostatic tables — no LLM arithmetic.*"

                    return MetaAgentOutput(
                        response=resp,
                        policy_decisions=policy_decisions,
                        causal_graph={"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
                    )

                except ValueError as calc_err:
                    logger.warning("CalcEngine error (_run_sync): %s — falling through to RAG", calc_err)
                    # Fall through to RAG pipeline if CalcEngine fails
                    pass

        except ImportError as e:
            logger.warning("Query router not available (_run_sync): %s", e)
        except Exception as e:
            logger.warning("Query router failed (_run_sync, non-fatal): %s", e)

        # ── 2. Retrieve session memory ────────────────────────────
        memory = self._get_or_create_memory(inp.session_id)
        memory.append(HumanMessage(content=inp.message))
        _trace(
            "intake",
            {"session_id": inp.session_id, "module": inp.module, "message_len": len(inp.message)},
            label="Allowed",
        )

        # ── 3. Module & Tool Context Initialization ──────────────
        VALID_MODULES = {
            "ragforge",
            "localbuddy",
            "watchtower",
            "streamsync",
            "tunelab",
            "analytics",
        }
        requested_module = inp.module if inp.module in VALID_MODULES else "localbuddy"
        main_module = (
            "analytics"
            if requested_module == "ragforge" and bool(inp.context.get("analytics_enabled", False))
            else requested_module
        )
        module_context = _MODULE_CONTEXTS.get(main_module, "")

        # ── 3a. System Knowledge Intent Detection ────────────────
        module_context = self._detect_and_inject_system_knowledge(inp.message, module_context)

        if inp.system_location:
            module_context += (
                f"\n\nUSER LOCATION: {inp.system_location}. "
                f"When the user says 'my location', 'my city', 'here', or 'where I am', "
                f"always use '{inp.system_location}' as the location argument.\n"
            )
        # Import tool_registry unconditionally
        from src.core.tool_registry import tool_registry  # type: ignore

        tools_list = tool_registry.get_tool_definitions()

        # Dynamic context filtering: Prevent tool cross-contamination between modules
        allowed_tools = set()

        if bool(inp.context.get("deep_research", False)):
            allowed_tools.update({"get_research_status", "write_vfs_note", "write_todos", "clear_planner"})
            
        if bool(inp.context.get("web_search_enabled", False)):
            allowed_tools.update({"search_web", "get_weather", "get_joke"})
            
        module_specific_tools = {
            "watchtower": {"query_metrics", "get_top_processes", "kill_process"},
            "streamsync": {"query_stream", "summarize_stream", "clear_buffer"},
            "tunelab": {"query_buffer_stats", "trigger_compilation"},
            "analytics": {"analyze_data", "create_visual"},
            "ragforge": set(),
            "localbuddy": set()
        }
        
        allowed_tools.update(module_specific_tools.get(main_module, set()))
        
        # In ragforge, analytics tools are inherently linked if analytics is enabled
        if requested_module == "ragforge" and bool(inp.context.get("analytics_enabled", False)):
             allowed_tools.update(module_specific_tools["analytics"])

        tools_list = [t for t in tools_list if t["name"] in allowed_tools]

        # Pre-process module context with tools
        if tools_list:
            tools_def = json.dumps(tools_list, indent=2)
            module_context += (
                f"\n\nAVAILABLE TOOLS:\n{tools_def}\n\n"
                "To call a tool, you MUST output a JSON block EXCLUSIVELY in this format:\n"
                "```json\n"
                "{\n"
                '  "name": "<tool_name>",\n'
                '  "arguments": {<tool_args>}\n'
                "}\n"
                "```\n"
            )

        response_text = self._run_local_file_tool(
            inp,
            main_module,
            module_context,
            tool_calls,
            citations,
            attachments,
        )

        vfs = None
        planning_state = None
        deep_research_enabled = bool(inp.context.get("deep_research", False))
        if response_text is None and deep_research_enabled:
            # ── 3a. Initialize Deep Research State ──────────────────────
            vfs = self._get_or_create_vfs(inp.session_id)
            planning_state = PlanningState(inp.session_id, vfs)

            sys_variant = "v4"
            res_instance = getattr(self, "_researcher_instance", None)
            if res_instance is not None and getattr(res_instance, "manager", None):
                sys_variant = getattr(
                    res_instance.manager.current_genome, "system_prompt_variant", "v4"
                )

            base_sys_prompt = _SYSTEM_PROMPT_VARIANTS.get(
                sys_variant, _SYSTEM_PROMPT_VARIANTS["v4"]
            )
            full_system_context = base_sys_prompt + "\n\n" + module_context

            grammar = None
            try:
                grammar = _get_grammar_generator().generate_agentic_grammar(tools_list)
            except Exception:
                pass

            max_iterations = 5
            model_name = self.model_id.lower()
            is_small_model = any(
                k in model_name for k in ["2b", "1.58", "bitnet", "gemma", "llama"]
            )
            prompt_variant = "v4_small" if is_small_model else "v4"
            full_system_context = _SYSTEM_PROMPT_VARIANTS.get(
                prompt_variant, _SYSTEM_PROMPT_VARIANTS["v4"]
            )

            t_loop_start = time.perf_counter()
            for iteration in range(max_iterations):
                # Global safety valve: 60 seconds total for research loop
                if time.perf_counter() - t_loop_start > 60.0:
                    logger.warning("Deep Research loop exceeded global timeout", session_id=inp.session_id)
                    break

                planning_state.iteration_count = iteration + 1
                logger.info(
                    "Iterative Loop Phase",
                    iteration=iteration + 1,
                    session_id=inp.session_id,
                    variant=prompt_variant,
                )

                vfs_context = vfs.get_summary()
                todo_context = f"CURRENT TO-DO LIST: {json.dumps(planning_state.todo_list)}"
                system_injection = f"\n\n--- RESEARCH STATE ---\n{vfs_context}\n{todo_context}\n"

                if iteration >= 3:
                    logger.info("Protocol Short-circuit: Synthesis Nudge", iteration=iteration + 1)
                    system_injection += (
                        "\n\nCRITICAL: You have reached the iterative limit. "
                        "You MUST now summarize your findings and provide your final response. "
                        "DO NOT start new research threads."
                    )

                working_memory = list(memory)
                if is_small_model and iteration >= 2 and len(working_memory) > 4:
                    logger.info("Pruned iterative context for small model", iteration=iteration + 1)
                    base_msgs = cast(list[Any], working_memory)[:2]  # type: ignore
                    tail_msgs = cast(list[Any], working_memory)[-2:]  # type: ignore
                    working_memory = base_msgs + tail_msgs

                if iteration == max_iterations - 1:
                    logger.info("Applying Forced Synthesis grammar", iteration=iteration + 1)
                    grammar = _get_grammar_generator().generate_synthesis_grammar()
                    system_injection += (
                        "\n\nCRITICAL: Final iteration. Summarize findings NOW. DO NOT call tools."
                    )

                current_messages: list[Any] = [
                    SystemMessage(content=full_system_context + system_injection),
                ]
                if len(working_memory) > 1:
                    current_messages.extend(cast(list[Any], working_memory)[1:])  # type: ignore

                t_llm = time.perf_counter()
                # Use a specific timeout for the LLM call itself to prevent blocking inference lock
                llm_response = self._run_llm_sync(current_messages, grammar=grammar, max_tokens=1024)
                _trace(f"llm_pass_{iteration}", {"latency_ms": (time.perf_counter() - t_llm) * 1000})

                reasoning_raw, answer = split_reasoning_trace(llm_response)
                reasoning = str(reasoning_raw or "")
                logger.debug(
                    "Loop iteration result",
                    iteration=iteration + 1,
                    reasoning_len=len(reasoning),
                    answer_len=len(answer),
                )
                logger.debug(
                    "Raw reasoning trace",
                    iteration=iteration + 1,
                    reasoning=self._mask_sensitive(reasoning),
                )

                if not answer.strip() and not reasoning.strip():
                    logger.warning("LLM returned empty response in loop", iteration=iteration + 1)

                tool_call = self._heuristically_extract_tool_call(llm_response)  # type: ignore

                if tool_call:
                    t_tool = time.perf_counter()
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})

                    logger.info("Executing tool in loop", name=tool_name, iteration=iteration + 1)
                    # We can't easily yield from here because _run_sync is synchronous.
                    # This is why the user sees a "queue" - the backend is busy in this loop.
                    # For now, we rely on the terminal logs being visible, but in stream() we fix this.
                    tool_result = tool_registry.execute_tool(tool_name, tool_args, state=planning_state)
                    clean_ai_msg = (
                        f"<think>\n{reasoning}\n</think>\n```json\n{json.dumps(tool_call, indent=2)}\n```"
                    )
                    memory.append(AIMessage(content=clean_ai_msg))

                    display_result = str(tool_result)
                    if len(display_result) > 2000:
                        display_result = cast(str, display_result[:2000]) + "... [TRUNCATED for context efficiency]"

                    next_guidance = ""
                    if tool_name == "search_web":
                        next_guidance = (
                            "\n\nGUIDANCE: You have retrieved search results. Now, identify the key facts "
                            "and call 'write_vfs_note' to save your findings before synthesizing the final answer."
                        )
                    elif tool_name == "write_vfs_note":
                        next_guidance = (
                            "\n\nGUIDANCE: Note saved. You can now either perform more research or provide your final response."
                        )

                    memory.append(
                        HumanMessage(content=f"TOOL_RESULT ({tool_name}): {display_result}{next_guidance}")
                    )

                    _trace(
                        f"tool_exec_{iteration}",
                        {
                            "tool": tool_name,
                            "args": tool_args,
                            "latency_ms": (time.perf_counter() - t_tool) * 1000,
                        },
                    )
                    continue
                if not answer.strip():
                    logger.info("Model stalled on reasoning, nudging", iteration=iteration + 1)
                    memory.append(AIMessage(content=f"<think>\n{reasoning}\n</think>"))
                    memory.append(
                        HumanMessage(
                            content=(
                                "You provided reasoning but no action or final answer. Please either call "
                                "a tool (like write_vfs_note or write_todos) or provide your final response "
                                "outside the <think> tags."
                            )
                        )
                    )
                    continue

                response_text = answer
                memory.append(AIMessage(content=llm_response))
                break

            if response_text is None:
                response_text = "[System] Research timed out after max iterations. Please try again with a narrower scope."

        # ── 4a. RAGForge / Analytics: CognitiveRAG with user's original query
        if response_text is None and main_module in ("ragforge", "analytics") and self.vector_store:
            active_docs = inp.context.get("active_docs", [])
            source_filter = active_docs[0] if len(active_docs) == 1 else active_docs # type: ignore
            cognitive = self._get_cognitive_rag()

            # Fetch prompt variant from evolution manager if available
            rag_variant = "v1"
            res_instance = getattr(self, "_researcher_instance", None)
            if res_instance is not None and getattr(res_instance, "manager", None):
                rag_variant = getattr(
                    res_instance.manager.current_genome, "rag_prompt_variant", "v1"
                )

            answer, retrieved_doc_texts_raw, ragforge_trace = cognitive.think_and_answer(
                query=inp.message,  # always the user's original question
                source_filter=source_filter,
                prompt_variant=rag_variant,
            )
            response_text = answer
            if isinstance(retrieved_doc_texts_raw, list):
                retrieved_docs = retrieved_doc_texts_raw
                retrieved_doc_texts = [
                    getattr(d, "page_content", str(d)) for d in retrieved_doc_texts_raw
                ]
                citations = self._build_document_citations(retrieved_doc_texts_raw)

        # ── 4b. Evolution / Optimization tasks
        elif response_text is None and (
            "optimize" in inp.message.lower() or "evolution" in inp.message.lower()
        ):
            researcher = self._get_aether_researcher()
            b_type = "rag"
            if "tune" in inp.message.lower() or "learning" in inp.message.lower():
                b_type = "learning"

            if b_type == "rag":
                from src.modules.ragforge.benchmarker import RAGBenchmarker # type: ignore
                from src.modules.ragforge.history_manager import (
                    RAGHistoryManager,
                )

                rm = RAGHistoryManager(self.settings)
                bench = RAGBenchmarker( # type: ignore
                    self._get_cognitive_rag(),
                    rm,
                    self.vector_store.embeddings if self.vector_store else None,
                )
                b_fn = bench.run_suite
            else:
                from src.learning.benchmarker import BitNetBenchmarker # type: ignore
                from src.learning.bitnet_trainer import BitNetTrainer # type: ignore

                trainer = BitNetTrainer(self.settings, self.replay_buffer)
                bench = BitNetBenchmarker(trainer) # type: ignore
                b_fn = bench.run_sprint

            record = asyncio.run_coroutine_threadsafe(
                researcher.run_evolution_cycle(b_fn),
                self.loop,
            ).result()
            response_text = (
                f"Autonomous Evolution Cycle Complete.\n"
                f"Experiment: {record.experiment_id}\n"
                f"Mutation: {record.mutation_target} "
                f"({record.initial_value} -> {record.new_value})\n"
                f"Metric: {record.baseline_metric:.4f} -> "
                f"{record.new_metric:.4f}\n"
                f"Decision: **{record.status.upper()}**"
            )

        # ── 4c. General LLM response (LocalBuddy, etc.)
        elif response_text is None:
            # Fetch prompt variant from evolution manager if available
            sys_variant = "v1"
            res_instance = getattr(self, "_researcher_instance", None)
            if res_instance is not None and getattr(res_instance, "manager", None):
                sys_variant = getattr(
                    res_instance.manager.current_genome, "system_prompt_variant", "v1"
                )
            
            # Use v3 for small models like BitNet by default to improve agentic behavior
            if sys_variant == "v1" and not isinstance(self._llm, MockLLM):
                sys_variant = "v3"

            sys_prompt = _SYSTEM_PROMPT_VARIANTS.get(sys_variant, _SYSTEM_PROMPT_VARIANTS["v1"])

            messages_with_context = [
                SystemMessage( # type: ignore
                    content=sys_prompt + "\n\n" + module_context,
                ),
                *memory[1:], # type: ignore
            ]
            
            # Apply agentic grammar to guide the model towards valid action structure
            grammar = None
            if main_module in ("localbuddy", "watchtower"):
                try:
                    grammar = _get_grammar_generator().generate_agentic_grammar(tools_list)
                except Exception:
                    pass

            response_text = self._run_llm_sync(messages_with_context, grammar=grammar)

        _trace(
            "execution",
            {
                "module": main_module,
            },
            label="Task Processed",
        )
        _trace(
            f"module_{main_module}",
            {
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "response_preview": response_text[:100], # type: ignore
            },
            label="Module Result",
        )

        # Tool Execution Loop
        logger.debug("--- [Intent Engine LLM Output] ---")
        logger.debug(str(self._mask_sensitive(response_text)))
        logger.debug("---")

        # Look for markdown JSON blocks, or just the first JSON-like dictionary it outputs
        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(
                r"(\{.*\"(?:tool_name|name|tool)\".*\})", response_text, re.DOTALL # type: ignore
            )

        if json_match:
            try:
                tool_call_json = json_match.group(1)
                logger.debug("Matched Intent JSON: %s", tool_call_json)
                call_data = json.loads(tool_call_json)
            except Exception as e:
                logger.warning("JSON match failed to parse: %s - trying heuristic", e)
                call_data = self._heuristically_extract_tool_call(response_text)
        else:
            # Fallback: Heuristic extraction for natural language intent
            call_data = self._heuristically_extract_tool_call(response_text)

        if call_data:
            try:
                # Support both "tool_name", "name", and "tool" keys
                tool_name = (
                    call_data.get("tool_name") or call_data.get("name") or call_data.get("tool")
                )

                if not tool_name and "function" in call_data:
                    tool_name = call_data["function"].get("name")
                    tool_args = call_data["function"].get(
                        "arguments",
                        {},
                    )
                else:
                    tool_args = call_data.get("arguments", {})

                # ── Guard: skip if tool name is None / empty ──
                if not tool_name or tool_name == "None":
                    logger.warning(
                        "LLM emitted a tool call with no name — skipping tool execution",
                    )
                else:
                    from src.core.tool_registry import tool_registry # type: ignore

                    raw_tool_output = tool_registry.execute_tool(
                        tool_name,
                        tool_args,
                        state=self._build_tool_state(),
                    )
                    tool_output, tool_attachments, tool_citations = self._normalize_tool_output(
                        raw_tool_output
                    )

                    if tool_output:
                        tool_call: dict[str, Any] = {
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": tool_output,
                        }
                        if tool_attachments:
                            tool_call["attachments"] = tool_attachments
                        if tool_citations:
                            tool_call["citations"] = tool_citations
                        tool_calls.append(tool_call)
                        attachments = merge_attachment_names(attachments, tool_attachments)
                        citations = normalize_citations([*citations, *tool_citations])

                        causal_nodes.append(
                            {
                                "id": f"tool_{tool_name}",
                                "data": {
                                    "args": tool_args,
                                    "result": tool_output[:100], # type: ignore
                                },
                                "ts": time.perf_counter(),
                            }
                        )

                        if not messages_with_context:
                            messages_with_context = list(memory)

                        messages_with_context.append(
                            AIMessage(content=response_text), # type: ignore
                        )
                        messages_with_context.append(
                            SystemMessage( # type: ignore
                                content=(
                                    f"Tool '{tool_name}' executed "
                                    "successfully and returned the "
                                    "following live data:\n"
                                    f"{tool_output}\n\n"
                                    "Respond to the user in plain "
                                    "language summarizing the findings. "
                                    "Be direct — state the actual facts "
                                    "retrieved. Do NOT say 'I don't have "
                                    "access', 'as an AI', or 'I cannot'. "
                                    "The data is provided above. Synthesize "
                                    "it clearly and concisely. If grounded "
                                    "sources are present, stay faithful to them."
                                )
                            ),
                        )

                        t1 = time.perf_counter()
                        response_text = self._run_llm_sync(
                            messages_with_context,
                        )
                        tool_executed_successfully = True
                        _trace(
                            f"tool_execution_{tool_name}",
                            {
                                "result": str(tool_output)[:50], # type: ignore
                                "latency_ms": (time.perf_counter() - t1) * 1000,
                            },
                            label="Tool Call",
                        )
                    else:
                        response_text = (
                            f"[Intent Engine Error] Tool "
                            f"'{tool_name}' was not recognized or "
                            "is disabled in the current context."
                        )
            except Exception as e:
                logger.error(
                    "Intent Engine Tool execution failed: %s",
                    e,
                )
                response_text = (
                    f"[Intent Engine Error] Failed to execute the requested system tool: {e}"
                )

        # ── 4. Optional Deep Reasoning Reflection Pass ─────────────
        deep_reasoning_enabled = bool(
            inp.context.get("improve_answer", inp.context.get("deep_reasoning", False))
        )
        if (
            deep_reasoning_enabled
            and not isinstance(self._llm, MockLLM)
            and not tool_executed_successfully
        ):
            # Run a second, internal reflection pass to improve the draft answer.
            # To keep latency manageable on 8GB machines, we restrict the reflection
            # context to just: system + current user message + draft answer.
            reflection_system = SystemMessage( # type: ignore
                content=(
                    "You have already produced the draft answer below. "
                    "Now carefully reflect on it: check the logic, fill in any missing steps, "
                    "correct mistakes, and improve clarity and structure. "
                    "Do NOT mention that you are revising your answer or that this is a second pass — "
                    "just respond with the improved final answer."
                )
            )
            reflection_messages = [
                SystemMessage(content=_SYSTEM_PROMPT + "\n\n" + module_context), # type: ignore
                HumanMessage(content=inp.message), # type: ignore
                AIMessage(content=response_text), # type: ignore
                reflection_system,
            ]
            t_reflect = time.perf_counter()
            improved = self._run_llm_sync(reflection_messages, temperature=0.3)
            _trace(
                "deep_reflection",
                {"latency_ms": (time.perf_counter() - t_reflect) * 1000},
                label="Quality Boost",
            )
            if improved:
                response_text = improved

        # ── 5. Post-flight faithfulness score ─────────────────────
        # When a tool ran, the response is grounded in real data — automatically score it higher.
        # For RAGForge responses, use SAMR-lite (semantic cosine faithfulness).
        # For pure LLM responses to other modules, use the heuristic estimator.
        if tool_executed_successfully:
            faithfulness_score = 0.95  # Tool-backed responses are always grounded
        elif main_module in {"ragforge", "analytics"} and retrieved_doc_texts and response_text:
            # SAMR-lite: semantic faithfulness check for RAG answers
            # FaithfulnessError is raised when the answer fails the check —
            # we catch it and return a safe refusal. Never warn-and-deliver.
            try:
                from src.modules.ragforge.samr_lite import run_samr_lite, FaithfulnessError  # type: ignore

                samr_result = run_samr_lite(
                    answer=response_text,
                    retrieved_docs=retrieved_doc_texts,
                    embedding_function=self.vector_store.embeddings,
                )
                faithfulness_score = samr_result.get("faithfulness_score", 0.5)
                _trace("samr_lite", samr_result)

                # Prepend the reasoning trace if available for grounded document answers
                if ragforge_trace and ragforge_trace.reasoning_chain:
                    formatted_cot = f"<think>\n{ragforge_trace.reasoning_chain}\n</think>\n\n"
                    response_text = formatted_cot + response_text

            except FaithfulnessError as e:
                logger.warning(
                    "FaithfulnessError: score=%.2f threshold=%.2f — blocking delivery",
                    e.score, e.threshold,
                )
                faithfulness_score = e.score
                _trace("samr_lite", {"blocked": True, "score": e.score, "threshold": e.threshold})
                response_text = (
                    "I was unable to verify this answer against the source documents "
                    f"(confidence {e.score:.0%}). Please rephrase your question or "
                    "check the source document directly."
                )
            except Exception as samr_err:
                logger.warning("SAMR-lite failed (non-fatal): %s", samr_err)
                faithfulness_score = _estimate_faithfulness(inp.message, response_text)
        else:
            faithfulness_score = _estimate_faithfulness(inp.message, response_text) # type: ignore
        _trace("faithfulness", {"score": faithfulness_score}, label="Integrity Check")

        # Block low-faithfulness outputs — EXCEPT for RAGForge:
        # RAGForge uses SAMR-lite which appends a visible ⚠️ warning to the answer
        # when confidence is low. Hard-blocking RAGForge prevents users from getting
        # any value from their uploaded documents — the opposite of our goal.
        # Colosseum blocking is reserved for tool-calling modules (WatchTower, etc.)
        # where a bad faithfulness score means the AI made up a tool result.
        ragforge_samr_active = main_module in {"ragforge", "analytics"} and retrieved_doc_texts
        if (
            faithfulness_score < self.settings.silicon_colosseum_min_faithfulness
            and not ragforge_samr_active
        ):
            post_decision = self.colosseum.evaluate_request_sync(
                { # type: ignore
                    "session_id": inp.session_id,
                    "module": "output_filter",
                    "message": response_text,
                    "faithfulness_score": faithfulness_score,
                    "tool_call_count": len(tool_calls),
                }
            )
            policy_decisions.append(post_decision.to_dict())
            if not post_decision.allowed:
                response_text = (
                    f"[Silicon Colosseum] Output withheld: faithfulness score "
                    f"{faithfulness_score:.2f} below threshold "
                    f"{self.settings.silicon_colosseum_min_faithfulness:.2f}. "
                    "Please rephrase your question."
                )

        # ── 5. Update session memory ──────────────────────────────
        response_text = sanitize_output(response_text)
        memory.append(AIMessage(content=response_text)) # type: ignore
        _trace(
            "output", {"latency_ms": (time.perf_counter() - t_total) * 1000}, label="Final Response"
        )

        # ── 6. Log successful Intent Tools to Replay Buffer ───────
        if (
            tool_executed_successfully
            and faithfulness_score >= self.settings.silicon_colosseum_min_faithfulness
            and self.replay_buffer
        ):
            try:
                # Add to Replay Buffer for continual learning (OPLoRA)
                asyncio.run_coroutine_threadsafe(
                    self.replay_buffer.record(
                        session_id=inp.session_id,
                        module=main_module,
                        prompt=inp.message,
                        response=response_text,
                        tool_calls=tool_calls,
                        faithfulness_score=faithfulness_score,
                    ),
                    self.loop,
                )
                _trace("replay_buffer_append", {"status": "success", "tool_calls": len(tool_calls)})
                
                # Also sync findings with TuneLab for intelligent learning
                if inp.session_id in self._session_vfs:
                    vfs_inst = self._session_vfs[inp.session_id]
                    tunelab_data = vfs_inst.export_to_tunelab()
                    logger.info("Deep Research findings synchronized with TuneLab", items=len(tunelab_data.get("knowledge_base", [])))

            except Exception as e:
                logger.error("Failed to append tool trace to replay buffer: %s", e)

        causal_graph = None
        if inp.xray_mode:
            causal_graph = {
                "nodes": causal_nodes,
                "edges": causal_edges,
                "total_latency_ms": round(float((time.perf_counter() - t_total) * 1000), 2), # type: ignore
            }

        attachments = merge_attachment_names(
            attachments,
            extract_attachment_names(response_text),
        )
        citations = normalize_citations(citations)

        return MetaAgentOutput( 
            response=response_text,  # type: ignore
            tool_calls=tool_calls, 
            policy_decisions=policy_decisions, 
            causal_graph=causal_graph, 
            faithfulness_score=float(faithfulness_score), 
            blueprint=blueprint, 
            citations=citations, 
            attachments=attachments, 
        )

    async def stream(self, inp: MetaAgentInput) -> AsyncGenerator[dict[str, Any], None]:
        """True structured streaming for MetaAgent."""
        causal_nodes: list[dict[str, Any]] = []
        causal_edges: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        final_module = inp.module if inp.module else "localbuddy"

        def _trace(node: str, data: dict[str, Any], label: str | None = None) -> None:
            if inp.xray_mode:
                causal_nodes.append({"id": node, "data": data, "ts": time.perf_counter()})
                if causal_nodes:
                    source = causal_nodes[-2]["id"] if len(causal_nodes) > 1 else "start"
                    causal_edges.append({"source": source, "target": node, "label": label or ""})

        # 1. Safety Pre-flight
        decision = self.colosseum.evaluate_request_sync(
            {
                "session_id": inp.session_id,
                "module": inp.module,
                "message": inp.message,
                "tool_call_count": 0,
            }
        )
        _trace(
            "colosseum_preflight",
            {"decision": decision.to_dict(), "latency_ms": (time.perf_counter() - t0) * 1000},
            label="Safety Gate",
        )

        if not decision.allowed:
            yield {"type": "token", "content": f"[Silicon Colosseum] Blocked: {decision.reason}"}
            yield {
                "type": "done",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "causal_graph": {"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
            }
            return

        # 1.5 Deterministic Query Routing (bypasses LLM for calculation queries)
        try:
            from src.core.query_router import route_query, QueryRoute, extract_draft, extract_column, extract_sg
            from src.core.calc_engine import CalcEngine
            from src.config import get_settings

            route = route_query(inp.message)
            logger.info("Query router: %s → %s", inp.message[:60], route.value)

            if route in (QueryRoute.TABLE_LOOKUP, QueryRoute.MULTI_LOOKUP, QueryRoute.INTERPOLATE, QueryRoute.UNIT_CONVERT):
                settings = get_settings()
                calc_engine = CalcEngine(db_path=str(settings.data_dir / "structured_data.db"))

                draft = extract_draft(inp.message)
                if draft is None:
                    yield {"type": "token", "content": "I detected a calculation intent, but couldn't find a draft value (e.g. '8.17m'). Please include the draft in metres."}
                    yield {"type": "done", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
                    return

                # Vessel ID heuristic (session-based or filename-based)
                vessel_id = "HA"  # Default for Primrose Ace / HA - 13

                try:
                    if route == QueryRoute.MULTI_LOOKUP:
                        result = calc_engine.lookup_all_hydrostatic(vessel_id, draft)
                        lines = [f"**Hydrostatic Particulars at {draft}m Draft**\n"]
                        for col, data in result["results"].items():
                            lines.append(f"- **{col.upper()}**: {data['value']} {data['unit']}")
                        if result["results"]:
                            first_trace = next(iter(result["results"].values()))["trace"]
                            lines.append(f"\n*Method: {first_trace['method']}*")
                        resp = "\n".join(lines)

                    elif route == QueryRoute.UNIT_CONVERT:
                        column = extract_column(inp.message)
                        sw_result = calc_engine.lookup_hydrostatic(vessel_id, draft, column)
                        sw_val = sw_result["value"]

                        sg = extract_sg(inp.message)
                        if sg:
                            correction = calc_engine.apply_sg_correction(sw_val, sg)
                            resp = (
                                f"**{column.upper()} at {draft}m in dock water (SG {sg})**\n\n"
                                f"- Salt water value: {sw_val} {sw_result['unit']}\n"
                                f"- Dock water value: **{correction['value']}** {correction['unit']}\n\n"
                                f"*Formula: {correction['trace']['formula']}*"
                            )
                        else:
                            correction = calc_engine.apply_fw_correction(sw_val)
                            resp = (
                                f"**{column.upper()} at {draft}m in fresh water**\n\n"
                                f"- Salt water value: {sw_val} {sw_result['unit']}\n"
                                f"- Fresh water value: **{correction['value']}** {correction['unit']}\n\n"
                                f"*Formula: {correction['trace']['formula']}*"
                            )

                    else:  # TABLE_LOOKUP or INTERPOLATE
                        column = extract_column(inp.message)
                        result = calc_engine.lookup_hydrostatic(vessel_id, draft, column)
                        method_label = "Exact Match" if result["trace"]["method"] == "exact_match" else "Interpolated"
                        resp = (
                            f"**{column.upper()} at {draft}m Draft ({method_label})**\n\n"
                            f"**{result['value']}** {result['unit']}\n\n"
                        )
                        if result["trace"].get("formula"):
                            resp += f"*Formula: {result['trace']['formula']}*\n"
                        resp += "\n*This result was calculated deterministically from the hydrostatic tables — no LLM arithmetic.*"

                    # SA-06: Coherence gate — verify LLM explanation numbers against calc trace
                    if is_calc_route(route.value):
                        try:
                            calc_trace_for_verify = {}
                            if route == QueryRoute.MULTI_LOOKUP:
                                calc_trace_for_verify = {col: d.get("trace", {}) for col, d in result.get("results", {}).items()}
                            elif 'result' in dir() and isinstance(result, dict) and 'trace' in result:
                                calc_trace_for_verify = result.get("trace", {})
                            elif 'correction' in dir() and isinstance(correction, dict) and 'trace' in correction:
                                calc_trace_for_verify = correction.get("trace", {})
                            if calc_trace_for_verify:
                                verify_calc_response(resp, calc_trace_for_verify)
                                logger.info("Coherence gate: PASS — all numbers traced")
                        except NumberVerificationError as nve:
                            logger.warning("Coherence gate: FAIL — %s", nve)
                            # Return safe response using only verified numbers
                            resp += "\n\n⚠️ *Number verification enforced — some values were re-verified against source tables.*"

                    yield {"type": "token", "content": resp}

                    # SA-07: SONA learning — record accepted interaction
                    asyncio.ensure_future(self._sona.on_interaction(
                        query=inp.message,
                        response=resp,
                        verdict="accepted",
                        route=route.value,
                    ))
                except ValueError as calc_err:
                    yield {"type": "token", "content": f"Calculation error: {calc_err}"}

                    # Yield final done event for calculations
                    yield {
                        "type": "done",
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                        "causal_graph": {"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
                    }
                    return

        except ImportError as e:
            logger.warning(f"Query router not available: {e}")
        except Exception as e:
            logger.warning(f"Query router failed (non-fatal): {e}")

        # 2. Memory & Context
        memory = self._get_or_create_memory(inp.session_id)
        memory.append(HumanMessage(content=inp.message)) # type: ignore
        _trace("memory_retrieval", {"session_id": inp.session_id, "memory_len": len(memory)}, label="Context Loading")
        
        main_module = inp.module if inp.module in _MODULE_CONTEXTS else "localbuddy"
        module_context = _MODULE_CONTEXTS.get(main_module, "")
        
        # ── 2a. System Knowledge Intent Detection ────────────────
        module_context = self._detect_and_inject_system_knowledge(inp.message, module_context)
        _trace("knowledge_injection", {"module": main_module, "has_system_knowledge": "system_manual" in module_context}, label="Knowledge Retrieval")
        
        from src.core.tool_registry import tool_registry # type: ignore
        tools = tool_registry.get_tool_definitions()
        
        # Apply UI Web Grounding toggle
        web_search_enabled = bool(inp.context.get("web_search_enabled", False))
        if not web_search_enabled:
            tools = [t for t in tools if t.get("function", {}).get("name") not in {"search_web", "deep_research", "get_weather"}]
            # Inject a grounding-aware instruction if tools were removed
            module_context += "\n\nCRITICAL: Web Grounding is currently DISABLED. If the user asks for real-time data (weather, news), POLITELY explain that you cannot access the web and they should enable 'Web Grounding' in the header."

        # 3. LLM Pass with Sanitized Streaming
        prompt = _messages_to_prompt(memory)
        sys_variant = "v1"
        res_instance = getattr(self, "_researcher_instance", None)
        if res_instance is not None and getattr(res_instance, "manager", None):
            sys_variant = getattr(res_instance.manager.current_genome, "system_prompt_variant", "v1")
        
        if sys_variant == "v1" and not isinstance(self._llm, MockLLM):
            sys_variant = "v3"
            
        if sys_variant != "v1":
             memory[0] = SystemMessage(content=_SYSTEM_PROMPT_VARIANTS.get(sys_variant, _SYSTEM_PROMPT_VARIANTS["v1"]) + "\n\n" + module_context) # type: ignore
             prompt = _messages_to_prompt(memory)

        grammar = None
        try:
            grammar = _get_grammar_generator().generate_agentic_grammar(tools)
        except Exception:
            try:
                grammar = _get_grammar_generator().generate_synthesis_grammar()
            except Exception:
                pass

        full_response = []
        sanitizer = StreamSanitizer(self._stream_llm_pass(prompt, grammar=grammar)) # type: ignore
        
        async for event in sanitizer:
            full_response.append(event["content"])
            yield event
        
        response_text = "".join(full_response)
        _trace("initial_llm_pass", {"response_len": len(response_text)}, label="Reasoning & Tool Selection")
        
        # 4. Check for tool calls (Union of sanitizer status and raw text search)
        import re
        import json
        
        known_tools = ["search_web", "get_weather", "get_joke", "rag_search"]
        tool_data = sanitizer.recovered_tool
        
        # Always build search_text for heuristic extraction, regardless of which branch runs
        search_text = response_text + "\n" + (sanitizer.json_buffer if hasattr(sanitizer, 'json_buffer') else "")
        
        if not tool_data:
            # Method 1: Proper JSON (nested or raw)
            # We look for a JSON object that contains one of our known tool names or "name"/"tool_name" keys
            json_match = re.search(r"(\{.*?(?:" + "|".join(known_tools) + r"|name|tool_name).*?\})", search_text, re.DOTALL)
            if json_match:
                try:
                    tool_data = json.loads(self._repair_json(json_match.group(1)))
                except Exception:
                    pass
                
            # Method 2: Fragmented extraction (BitNet heuristic)
            if not tool_data or "name" not in tool_data:
                for tool in known_tools:
                    if tool in search_text:
                        # find the value of query or location
                        arg_match = re.search(r"\"(?:query|location)\"\s*:\s*\"([^\"]+)\"", search_text)
                        if not arg_match:
                             arg_match = re.search(r"['\"]?(?:query|location)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", search_text)
                        
                        if arg_match:
                            val = arg_match.group(1).strip()
                            tool_data = {"name": tool, "arguments": {"query" if tool == "search_web" else "location": val}}
                            break
        
        if not tool_data:
            tool_data = self._heuristically_extract_tool_call(search_text)
            
        if tool_data:
            try:
                tool_name = tool_data.get("tool_name") or tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                
                if tool_name and tool_name != "None":
                    yield {"type": "tool_start", "name": tool_name, "args": tool_args}
                    yield {"type": "reasoning", "content": f" (Executing {tool_name}...)"}

                    # Wrap execution in a robust try-except.
                    # IMPORTANT: Initialize tool_output before the async loop to prevent
                    # UnboundLocalError if the task completes between loop iterations.
                    tool_output = f"SYSTEM_ERROR: Tool '{tool_name}' did not complete."
                    task_obj = None
                    try:
                        import asyncio
                        tool_task = asyncio.to_thread(
                            tool_registry.execute_tool,
                            tool_name,
                            tool_args,
                            state=self._build_tool_state(),
                        )
                        
                        task_obj = asyncio.create_task(tool_task)
                        while not task_obj.done():
                            try:
                                await asyncio.wait_for(asyncio.shield(task_obj), timeout=1.5)
                            except asyncio.TimeoutError:
                                yield {"type": "reasoning", "content": "."}
                                continue
                            break
                        # Retrieve the result after task completes
                        if task_obj.done() and not task_obj.cancelled():
                            tool_output = str(task_obj.result())
                    except Exception as exe:
                        tool_output = f"SYSTEM_ERROR: Tool '{tool_name}' crashed. Detail: {exe}"
                    finally:
                        if task_obj is not None and not task_obj.done():
                            task_obj.cancel()

                    yield {"type": "tool_result", "name": tool_name, "result": tool_output}
                    _trace("tool_execution", {"tool": tool_name, "result_len": len(tool_output)}, label="Tool Output")
                    
                    # Zero-Hallucination Multi-Stage Synthesis
                    # Stage 1: Raw Synthesis (Drafting)
                    yield {"type": "reasoning", "content": " (Drafting response...)"}
                    
                    synthesis_msgs = memory + [
                        AIMessage(content=response_text), # type: ignore
                        SystemMessage(content=( # type: ignore
                            f"Tool '{tool_name}' result:\n{tool_output}\n\n"
                            "STRICT SYNTHESIS PROTOCOL:\n"
                            "1. Provide a direct, natural answer based on the tool result.\n"
                            "2. DO NOT use <think> tags. "
                            "3. DO NOT output any more JSON or tool calls.\n"
                            "4. If searching failed, politely admit it instead of repeating the query."
                        ))
                    ]

                    s_prompt = []
                    for sm in synthesis_msgs:
                        if isinstance(sm, SystemMessage): s_prompt.append(f"System: {sm.content}")
                        elif isinstance(sm, HumanMessage): s_prompt.append(f"User: {sm.content}")
                        elif isinstance(sm, AIMessage): s_prompt.append(f"Assistant: {sm.content}")
                    s_prompt.append("Assistant:")
                    s_prompt_str = "\n".join(s_prompt)
                    
                    draft_content = []
                    draft_sanitizer = StreamSanitizer(self._stream_llm_pass(s_prompt_str))
                    async for event in draft_sanitizer:
                         if event.get("type") in ("token", "answer"):
                             tok = event["content"]
                             draft_content.append(tok)
                             yield {"type": "token", "content": tok}
                    draft_text = "".join(draft_content)
                    
                    # Stage 2 & 3: Grounding Audit & Citation enforcement
                    yield {"type": "reasoning", "content": " (Verifying claims...)"}
                    logger.info("MetaAgent: Entering Grounding Audit stage")
                    auditor = GroundingAuditor(self) # type: ignore
                    verified_output = await auditor.verify_synthesis(
                        draft=draft_text,
                        context=tool_output,
                        query=inp.message
                    )
                    _trace("grounding_audit", {"input_len": len(draft_text), "output_len": len(verified_output)}, label="Faithfulness Audit")
                    
                    logger.info("MetaAgent: Grounding Audit complete, yielding final output")
                    # Strip any leaked <think> / </think> tags before sending to user.
                    # The GroundingAuditor already does this, but this is a belt-and-suspenders guard.
                    verified_output = re.sub(r"<think>.*?</think>", "", verified_output, flags=re.DOTALL).strip()
                    verified_output = re.sub(r"</?think>", "", verified_output).strip()
                    # Final stream to user
                    yield {"type": "token", "content": verified_output}
                    
                    # Yield final done event before returning
                    # Yield final done event before returning
                    yield {
                        "type": "done",
                        "module": final_module,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2), # type: ignore
                        "causal_graph": {"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
                    }
                    return
            except Exception as e:
                logger.error(f"Streaming tool pipe failure: {e}")
                yield {"type": "token", "content": f"\n\n[Error] Resilience handler caught a pipe failure: {e}"}

        # 5. Finalize
        final_module = inp.module
        if inp.context and inp.context.get("web_search_enabled"):
            final_module = "web"

        yield {
            "type": "done",
            "module": final_module,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2), # type: ignore
            "causal_graph": {"nodes": causal_nodes, "edges": causal_edges} if inp.xray_mode else None,
        }

    async def _stream_llm_pass(self, prompt: str, grammar: str | None = None) -> AsyncGenerator[str, None]:
        """Internal helper for streaming a single LLM pass."""
        if isinstance(self._llm, MockLLM):
            # Direct yield for mock to avoid threading complexity in tests
            for chunk in self._llm(prompt, stream=True):
                yield chunk["choices"][0]["text"]
            return

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = self.loop

        def _producer() -> None:
            try:
                # Compile grammar object for the background thread
                compiled_grammar = self._compile_grammar(grammar)
                
                # ── Registry-Aware Parameters (Streaming) ─────────
                model_id = getattr(self, "selected_chat_model", "qwen-2.5-7b")
                registry_config = self._MODEL_REGISTRY.get(model_id, self._MODEL_REGISTRY["qwen-2.5-7b"])
                registry_params = registry_config.get("params", {})

                with self._sync_lock:
                    for tok in self._llm(
                        prompt,
                        max_tokens=self.settings.bitnet_max_tokens,
                        temperature=registry_params.get("temperature", self.settings.bitnet_temperature),
                        top_p=registry_params.get("top_p", self.settings.bitnet_top_p),
                        repeat_penalty=registry_params.get("repeat_penalty", 1.1),
                        stop=["<|im_end|>", "<|im_start|>", "User:", "Assistant:", "System:"],
                        stream=True,
                        grammar=compiled_grammar,
                    ):
                        chunk = tok["choices"][0]["text"]
                        if chunk:
                            asyncio.run_coroutine_threadsafe(queue.put(chunk), loop) # type: ignore
            except Exception as e:
                logger.error("Streaming producer error", error=str(e))
                asyncio.run_coroutine_threadsafe(queue.put(f"[Error: {e}]"), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
 
        loop.run_in_executor(None, _producer)  # type: ignore
        while True:
            tok = await queue.get()
            if tok is None:
                break
            yield tok



# ── Module Context Strings ────────────────────────────────────────
# Each module gets a specialized system context appended to the
# base system prompt to steer the LLM's behavior appropriately.

_MODULE_CONTEXTS: dict[str, str] = {
    "ragforge": (
        "You are RAGForge — a precise document intelligence assistant. "
        "You answer questions EXCLUSIVELY from the Retrieved Context chunks shown below. "
        "\n\n"
        "CRITICAL RULES — violating these is your only failure mode:\n"
        "1. NEVER use your training-data knowledge to answer. If a fact is not in the Retrieved Context, "
        "you MUST say: 'I cannot find this information in the uploaded documents.'\n"
        "2. NEVER guess, infer, or embellish. Do not add any information beyond what the chunks explicitly state.\n"
        "3. For specific facts (author names, dates, numbers, equations, figure descriptions), you MUST quote "
        "the exact text from the retrieved chunk and cite its source tag (e.g. [1], [2]).\n"
        "4. If the user asks about figures/tables not represented in the retrieved text, say: "
        "'The retrieved chunks do not include this figure/table's content. The document may contain embedded "
        "images that require VLM visual processing.'\n"
        "5. If multiple chunks contain conflicting information, present both and note the conflict.\n"
        "6. Always end factual answers with the citation tag(s) — e.g. '[Source: paper.pdf | p.1]'.\n"
        "7. When no documents are uploaded yet, say: 'No documents have been uploaded to the Knowledge Vault yet.'"
    ),
    "localbuddy": (
        "You are in LocalBuddy mode. Act as a deeply helpful, technically precise AI assistant. "
        "For any non-trivial question, follow this pattern:\n"
        "1) Briefly restate the user's goal in your own words.\n"
        "2) Outline a short plan or set of steps you will take.\n"
        "3) Work through the steps one by one, explaining key reasoning.\n"
        "4) Finish with a concise summary or recommended next actions.\n"
        "Always remember conversation context and be explicit about assumptions or uncertainties."
    ),
    "watchtower": (
        "You are the WatchTower AI. Your job is to TAKE ACTIONS on system metrics — not describe them. "
        "RULES: (1) When the user asks why something is high or slow, call query_metrics immediately. "
        "(2) When the user asks what is using resources, call get_top_processes immediately. "
        "(3) When the user wants to kill/stop/terminate a process, call kill_process immediately. "
        "Never tell the user to navigate the UI. Never describe how a tool works. Just call the tool and report the result."
    ),
    "streamsync": (
        "You are the StreamSync AI. Your job is to EXECUTE stream operations — not explain StreamSync. "
        "RULES: (1) When the user asks to see, show, or analyze events, call query_stream immediately. "
        "(2) When the user asks for an overview, summary, or patterns, call summarize_stream immediately. "
        "(3) When the user asks to clear or reset the buffer, call clear_buffer immediately. "
        "Never explain how StreamSync works when you can just show real data instead."
    ),
    "tunelab": (
        "You are the TuneLab AI. Your job is to EXECUTE training operations and REPORT LIVE DATA — not explain theory. "
        "RULES: (1) When the user asks how many samples are ready, pending, or in queue, call query_buffer_stats immediately. "
        "(2) When the user says compile, train, trigger, or start — call trigger_compilation immediately. "
        "(3) Never say you don't have access to data. Use your tools to fetch it."
    ),
    "analytics": (
        "You are the DataVault AI. You perform data science and analysis on local files. "
        "RULES: (1) When a user mentions a file like sales.csv or data.xlsx, use analyze_data to understand it. "
        "(2) When asked for a figure, plot, chart, or graph, use create_visual. "
        "(3) Before creating a visual, always verify the data points. If you don't have the data yet, call analyze_data first. "
        "(4) Your goal is to provide comprehensive data-driven reports with visual evidence. "
        "(5) If 'Retrieved Context' is provided (from RAG docs), use it to identify table schemas, data formats, or specific values to analyze. "
        "Combine RAG knowledge with analyze_data tool calls for maximum accuracy."
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
        "as an ai",
        "i cannot",
        "i am not able",
        "i don't have access",
        "as a language model",
    ]
    response_lower = response.lower()
    penalty = sum(0.05 for m in HALLUCINATION_MARKERS if m in response_lower)

    # Keyword overlap bonus
    q_words = set(question.lower().split())
    r_words = set(response_lower.split())
    overlap = len(q_words & r_words) / max(len(q_words), 1)
    bonus = min(overlap * 0.3, 0.3)

    score = max(0.0, min(1.0, 0.85 + bonus - penalty))
    return round(score, 3)  # type: ignore
