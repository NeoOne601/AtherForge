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
import structlog
import threading
import time
from typing import Any, Literal, Optional
from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import AetherForgeSettings
from src.guardrails.silicon_colosseum import SiliconColosseum

logger = structlog.get_logger("aetherforge.meta_agent")

_SYSTEM_PROMPT_VARIANTS = {
    "v1": (
        "You are AetherForge, a local AI assistant. You run entirely on-device and "
        "never send data to the cloud. "
        "Present a clear, structured answer to the user. "
        "Prefer numbered lists or sections for non-trivial problems, and call out any "
        "assumptions you make. "
        "Be concise, accurate, and transparent about uncertainty.\n\n"
        "THINKING RULES:\n"
        "Before answering, briefly reason through the problem inside <think> tags. "
        "Put ONLY your final answer OUTSIDE the <think> tags. Example:\n"
        "<think>The user is asking about X. I should consider Y and Z.</think>\n"
        "Here is my answer about X...\n\n"
        "TOOL CALLING RULES (CRITICAL):\n"
        "1. You MUST call a tool if the user's request requires live data (weather, news, system stats).\n"
        "2. Output the tool call in a ```json block BEFORE any other text.\n"
        "3. Do NOT invent data if a tool can provide it."
    ),
    "v2": (
        "You are AetherForge, a highly capable local AI running entirely on-device. "
        "Your first priority is FAITHFULNESS and TRUTH. Do not hallucinate. "
        "When explaining complex topics, use clear analogies and well-formatted markdown. "
        "If you are unsure, state 'I don't know' rather than guessing.\n\n"
        "THINKING PROCESS:\n"
        "You must analyze every request internally before responding. Use <think> tags "
        "for your step-by-step reasoning. Only write your final response outside the tags.\n"
        "<think>1. Identify intent. 2. Verify constraints. 3. Formulate answer.</think>\n"
        "Final response goes here.\n\n"
        "TOOL PROTOCOL (MANDATORY):\n"
        "If the user asks for real-time information (e.g. current weather), you MUST trigger a tool. "
        "Format the tool call as a JSON block immediately.\n"
        "```json\n{\"name\": \"tool_name\", \"arguments\": {...}}\n```"
    )
}



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
            
    # Add trailing prompt initiator
    parts.append("<|im_start|>assistant\n")
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
            messages.append(AIMessage(content=content))
    return messages


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
    system_location: Optional[str] = None


# ── IRA Framework Data Models ─────────────────────────────────────

class BlueprintTask(BaseModel):
    """A single sub-task within an IRA Blueprint."""
    id: str
    description: str
    module: str
    tool_call: Optional[dict[str, Any]] = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Optional[str] = None
    reflection: Optional[str] = None

class IRABlueprint(BaseModel):
    """The recursive plan generated before execution."""
    goal: str
    tasks: list[BlueprintTask] = []
    metadata: dict[str, Any] = {}

class IRAState(BaseModel):
    """Memory state for the recursive agentic loop."""
    session_id: str
    blueprint: Optional[IRABlueprint] = None
    current_task_idx: int = 0
    internal_history: list[dict[str, Any]] = []
    recursion_depth: int = 0
    max_recursion: int = 5
    is_complete: bool = False

class MetaAgentOutput(BaseModel):
    """Typed output from a single meta-agent turn."""
    response: str
    tool_calls: list[dict[str, Any]] = []
    policy_decisions: list[dict[str, Any]] = []
    causal_graph: dict[str, Any] | None = None
    faithfulness_score: float | None = None
    blueprint: Optional[IRABlueprint] = None


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
        replay_buffer: Any = None
    ) -> None:
        self.settings = settings
        self.colosseum = colosseum
        self.vector_store = vector_store
        self.replay_buffer = replay_buffer
        self._sparse_index = sparse_index
        self._llm: Any = None
        # Use asyncio.Lock for async-safe model switching, threading.Lock for sync inference
        self._lock: asyncio.Lock = asyncio.Lock()
        self._sync_lock: threading.Lock = threading.Lock()
        self._session_memories: collections.OrderedDict[str, list[Any]] = collections.OrderedDict()
        
        # Declare lazy-initialized attributes explicitly to satisfy static analysis
        self._cognitive_rag_instance: Optional[Any] = None
        self._researcher_instance: Optional[Any] = None
        self.loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        # Headroom attributes (declared here; conditionally initialized below)
        self.token_counter: Any = None
        self.headroom_tokenizer: Any = None
        self.headroom_config: Any = None
        self.context_manager: Any = None

        # ── Headroom Context Optimization ─────────────────────
        try:
            from headroom.tokenizer import Tokenizer
            from headroom.providers.openai_compatible import OpenAICompatibleTokenCounter
            from headroom.transforms.intelligent_context import IntelligentContextManager
            from headroom.config import IntelligentContextConfig

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
        
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        
        # Run synchronous LLM call in a thread pool to avoid blocking the event loop
        loop = self.loop
        def _run():
            return self._run_llm_sync(messages, max_tokens=256, temperature=0.1)
            
        refined = await loop.run_in_executor(None, _run)
        
        # Post-processing: remove any residual quotes or labels if the model leaked them
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1]
        if ":" in refined[:20] and any(lbl in refined[:20].lower() for lbl in ["corrected", "refined", "output"]):
            refined = refined.split(":", 1)[1].strip()
            
        return refined.strip()

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

        loop = self.loop
        self._llm = await loop.run_in_executor(None, _load)
        logger.info("BitNet model loaded")

    async def switch_model(self, model_id: str) -> bool:
        """
        Dynamically unloads the current LLM, downloads a new one via HF Hub if missing,
        and reloads it into unified memory.
        """
        import gc
        from huggingface_hub import hf_hub_download

        MODELS = {
            "bitnet-2b": {
                "repo": None,  # System default, should already exist
                "file": "bitnet-b1.58-2b-4t.gguf"
            },
            "gemma2-2b": {
                "repo": "bartowski/gemma-2-2b-it-GGUF",
                "file": "gemma-2-2b-it-Q4_K_M.gguf"
            },
            "llama3.2-3b": {
                "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            }
        }
        
        if model_id not in MODELS:
            raise ValueError(f"Unknown chat model ID: {model_id}")

        info = MODELS[model_id]
        filename = info["file"]
        repo_id = info["repo"]
        target_path = self.settings.bitnet_model_path.parent / filename

        async with self._lock:
            logger.info("Unloading current LLM to free memory...")
            if self._llm is not None:
                del self._llm
                self._llm = None
                gc.collect()

            if not target_path.exists():
                if not repo_id:
                    raise FileNotFoundError(f"Model file {filename} missing and no repo to pull from.")
                logger.info("Downloading %s from HuggingFace (%s)...", filename, repo_id)
                def _download() -> str:
                    return hf_hub_download(  # type: ignore[return-value]
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(self.settings.bitnet_model_path.parent)
                    )
                loop = self.loop
                await loop.run_in_executor(None, _download)

            logger.info("Binding new model path: %s", target_path)
            self.settings.bitnet_model_path = target_path
            await self._load_model()
            return True
        return False  # pragma: no cover

    def _run_llm_sync(self, messages: list[Any], max_tokens: int | None = None, temperature: float | None = None) -> str:
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
                output_buffer=self.settings.bitnet_max_tokens + 100, # with safety margin
            )
            pruned_messages = _dicts_to_lc_messages(result.messages)
            if result.transforms_applied:
                tokens_saved = result.tokens_before - result.tokens_after
                logger.info("Headroom optimized context: %d tokens saved via %d transforms", 
                            tokens_saved, len(result.transforms_applied))
        else:
            # Legacy fallback
            pruned_messages = list(messages)
            while len(_messages_to_prompt(pruned_messages)) > 12000 and len(pruned_messages) > 3:
                pruned_messages.pop(1)
                pruned_messages.pop(1)

        prompt = _messages_to_prompt(pruned_messages)
        return self._call_llm_with_retry(
            prompt,
            max_tokens=max_tokens or self.settings.bitnet_max_tokens,
            temperature=temperature if temperature is not None else self.settings.bitnet_temperature,
            top_p=self.settings.bitnet_top_p,
            stop=["<|im_end|>"],
        )

    def _call_llm_with_retry(self, prompt: str, **kwargs: Any) -> str:
        """Calls the LLM with exponential backoff for transient failures."""
        max_retries = 3
        delay: float = 1.0
        last_err: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                # Thread-safe LLM access (serialize actual inference but not pipeline)
                with self._sync_lock:
                    result = self._llm(prompt, **kwargs)
                return str(result["choices"][0]["text"]).strip()
            except Exception as e:
                last_err = e
                logger.warning(f"LLM inference attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(float(delay))
                    delay = float(delay) * 2.0
                    
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
                self._session_memories.popitem(last=False)
        return self._session_memories[session_id]

    def _hybrid_search(
        self,
        query: str,
        k: int = 8,
        source_filter: str | list[str] | None = None,
    ) -> list[Any]:
        """
        Hybrid retrieval: Dense (ChromaDB/BGE-M3) + Sparse (FTS5/BM25).
        Falls back to dense-only if FTS5 index is not available.

        CRITICAL: Do NOT cache a permanent 'UNAVAILABLE' sentinel.
        The DB is created when the first document is uploaded, which happens
        AFTER the server starts. Always re-check if the DB now exists.
        """
        # Try to get/init the sparse index on every call when not yet loaded
        if self._sparse_index is None:
            try:
                from src.modules.ragforge.sparse_index import SparseIndex
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
                from src.modules.ragforge.sparse_index import hybrid_search
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
                visual_triggers = {"chart", "diagram", "figure", "table", "image", "visual", "snapshot", "graph"}
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
                        logger.info("Visual boost applied: prioritized %d VLM chunks for query '%s'", len(vlm_docs), query[:30])

                logger.info(
                    "Hybrid search: %d results for '%s...' (filter=%s)",
                    len(results), query[:50], source_filter,
                )
                return results
            except Exception as e:
                logger.warning("Hybrid search failed: %s — falling back to dense", e)

        # Fallback: dense-only (ChromaDB)
        logger.info("Dense-only search for '%s...' (sparse index not loaded yet)", query[:50])
        filter_kwargs: dict[str, Any] = {}
        if source_filter:
            if isinstance(source_filter, str):
                filter_kwargs["filter"] = {"source": source_filter}
            elif isinstance(source_filter, list) and len(source_filter) == 1:
                filter_kwargs["filter"] = {"source": source_filter[0]}
            elif isinstance(source_filter, list):
                filter_kwargs["filter"] = {"source": {"$in": source_filter}}
        return self.vector_store.similarity_search(query, k=k, **filter_kwargs)

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
            "{\"goal\": \"Overall user goal\", \"tasks\": [{\"id\": \"t1\", \"description\": \"...\", \"module\": \"...\"}]}\n"
            "```"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Request: {message}\nContext: {context}")
        ]
        
        try:
            raw_plan = self._run_llm_sync(messages, max_tokens=500, temperature=0.1)
            import json
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', raw_plan, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1))
                return IRABlueprint(**plan_data)
        except Exception as e:
            logger.warning("IRA Blueprint generation failed: %s", e)
            
        # Fallback: single-task blueprint
        return IRABlueprint(
            goal=message,
            tasks=[BlueprintTask(id="task_1", description=message, module="localbuddy")]
        )

    def _get_cognitive_rag(self) -> "CognitiveRAG":
        """Lazily create a CognitiveRAG instance that reuses the loaded BitNet + search."""
        if not hasattr(self, "_cognitive_rag_instance") or self._cognitive_rag_instance is None:
            from src.modules.ragforge.cognitive_rag import CognitiveRAG
            self._cognitive_rag_instance = CognitiveRAG(
                llm_fn=self._run_llm_sync,
                search_fn=self._hybrid_search,
            )
            logger.info("CognitiveRAG pipeline initialized (reusing BitNet + hybrid search)")
        return self._cognitive_rag_instance

    def _get_aether_researcher(self) -> "AetherResearcher":
        """Lazily create an AetherResearcher instance."""
        if not hasattr(self, "_researcher_instance") or self._researcher_instance is None:
            from src.learning.evolution import AetherResearcher, ExperimentManager
            manager = ExperimentManager(self.settings)
            self._researcher_instance = AetherResearcher(manager)
            logger.info("AetherResearcher specialized IRA agent initialized")
        return self._researcher_instance

    async def run(self, inp: MetaAgentInput) -> MetaAgentOutput:
        """
        Execute one full agent turn.
        Concurrency: The model inference is internally serialized by threading.Lock
        to prevent llama-cpp-python race conditions, but RAG and tool lookups
        are now non-blocking.
        """
        return await self.loop.run_in_executor(
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
        ragforge_trace = None
        retrieved_doc_texts: list[str] = []
        messages_with_context: list[Any] = []
        tool_executed_successfully = False

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

        # ── 3. Module & Tool Context Initialization ──────────────
        VALID_MODULES = {"ragforge", "localbuddy", "watchtower", "streamsync", "tunelab", "analytics"}
        main_module = inp.module if inp.module in VALID_MODULES else "localbuddy"
        module_context = _MODULE_CONTEXTS.get(main_module, "")
        
        if inp.system_location:
            module_context += (
                f"\n\nUSER LOCATION: The user is currently located in {inp.system_location}. "
                f"When the user says 'my location', 'my city', 'here', or 'where I am', "
                f"always use '{inp.system_location}' as the location argument.\n"
            )
        # web_search_enabled is checked via tools_list; not needed as a local flag
        
        from src.core.tool_registry import tool_registry
        tools_list = tool_registry.get_tool_definitions()

        # Inject Web Search & Weather tools if enabled (Legacy injection for backward compatibility with prompt)
        # Note: These are now properly registered in CoreModule as well.

        # Pre-process module context with tools
        if tools_list:
            import json
            tools_def = json.dumps(tools_list, indent=2)
            module_context += (
                f"\n\nAVAILABLE TOOLS:\n{tools_def}\n\n"
                "To call a tool, you MUST output a JSON block EXCLUSIVELY in this format:\n"
                "```json\n"
                "{\n"
                "  \"name\": \"<tool_name>\",\n"
                "  \"arguments\": {<tool_args>}\n"
                "}\n"
                "```\n"
            )

        # ── 4. Direct Execution (no IRA blueprint overhead) ─────────
        # Small models (2B-8B) can't reliably generate JSON plans.
        # Skip the planning LLM call entirely and use the user's
        # original message as the single task — saves 8-15s per query.
        blueprint = IRABlueprint(
            goal=inp.message,
            tasks=[BlueprintTask(
                id="task_1", description=inp.message, module=main_module,
            )],
        )
        _trace("planning", {"goal": blueprint.goal, "tasks": 1})

        t_task = time.perf_counter()

        # ── 4a. RAGForge / Analytics: CognitiveRAG with user's original query
        if main_module in ("ragforge", "analytics") and self.vector_store:
            active_docs = inp.context.get("active_docs", [])
            source_filter = (
                active_docs[0] if len(active_docs) == 1 else active_docs
            )
            cognitive = self._get_cognitive_rag()
            
            # Fetch prompt variant from evolution manager if available
            rag_variant = "v1"
            res_instance = getattr(self, "_researcher_instance", None)
            if res_instance is not None and getattr(res_instance, "manager", None):
                rag_variant = getattr(res_instance.manager.current_genome, "rag_prompt_variant", "v1")
                
            answer, retrieved_doc_texts_raw, ragforge_trace = (
                cognitive.think_and_answer(
                    query=inp.message,  # always the user's original question
                    source_filter=source_filter,
                    prompt_variant=rag_variant
                )
            )
            response_text = answer
            if isinstance(retrieved_doc_texts_raw, list):
                retrieved_doc_texts = [
                    getattr(d, "page_content", str(d))
                    for d in retrieved_doc_texts_raw
                ]

        # ── 4b. Evolution / Optimization tasks
        elif (
            "optimize" in inp.message.lower()
            or "evolution" in inp.message.lower()
        ):
            researcher = self._get_aether_researcher()
            b_type = "rag"
            if (
                "tune" in inp.message.lower()
                or "learning" in inp.message.lower()
            ):
                b_type = "learning"

            if b_type == "rag":
                from src.modules.ragforge.benchmarker import RAGBenchmarker
                from src.modules.ragforge.history_manager import (
                    RAGHistoryManager,
                )
                rm = RAGHistoryManager(self.settings)
                bench = RAGBenchmarker(
                    self._get_cognitive_rag(), 
                    rm, 
                    self.vector_store.embeddings if self.vector_store else None
                )
                b_fn = bench.run_suite
            else:
                from src.learning.benchmarker import BitNetBenchmarker
                from src.learning.bitnet_trainer import BitNetTrainer
                trainer = BitNetTrainer(self.settings, self.replay_buffer)
                bench = BitNetBenchmarker(trainer)
                b_fn = bench.run_sprint

            record = asyncio.run_coroutine_threadsafe(
                researcher.run_evolution_cycle(b_fn), self.loop,
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
        else:
            # Fetch prompt variant from evolution manager if available
            sys_variant = "v1"
            res_instance = getattr(self, "_researcher_instance", None)
            if res_instance is not None and getattr(res_instance, "manager", None):
                sys_variant = getattr(res_instance.manager.current_genome, "system_prompt_variant", "v1")
            sys_prompt = _SYSTEM_PROMPT_VARIANTS.get(sys_variant, _SYSTEM_PROMPT_VARIANTS["v1"])
            
            messages_with_context = [
                SystemMessage(
                    content=sys_prompt + "\n\n" + module_context,
                ),
                *memory[1:],
            ]
            response_text = self._run_llm_sync(messages_with_context)

        _trace("execution", {
            "latency_ms": (time.perf_counter() - t_task) * 1000,
            "module": main_module,
        })
        _trace(f"module_{main_module}", {"latency_ms": (time.perf_counter() - t0) * 1000, "response_preview": response_text[:100]})

        # Tool Execution Loop
        tool_executed_successfully = False
        import re
        
        logger.debug("--- [Intent Engine LLM Output] ---")
        logger.debug(str(response_text))
        logger.debug("---")

        # Look for markdown JSON blocks, or just the first JSON-like dictionary it outputs
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{.*\"(?:tool_name|name|tool|function)\".*\})', response_text, re.DOTALL)
            
        if json_match:
            try:
                import json
                import types
                tool_call_json = json_match.group(1)

                logger.debug(
                    "Matched Intent JSON: %s", tool_call_json,
                )
                call_data = json.loads(tool_call_json)

                # Support both "tool_name", "name", and "tool" keys
                tool_name = (
                    call_data.get("tool_name")
                    or call_data.get("name")
                    or call_data.get("tool")
                )

                if not tool_name and "function" in call_data:
                    tool_name = call_data["function"].get("name")
                    tool_args = call_data["function"].get(
                        "arguments", {},
                    )
                else:
                    tool_args = call_data.get("arguments", {})

                # ── Guard: skip if tool name is None / empty ──
                if not tool_name or tool_name == "None":
                    logger.warning(
                        "LLM emitted a tool call with no name "
                        "— skipping tool execution",
                    )
                else:
                    from src.core.tool_registry import tool_registry

                    mock_state = types.SimpleNamespace(
                        replay_buffer=self.replay_buffer,
                        settings=self.settings,
                    )

                    tool_output = str(
                        tool_registry.execute_tool(
                            tool_name, tool_args, state=mock_state,
                        ),
                    )

                    if tool_output:
                        tool_calls.append({
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": tool_output,
                        })

                        causal_nodes.append({
                            "id": f"tool_{tool_name}",
                            "data": {
                                "args": tool_args,
                                "result": tool_output[:100],
                            },
                            "ts": time.perf_counter(),
                        })

                        if not messages_with_context:
                            messages_with_context = list(memory)

                        messages_with_context.append(
                            AIMessage(content=response_text),
                        )
                        messages_with_context.append(
                            SystemMessage(content=(
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
                                "it clearly and concisely."
                            )),
                        )

                        t1 = time.perf_counter()
                        response_text = self._run_llm_sync(
                            messages_with_context,
                        )
                        tool_executed_successfully = True
                        _trace(
                            f"tool_execution_{tool_name}",
                            {
                                "result": str(tool_output)[:50],
                                "latency_ms": (
                                    time.perf_counter() - t1
                                ) * 1000,
                            },
                        )
                    else:
                        response_text = (
                            f"[Intent Engine Error] Tool "
                            f"'{tool_name}' was not recognized or "
                            "is disabled in the current context."
                        )
            except Exception as e:
                logger.error(
                    "Intent Engine Tool execution failed: %s", e,
                )
                response_text = (
                    "[Intent Engine Error] Failed to execute "
                    f"the requested system tool: {e}"
                )

        # ── 4. Optional Deep Reasoning Reflection Pass ─────────────
        deep_reasoning_enabled = bool(inp.context.get("deep_reasoning", False))
        if deep_reasoning_enabled and not isinstance(self._llm, MockLLM) and not tool_executed_successfully:
            # Run a second, internal reflection pass to improve the draft answer.
            # To keep latency manageable on 8GB machines, we restrict the reflection
            # context to just: system + current user message + draft answer.
            reflection_system = SystemMessage(
                content=(
                    "You have already produced the draft answer below. "
                    "Now carefully reflect on it: check the logic, fill in any missing steps, "
                    "correct mistakes, and improve clarity and structure. "
                    "Do NOT mention that you are revising your answer or that this is a second pass — "
                    "just respond with the improved final answer."
                )
            )
            reflection_messages = [
                SystemMessage(content=_SYSTEM_PROMPT + "\n\n" + module_context),
                HumanMessage(content=inp.message),
                AIMessage(content=response_text),
                reflection_system,
            ]
            t_reflect = time.perf_counter()
            improved = self._run_llm_sync(reflection_messages, temperature=0.3)
            _trace(
                "deep_reflection",
                {"latency_ms": (time.perf_counter() - t_reflect) * 1000},
            )
            if improved:
                response_text = improved
        
        # ── 5. Post-flight faithfulness score ─────────────────────
        # When a tool ran, the response is grounded in real data — automatically score it higher.
        # For RAGForge responses, use SAMR-lite (semantic cosine faithfulness).
        # For pure LLM responses to other modules, use the heuristic estimator.
        if tool_executed_successfully:
            faithfulness_score = 0.95  # Tool-backed responses are always grounded
        elif main_module == "ragforge" and retrieved_doc_texts and response_text:
            # SAMR-lite: semantic faithfulness check for RAG answers
            try:
                from src.modules.ragforge.samr_lite import run_samr_lite
                samr_result = run_samr_lite(
                    answer=response_text,
                    retrieved_docs=retrieved_doc_texts,
                    embedding_function=self.vector_store.embeddings,
                )
                faithfulness_score = samr_result.get("faithfulness_score", 0.5)
                _trace("samr_lite", samr_result)
                
                # Prepend the reasoning trace if available (ONLY for ragforge)
                if ragforge_trace and ragforge_trace.reasoning_chain:
                    formatted_cot = (
                        "<think>\n"
                        f"{ragforge_trace.reasoning_chain}\n"
                        "</think>\n\n"
                    )
                    response_text = formatted_cot + response_text

                # Append SAMR badge to answer when low confidence
                if samr_result.get("alert_triggered"):
                    icon = samr_result.get("alert_icon", "⚠️")
                    note = samr_result.get("interpretation", "")
                    response_text += f"\n\n{icon} **SAMR Faithfulness Notice:** {note}"
            except Exception as samr_err:
                logger.warning("SAMR-lite failed (non-fatal): %s", samr_err)
                faithfulness_score = _estimate_faithfulness(inp.message, response_text)
        else:
            faithfulness_score = _estimate_faithfulness(inp.message, response_text)
        _trace("faithfulness", {"score": faithfulness_score})

        # Block low-faithfulness outputs — EXCEPT for RAGForge:
        # RAGForge uses SAMR-lite which appends a visible ⚠️ warning to the answer
        # when confidence is low. Hard-blocking RAGForge prevents users from getting
        # any value from their uploaded documents — the opposite of our goal.
        # Colosseum blocking is reserved for tool-calling modules (WatchTower, etc.)
        # where a bad faithfulness score means the AI made up a tool result.
        ragforge_samr_active = (main_module == "ragforge" and retrieved_doc_texts)
        if faithfulness_score < self.settings.silicon_colosseum_min_faithfulness and not ragforge_samr_active:
            post_decision = self.colosseum.evaluate_request_sync({
                "session_id": inp.session_id,
                "module": "output_filter",
                "message": response_text,
                "faithfulness_score": faithfulness_score,
                "tool_call_count": len(tool_calls),
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

        # ── 6. Log successful Intent Tools to Replay Buffer ───────
        if tool_executed_successfully and faithfulness_score >= self.settings.silicon_colosseum_min_faithfulness and self.replay_buffer:
            try:
                # Add to Replay Buffer for continual learning (OPLoRA)
                asyncio.run_coroutine_threadsafe(
                    self.replay_buffer.record(
                        session_id=inp.session_id,
                        module=main_module,
                        prompt=inp.message,
                        response=response_text,
                        tool_calls=tool_calls,
                        faithfulness_score=faithfulness_score
                    ),
                    self.loop
                )
                _trace("replay_buffer_append", {"status": "success", "tool_calls": len(tool_calls)})
            except Exception as e:
                logger.error("Failed to append tool trace to replay buffer: %s", e)

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
            blueprint=blueprint,
        )

    async def stream(self, inp: MetaAgentInput) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields token-by-token chunks for WebSocket streaming.
        Incorporates IRA recursive logic: Planning -> Task Execution (with thinking tokens) -> Synthesis.
        """
        async with self._lock:
            if isinstance(self._llm, MockLLM):
                # ... (MockLLM logic)
                memory = self._get_or_create_memory(inp.session_id)
                memory.append(HumanMessage(content=inp.message))
                text = self._llm.generate(memory)
                memory.append(AIMessage(content=text))
                yield {"type": "token", "content": text}
                return

            memory = self._get_or_create_memory(inp.session_id)
            memory.append(HumanMessage(content=inp.message))
            
            # ── 1. Module & Tool Context Initialization ──────────────
            VALID_MODULES = {"ragforge", "localbuddy", "watchtower", "streamsync", "tunelab", "analytics"}
            main_module = inp.module if inp.module in VALID_MODULES else "localbuddy"
            module_context = _MODULE_CONTEXTS.get(main_module, "")
            
            if inp.system_location:
                module_context += (
                    f"\n\nUSER LOCATION: The user is currently located in {inp.system_location}. "
                    f"When the user says 'my location', 'my city', 'here', or 'where I am', "
                    f"always use '{inp.system_location}' as the location argument.\n"
                )
                
            from src.core.tool_registry import tool_registry
            tools_list = tool_registry.get_tool_definitions()
            if tools_list:
                import json
                tools_def = json.dumps(tools_list, indent=2)
                module_context += (
                    f"\n\nAVAILABLE TOOLS:\n{tools_def}\n\n"
                    "To call a tool, you MUST output a JSON block EXCLUSIVELY in this format:\n"
                    "```json\n"
                    "{\n"
                    "  \"name\": \"<tool_name>\",\n"
                    "  \"arguments\": {<tool_args>}\n"
                    "}\n"
                    "```\n"
                )

            # ── 2. Direct execution (no IRA planning overhead) ──────
            # For RAG: run CognitiveRAG synchronously, then yield result
            if main_module in ("ragforge", "analytics") and self.vector_store:
                active_docs = inp.context.get("active_docs", [])
                sf = active_docs[0] if len(active_docs) == 1 else active_docs
                cog = self._get_cognitive_rag()
                msg = inp.message  # bind for closure
                
                # Fetch prompt variant from evolution manager if available
                rag_variant = "v1"
                res_instance = getattr(self, "_researcher_instance", None)
                if res_instance is not None and getattr(res_instance, "manager", None):
                    rag_variant = getattr(res_instance.manager.current_genome, "rag_prompt_variant", "v1")

                answer, _, ragforge_trace = await self.loop.run_in_executor(
                    None,
                    lambda _c=cog, _q=msg, _sf=sf, _rv=rag_variant: _c.think_and_answer(
                        query=_q, source_filter=_sf, prompt_variant=_rv
                    ),
                )

                # Yield thinking trace FIRST so frontend shows it before the answer
                full_response = ""
                if ragforge_trace and ragforge_trace.reasoning_chain:
                    think_block = (
                        "<think>\n"
                        f"{ragforge_trace.reasoning_chain}\n"
                        "</think>\n\n"
                    )
                    yield {"type": "token", "content": think_block}
                    full_response += think_block

                # Then yield the answer
                yield {"type": "token", "content": answer}
                full_response += answer
                memory.append(AIMessage(content=full_response))
                return

            # ── 3. General LLM streaming (LocalBuddy, etc.) ─────────
            
            # Fetch prompt variant from evolution manager if available
            sys_variant = "v1"
            res_instance = getattr(self, "_researcher_instance", None)
            if res_instance is not None and getattr(res_instance, "manager", None):
                sys_variant = getattr(res_instance.manager.current_genome, "system_prompt_variant", "v1")
            sys_prompt = _SYSTEM_PROMPT_VARIANTS.get(sys_variant, _SYSTEM_PROMPT_VARIANTS["v1"])
            
            messages_with_context = [
                SystemMessage(
                    content=sys_prompt + "\n\n" + module_context,
                ),
                *memory[1:],
            ]
            
            prompt = _messages_to_prompt(messages_with_context)
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            loop = self.loop

            def _producer() -> None:
                max_retries = 3
                delay: float = 1.0
                last_err: Optional[Exception] = None
                full_response_buffer: str = ""  # Buffer for tokens in case of retry
                
                for attempt in range(max_retries):
                    try:
                        with self._sync_lock:  # threading.Lock: safe to use in a thread
                            for tok in self._llm(
                                prompt,
                                max_tokens=self.settings.bitnet_max_tokens,
                                temperature=self.settings.bitnet_temperature,
                                top_p=self.settings.bitnet_top_p,
                                stop=["<|im_end|>"],
                                stream=True,
                            ):
                                chunk: str = tok["choices"][0]["text"]
                                if chunk:
                                    full_response_buffer = str(full_response_buffer) + str(chunk)
                                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                        # If stream completes without error, break retry loop
                        break 
                    except Exception as e:
                        last_err = e
                        logger.warning(f"Streaming LLM inference attempt {attempt+1} failed: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(float(delay))
                            delay = float(delay) * 2.0
                            # Clear buffer for next attempt, or decide to keep it
                            full_response_buffer = "" 
                        else:
                            # Exhausted retries, send error token
                            error_msg = f" [⚠️ Stream Interrupted after {max_retries} attempts: {last_err}]"
                            asyncio.run_coroutine_threadsafe(queue.put(error_msg), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop.run_in_executor(None, _producer)  # fire-and-forget
            full_response = []
            while True:
                tok = await queue.get()
                if tok is None:
                    break
                full_response.append(tok)
                yield {"type": "token", "content": tok}

            # Strip unexecuted tool call JSON from the accumulated response
            # This prevents raw JSON blobs like {"name": "...", "arguments": {...}}
            # from appearing in the chat when the LLM hallucinates a tool call.
            import re as _re
            joined = "".join(full_response)
            cleaned = _re.sub(
                r'```(?:json)?\s*\{[^}]*"(?:name|tool_name|tool|function)"[^}]*\}\s*```',
                '',
                joined,
                flags=_re.DOTALL,
            ).strip()
            # Also strip bare JSON tool calls (no markdown fences)
            cleaned = _re.sub(
                r'\{\s*"(?:name|tool_name|tool|function)"\s*:.*?"arguments"\s*:\s*\{.*?\}\s*\}',
                '',
                cleaned,
                flags=_re.DOTALL,
            ).strip()
            if not cleaned:
                cleaned = "I wasn't able to process that request. Could you rephrase?"
            memory.append(AIMessage(content=cleaned))


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
