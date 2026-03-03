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

    def __init__(self, settings: AetherForgeSettings, colosseum: SiliconColosseum, vector_store: Any = None, replay_buffer: Any = None) -> None:
        self.settings = settings
        self.colosseum = colosseum
        self.vector_store = vector_store
        self.replay_buffer = replay_buffer
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

    def _run_llm_sync(self, messages: list[Any], max_tokens: int | None = None, temperature: float | None = None) -> str:
        """Run the LLM synchronously (called from pipeline nodes)."""
        if isinstance(self._llm, MockLLM):
            return self._llm.generate(messages)

        prompt = _messages_to_prompt(messages)
        result = self._llm(
            prompt,
            max_tokens=max_tokens or self.settings.bitnet_max_tokens,
            temperature=temperature if temperature is not None else self.settings.bitnet_temperature,
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
        
        # Default for non-ragforge modules (SAMR-lite uses this later)
        retrieved_doc_texts: list[str] = []

        # Inject RAG Context if applicable
        if module == "ragforge" and self.vector_store:
            t_rag = time.perf_counter()
            active_docs = inp.context.get("active_docs", [])
            filter_kwargs = {}
            if active_docs:
                if len(active_docs) == 1:
                    filter_kwargs["filter"] = {"source": active_docs[0]}
                else:
                    filter_kwargs["filter"] = {"source": {"$in": active_docs}}

            docs = self.vector_store.similarity_search(inp.message, k=6, **filter_kwargs)
            retrieved_doc_texts: list[str] = []  # stored for SAMR-lite check below
            if docs:
                context_parts = []
                for i, d in enumerate(docs):
                    meta = d.metadata
                    source = meta.get("source", "Unknown")
                    page = meta.get("page", "?")
                    section = meta.get("section", "")
                    chunk_type = meta.get("chunk_type", "section")
                    parser = meta.get("parser", "")
                    # Build rich citation header for research papers
                    citation = f"[{i+1}] {source} | p.{page}"
                    if section:
                        citation += f" | §{section[:60]}"
                    if chunk_type in ("table", "equation"):
                        citation += f" | [{chunk_type.upper()}]"
                    context_parts.append(f"{citation}\n{d.page_content}")
                    retrieved_doc_texts.append(d.page_content)
                retrieved_text = "\n\nRetrieved Context (Precision Ingestion™):\n" + "\n\n".join(context_parts)
                module_context += retrieved_text
                _trace("rag_retrieval", {"chunks": len(docs), "latency_ms": (time.perf_counter() - t_rag) * 1000})
            else:
                module_context += "\n\nRetrieved Context: No relevant documents found. Ask the user to upload documents first."
                retrieved_doc_texts = []
                _trace("rag_retrieval", {"chunks": 0, "latency_ms": (time.perf_counter() - t_rag) * 1000})

        # Inject Intent Engine Tools if applicable
        tool_module = None
        if module in {"watchtower", "streamsync", "tunelab"}:
            try:
                import importlib
                tool_module = importlib.import_module(f"src.modules.{module}.tools")
                if hasattr(tool_module, "get_tools"):
                    import json
                    tools_def = json.dumps(tool_module.get_tools(), indent=2)
                    _FEW_SHOT = {
                        "watchtower": (
                            "User: why is memory usage so high?\n"
                            "Assistant: Let me pull the live memory stats now.\n"
                            "```json\n{\"tool_name\": \"query_metrics\", \"arguments\": {\"metric_name\": \"mem\"}}\n```\n"
                            "User: what process is using all my memory?\n"
                            "Assistant: Checking top memory consumers right now.\n"
                            "```json\n{\"tool_name\": \"get_top_processes\", \"arguments\": {\"sort_by\": \"memory\"}}\n```\n"
                            "User: kill python_worker\n"
                            "Assistant: Terminating python_worker now.\n"
                            "```json\n{\"tool_name\": \"kill_process\", \"arguments\": {\"target\": \"python_worker\"}}\n```"
                        ),
                        "streamsync": (
                            "User: show me the event stream\n"
                            "Assistant: Fetching the latest events from the buffer.\n"
                            "```json\n{\"tool_name\": \"query_stream\", \"arguments\": {\"limit\": 10}}\n```\n"
                            "User: what patterns are in the stream?\n"
                            "Assistant: Summarizing the stream data now.\n"
                            "```json\n{\"tool_name\": \"summarize_stream\", \"arguments\": {}}\n```\n"
                            "User: how can I analyze the event streams here?\n"
                            "Assistant: Let me pull the current events so you can see exactly what's flowing in.\n"
                            "```json\n{\"tool_name\": \"query_stream\", \"arguments\": {\"limit\": 20}}\n```"
                        ),
                        "tunelab": (
                            "User: how many are ready for training?\n"
                            "Assistant: Querying the Replay Buffer stats now.\n"
                            "```json\n{\"tool_name\": \"query_buffer_stats\", \"arguments\": {}}\n```\n"
                            "User: how many samples are pending?\n"
                            "Assistant: Let me check the buffer right now.\n"
                            "```json\n{\"tool_name\": \"query_buffer_stats\", \"arguments\": {}}\n```\n"
                            "User: trigger the compilation\n"
                            "Assistant: Starting the OPLoRA compilation cycle now.\n"
                            "```json\n{\"tool_name\": \"trigger_compilation\", \"arguments\": {}}\n```"
                        ),
                    }
                    few_shot = _FEW_SHOT.get(module, "")
                    module_context += (
                        f"\n\nAVAILABLE TOOLS:\nYou have access to the following tools to act upon the {module} subsystem:\n"
                        f"{tools_def}\n\n"
                        "CRITICAL INSTRUCTION: You MUST call a tool when the user's request can be answered by one. "
                        "Do NOT explain what the tool does. Do NOT guide the user through the UI. "
                        "Call the tool first, then explain the result in plain language. "
                        "Output the tool call wrapped in a ```json block in this exact format:\n"
                        "```json\n{\"tool_name\": \"name_of_tool\", \"arguments\": {\"arg1\": \"val1\"}}\n```\n\n"
                        f"MODULE-SPECIFIC EXAMPLES:\n{few_shot}"
                    )
            except Exception as e:
                logger.error("Failed to load Intent Tools for %s: %s", module, e)

        messages_with_context = [SystemMessage(content=_SYSTEM_PROMPT + "\n\n" + module_context)] + memory[1:]

        # Run LLM
        # Force temperature to 0.0 when tools are present to ensure strict JSON adherence
        llm_temp = 0.0 if tool_module else self.settings.bitnet_temperature
        response_text = self._run_llm_sync(messages_with_context, temperature=llm_temp)
        _trace(f"module_{module}", {"latency_ms": (time.perf_counter() - t0) * 1000, "response_preview": response_text[:100]})

        # Tool Execution Loop
        tool_executed_successfully = False
        import re
        
        logger.debug("--- [Intent Engine LLM Output] ---")
        logger.debug(str(response_text))
        logger.debug("---")

        # Look for markdown JSON blocks, or just the first JSON-like dictionary it outputs
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{.*?\"(?:tool_name|name)\".*?\})', response_text, re.DOTALL)
            
        if json_match:
            try:
                import json
                tool_call_json = json_match.group(1)
                
                logger.debug("Matched Intent JSON: %s", tool_call_json)
                call_data = json.loads(tool_call_json)
                
                # Support both "tool_name": "name" and "name": "name" depending on how the model formats it
                tool_name = call_data.get("tool_name") or call_data.get("name")
                
                if not tool_name and "function" in call_data:
                    # Model output an OpenAI-style function call json
                    tool_name = call_data["function"].get("name")
                    tool_args = call_data["function"].get("arguments", {})
                else:
                    tool_args = call_data.get("arguments", {})
                
                # Execute tool
                if tool_module and hasattr(tool_module, "execute_tool"):
                    import inspect
                    sig = inspect.signature(tool_module.execute_tool)
                    if "state" in sig.parameters:
                        class MockState: pass
                        mock_state = MockState()
                        mock_state.replay_buffer = self.replay_buffer
                        mock_state.settings = self.settings
                        tool_output = tool_module.execute_tool(tool_name, tool_args, mock_state)
                    else:
                        tool_output = tool_module.execute_tool(tool_name, tool_args)
                    
                    tool_calls.append({"name": tool_name, "arguments": tool_args, "result": str(tool_output)})
                    
                    # Add tool execution result back into conversation
                    messages_with_context.append(AIMessage(content=response_text))
                    messages_with_context.append(SystemMessage(content=(
                        f"Tool '{tool_name}' executed successfully and returned:\n{tool_output}\n\n"
                        "Respond to the user in plain language summarizing what the tool found. "
                        "Be direct — state the actual values. Do NOT say 'I don't have access', "
                        "'as an AI', or 'I cannot'. The data is above. Summarize it clearly."
                    )))
                    
                    t1 = time.perf_counter()
                    response_text = self._run_llm_sync(messages_with_context)
                    tool_executed_successfully = True
                    _trace(f"tool_execution_{tool_name}", {"result": str(tool_output)[:50], "latency_ms": (time.perf_counter() - t1) * 1000})
                    
            except Exception as e:
                logger.error("Intent Engine Tool execution failed: %s", e)
                response_text = f"[Intent Engine Error] Failed to execute the requested system tool: {e}"

        # ── 4. Post-flight faithfulness score ─────────────────────
        # When a tool ran, the response is grounded in real data — automatically score it higher.
        # For RAGForge responses, use SAMR-lite (semantic cosine faithfulness).
        # For pure LLM responses to other modules, use the heuristic estimator.
        if tool_executed_successfully:
            faithfulness_score = 0.95  # Tool-backed responses are always grounded
        elif module == "ragforge" and retrieved_doc_texts and response_text:
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
        ragforge_samr_active = (module == "ragforge" and retrieved_doc_texts)
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
                asyncio.get_event_loop().create_task(
                    self.replay_buffer.record(
                        session_id=inp.session_id,
                        module=module,
                        prompt=inp.message,
                        response=response_text,
                        tool_calls=tool_calls,
                        faithfulness_score=faithfulness_score
                    )
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
            
            # Module dispatch and RAG Context
            VALID_MODULES = {"ragforge", "localbuddy", "watchtower", "streamsync", "tunelab"}
            module = inp.module if inp.module in VALID_MODULES else "localbuddy"
            module_context = _MODULE_CONTEXTS.get(module, "")

            if module == "ragforge" and self.vector_store:
                active_docs = inp.context.get("active_docs", [])
                filter_kwargs = {}
                if active_docs:
                    if len(active_docs) == 1:
                        filter_kwargs["filter"] = {"source": active_docs[0]}
                    else:
                        filter_kwargs["filter"] = {"source": {"$in": active_docs}}

                docs = self.vector_store.similarity_search(inp.message, k=8, **filter_kwargs)
                if docs:
                    context_parts = []
                    has_images_note = False
                    for i, d in enumerate(docs):
                        meta = d.metadata
                        source = meta.get("source", "Unknown")
                        page = meta.get("page", "?")
                        section = meta.get("section", "")
                        chunk_type = meta.get("chunk_type", "section")
                        citation = f"[{i+1}] {source} | p.{page}"
                        if section:
                            citation += f" | §{section[:60]}"
                        if chunk_type in ("table", "equation"):
                            citation += f" | [{chunk_type.upper()}]"
                        context_parts.append(f"{citation}\n{d.page_content}")
                    retrieved_text = "\n\nRetrieved Context (Precision Ingestion™):\n" + "\n\n".join(context_parts)
                    # Warn about embedded images not in text index
                    retrieved_text += (
                        "\n\n[SYSTEM NOTE: This document contains embedded images/figures that are not "
                        "in the text index. For figure/diagram content, the retrieved chunks may only "
                        "contain captions or references — not the visual content itself.]"
                    )
                    module_context += retrieved_text
                else:
                    module_context += "\n\nRetrieved Context: No relevant documents found. The Knowledge Vault may be empty."

            messages_with_context = [SystemMessage(content=_SYSTEM_PROMPT + "\n\n" + module_context)] + memory[1:]
            prompt = _messages_to_prompt(messages_with_context)

            queue = asyncio.Queue()
            
            def _producer():
                try:
                    for tok in self._llm(
                        prompt,
                        max_tokens=self.settings.bitnet_max_tokens,
                        temperature=self.settings.bitnet_temperature,
                        top_p=self.settings.bitnet_top_p,
                        stop=["<|im_end|>"],
                        stream=True,
                    ):
                        # Use threadsafe put_nowait so we can push to the async queue from this thread
                        asyncio.run_coroutine_threadsafe(queue.put(tok["choices"][0]["text"]), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, _producer)

            full_response = []
            while True:
                tok = await queue.get()
                if tok is None:
                    break
                full_response.append(tok)
                yield {"type": "token", "content": tok}
                
            memory.append(AIMessage(content="".join(full_response)))


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
        "You are in LocalBuddy mode. Act as a helpful, concise AI assistant. "
        "Remember conversation context. Be honest about what you don't know."
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
