# AetherForge v1.0 — src/core/orchestrator.py
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.modules.base import BaseModule

logger = structlog.get_logger("aetherforge.core.orchestrator")


class ForensicOrchestrator:
    """
    The next-generation replacement for MetaAgent.
    Stateless, modular, and designed for 100% causal observability.
    """

    def __init__(self, modules: list[BaseModule], settings: Any = None, llm: Any = None) -> None:
        self.modules = {m.name: m for m in modules}
        self.settings = settings
        self._llm = llm
        logger.info("ForensicOrchestrator initialized", modules=list(self.modules.keys()))

    def _sanitize_tool_result(self, result: object) -> str:
        """Sanitize tool output to prevent prompt injection or model confusion."""
        if result is None:
            return "None"
        text = str(result)
        text = text.replace("<|im_start|>", "[IM_START]").replace("<|im_end|>", "[IM_END]")
        if len(text) > 8000:
            # Shielding from linter's inability to resolve str.__getitem__
            text_str: str = str(text)
            text = text_str[:8000] + "... [Truncated for stability]"
        return text

    async def run(self, session_id: str, module_name: str, message: str) -> dict[str, Any]:
        """
        The primary execution entry point using Forensic AI protocols.
        """
        if module_name not in self.modules:
            raise ValueError(f"Unknown module: {module_name}")

        module = self.modules[module_name]
        logger.audit("Forensic Protocol Start", session_id=session_id, module=module_name)

        # 1. Generate Grammar for active module tools
        from src.core.grammar import GrammarGenerator

        tools = module.get_tool_definitions()
        grammar = GrammarGenerator.generate_tool_grammar(tools)

        # 2. Call LLM with Grammar (Constraint Pass)
        # In Shadow Mode, we might not have the LLM yet if it's held by MetaAgent
        # We'll assume LLM is provided via Container or AppState
        if not self._llm:
            return {"response": "[Forensic] LLM not available in this context."}

        # 3. Formulate Prompt
        system_prompt = (
            f"You are AetherForge Forensic Orchestrator. {module.system_prompt_extension}"
        )
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

        # 4. Generate Tool Call (FORCED by GBNF)
        try:
            # Synchronous LLM call wrapped in executor (legacy-compatible)
            import json

            loop = asyncio.get_event_loop()

            raw_result = await loop.run_in_executor(
                None, lambda: self._llm(prompt, grammar=grammar, max_tokens=512, temperature=0.0)
            )
            tool_call = json.loads(raw_result["choices"][0]["text"])

            # 5. Execute Tool
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            raw_tool_result = module.execute_tool(tool_name, tool_args)
            tool_result = self._sanitize_tool_result(raw_tool_result)

            # 6. Synthesis Pass (Multi-Pass)
            # Use a different grammar or just free-form text for synthesis
            synth_prompt = (
                f"{prompt}{raw_result['choices'][0]['text']}<|im_end|>\n"
                f"<|im_start|>system\nTool Result: {tool_result}\n"
                f"Synthesize the final answer based on this data.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            synth_result = await loop.run_in_executor(
                None, lambda: self._llm(synth_prompt, max_tokens=1024, temperature=0.7)
            )
            final_response = synth_result["choices"][0]["text"].strip()

            return {
                "session_id": session_id,
                "response": final_response,
                "tool_calls": [tool_call],
                "metadata": {"tool_result": tool_result},
                "causal_graph": {
                    "nodes": [
                        {"id": "intake", "label": "User Request"},
                        {"id": "reasoning", "label": "GBNF Tool Selection"},
                        {"id": "execution", "label": f"Tool: {tool_name}"},
                        {"id": "synthesis", "label": "Multi-Pass Response"},
                    ],
                    "edges": [
                        {"source": "intake", "target": "reasoning", "label": "Analyzed"},
                        {
                            "source": "reasoning",
                            "target": "execution",
                            "label": f"Called {tool_name}",
                        },
                        {"source": "execution", "target": "synthesis", "label": "Data Grounded"},
                    ],
                },
            }

        except Exception as e:
            logger.exception("Forensic Execution failed", error=str(e))
            return {"response": f"Forensic Orchestration Error: {str(e)}"}

    async def stream(self, session_id: str, module_name: str, message: str) -> AsyncGenerator[str, None]:
        """
        Streaming version of the Forensic protocol.
        Yields tokens for the synthesis pass.
        """
        if module_name not in self.modules:
            yield {"type": "error", "content": f"Unknown module: {module_name}"}
            return

        module = self.modules[module_name]

        # 1. Grammar Generation
        from src.core.grammar import GrammarGenerator

        tools = module.get_tool_definitions()
        grammar = GrammarGenerator.generate_tool_grammar(tools)

        if not self._llm:
            yield {"type": "token", "content": "[Forensic] LLM not available."}
            return

        # 2. Prompt Formulation
        system_prompt = (
            f"You are AetherForge Forensic Orchestrator. {module.system_prompt_extension}"
        )
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

        try:
            import json

            loop = asyncio.get_event_loop()

            # 3. Tool Selection (Constraint Pass)
            raw_result = await loop.run_in_executor(
                None, lambda: self._llm(prompt, grammar=grammar, max_tokens=512, temperature=0.0)
            )
            tool_call = json.loads(raw_result["choices"][0]["text"])

            # 4. Tool Execution
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            raw_tool_result = module.execute_tool(tool_name, tool_args)
            tool_result = self._sanitize_tool_result(raw_tool_result)

            # 5. Synthesis Pass (Streaming)
            synth_prompt = (
                f"{prompt}{raw_result['choices'][0]['text']}<|im_end|>\n"
                f"<|im_start|>system\nTool Result: {tool_result}\n"
                f"Synthesize the final answer based on this data.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _producer() -> None:
                try:
                    for tok in self._llm(
                        synth_prompt, max_tokens=1024, temperature=0.7, stream=True
                    ):
                        chunk = tok["choices"][0]["text"]
                        if chunk:
                            asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(f" [Stream Error: {e}]"), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop.run_in_executor(None, _producer)

            while True:
                tok = await queue.get()
                if tok is None:
                    break
                yield {"type": "token", "content": tok}

            # Final completion chunk
            yield {"type": "done", "latency_ms": 0}  # Latency set by caller

        except Exception as e:
            logger.exception("Forensic Streaming failed", error=str(e))
            yield {"type": "error", "content": str(e)}
