from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from src.chat_contract import normalize_citations, resolve_session_id, split_reasoning_trace
from src.meta_agent import MetaAgentInput
from src.schemas import ChatResponse
from src.utils import safe_create_task

logger = structlog.get_logger("aetherforge.chat_turns")


def build_reasoning_summary(
    *,
    module: str,
    message: str,
    answer_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> str | None:
    tool_calls = tool_calls or []
    citations = citations or []
    if not message and not answer_text:
        return None

    steps: list[str] = [f"- Goal: {' '.join(message.split())[:180]}"]
    if module in {"ragforge", "analytics"}:
        if citations:
            steps.append(f"- Grounding: Used {len(citations)} document reference(s) from the indexed knowledge base.")
        else:
            steps.append("- Grounding: Checked the indexed document registry and available context.")
    elif tool_calls:
        tool_names = ", ".join(str(call.get("name", "tool")) for call in tool_calls if call.get("name"))
        steps.append(f"- Approach: Used the available local tools ({tool_names}).")
    else:
        steps.append("- Approach: Answered from the current local context and session history.")

    if citations:
        sources = ", ".join(sorted({str(item.get('source', 'Unknown source')) for item in citations})[:3])
        steps.append(f"- Evidence: Attached source references from {sources}.")
    if answer_text:
        compact_answer = " ".join(answer_text.split())
        if len(compact_answer) > 180:
            compact_answer = compact_answer[:177].rstrip() + "..."
        steps.append(f"- Result: {compact_answer}")
    return "\n".join(steps)


def generate_suggestions(
    *,
    module: str,
    answer_text: str,
    active_docs: list[str] | None = None,
    attachments: list[str] | None = None,
) -> list[str]:
    active_docs = active_docs or []
    attachments = attachments or []
    lowered = answer_text.lower()

    if "cannot find" in lowered or "no retrievable chunks" in lowered or "failed" in lowered:
        suggestions = [
            "Ask me to list the indexed documents and their ingest status.",
            "Ask me to re-check a specific file by name.",
            "If the file is scanned, ask for OCR/VLM extraction status.",
        ]
        if active_docs:
            suggestions.insert(0, f"Ask a broader question about {active_docs[0]}.")
        return suggestions[:4]

    if module == "ragforge":
        suggestions = [
            "Ask for a page-by-page summary of the selected document.",
            "Ask me to extract a table, checklist, or key figures.",
            "Ask for a comparison between two uploaded documents.",
        ]
        if attachments:
            suggestions.insert(0, "Ask me to explain the generated artifact or chart.")
        return suggestions[:4]

    if module == "analytics":
        return [
            "Ask me to generate a chart or table from the uploaded data.",
            "Ask for a markdown or PDF report export.",
            "Ask for anomalies, trends, or a column-by-column summary.",
        ]

    return [
        "Ask a follow-up question using the current context.",
        "Ask me to summarize the answer into action items.",
        "Ask me what the next useful step is.",
    ]


def bootstrap_session(state: Any, session_id: str, module: str, is_new: bool) -> None:
    session_store = state.session_store
    if is_new or not session_store.session_exists(session_id):
        session_store.create_session(module=module, session_id=session_id)
        return
    if session_id in state.meta_agent._session_memories:
        return
    try:
        restored = session_store.to_langchain_messages(session_id)
        if restored:
            state.meta_agent._session_memories[session_id] = restored
    except Exception as exc:
        logger.warning("Session hydration failed", session_id=session_id, error=str(exc))


def resolve_chat_session(raw_id: str, module: str) -> tuple[str, bool]:
    return resolve_session_id(raw_id, module)


async def persist_turn(
    state: Any,
    *,
    session_id: str,
    module: str,
    message: str,
    result: ChatResponse,
    wait_for_persist: bool = False,
) -> None:
    def _persist() -> None:
        try:
            meta = {
                "module": result.module,
                "latency_ms": result.latency_ms,
                "faithfulness_score": result.faithfulness_score,
                "reasoning_summary": result.reasoning_summary,
                "reasoning_trace": result.reasoning_trace,
                "answer_text": result.answer_text,
                "citations": [citation.model_dump() for citation in result.citations],
                "attachments": result.attachments,
                "suggestions": result.suggestions,
            }
            state.session_store.append_turn(
                session_id=session_id,
                user_content=message,
                assistant_content=result.response,
                assistant_metadata=meta,
            )
        except Exception as exc:
            logger.exception("Failed to persist session turn", session_id=session_id, error=str(exc))

    persist_coro = asyncio.to_thread(_persist)
    if wait_for_persist:
        await persist_coro
    else:
        safe_create_task(persist_coro, name="persist_session_turn")
    safe_create_task(
        state.replay_buffer.record(
            session_id=session_id,
            module=module,
            prompt=message,
            response=result.response,
            tool_calls=result.tool_calls,
            faithfulness_score=result.faithfulness_score,
        ),
        name="record_replay",
    )


async def execute_turn(
    state: Any,
    *,
    session_id: str,
    module: str,
    message: str,
    xray_mode: bool,
    context: dict[str, Any],
) -> ChatResponse:
    t0 = time.perf_counter()
    result = await state.meta_agent.run(
        MetaAgentInput(
            session_id=session_id,
            module=module,
            message=message,
            xray_mode=xray_mode,
            context=context,
            system_location=state.system_location,
        )
    )
    latency_ms = round(float((time.perf_counter() - t0) * 1000), 2)
    _reasoning, answer_text = split_reasoning_trace(result.response)
    answer_text = answer_text or result.response
    citations = normalize_citations(result.citations)
    reasoning_summary = build_reasoning_summary(
        module=module,
        message=message,
        answer_text=answer_text,
        tool_calls=result.tool_calls,
        citations=citations,
    )
    suggestions = generate_suggestions(
        module=module,
        answer_text=answer_text,
        active_docs=context.get("active_docs") if isinstance(context.get("active_docs"), list) else None,
        attachments=result.attachments,
    )
    return ChatResponse(
        session_id=session_id,
        response=answer_text,
        module=module,
        latency_ms=latency_ms,
        tool_calls=result.tool_calls,
        policy_decisions=result.policy_decisions,
        causal_graph=result.causal_graph,
        faithfulness_score=result.faithfulness_score,
        reasoning_summary=reasoning_summary,
        reasoning_trace=reasoning_summary,
        answer_text=answer_text,
        citations=citations,
        attachments=result.attachments,
        suggestions=suggestions,
    )


def iter_stream_chunks(text: str, chunk_size: int = 48) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
