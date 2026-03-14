from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from typing import Any

_GENERIC_PREFIXES = ("ui-session-",)
_ATTACHMENT_RE = re.compile(r"\[attachment:([^\]]+)\]", re.IGNORECASE)


def resolve_session_id(raw_id: str, module: str) -> tuple[str, bool]:
    """Return (internal_session_id, is_new_session)."""
    is_new = any(raw_id.startswith(prefix) for prefix in _GENERIC_PREFIXES)
    base = str(uuid.uuid4()) if is_new else raw_id

    if not base.startswith(f"{module}:"):
        return f"{module}:{base}", is_new
    return base, is_new


def split_reasoning_trace(text: str) -> tuple[str | None, str]:
    """Extract a visible reasoning block from <think>...</think> output."""
    open_tag = "<think>"
    close_tag = "</think>"
    start = text.find(open_tag)
    if start == -1:
        return None, text.strip()

    before = text[:start].strip()
    after_open = text[start + len(open_tag) :]
    end = after_open.find(close_tag)
    if end == -1:
        reasoning = after_open.strip() or None
        return reasoning, before

    reasoning = after_open[:end].strip() or None
    answer = f"{before}\n{after_open[end + len(close_tag):]}".strip()
    return reasoning, answer


def extract_attachment_names(text: str) -> list[str]:
    """Return unique attachment filenames embedded in a response."""
    seen: set[str] = set()
    attachments: list[str] = []
    for raw_name in _ATTACHMENT_RE.findall(text or ""):
        name = raw_name.strip()
        if name and name not in seen:
            seen.add(name)
            attachments.append(name)
    return attachments


def merge_attachment_names(*groups: Iterable[str]) -> list[str]:
    """Merge attachment groups while preserving order."""
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            name = str(item).strip()
            if name and name not in seen:
                seen.add(name)
                merged.append(name)
    return merged


def normalize_citations(items: Iterable[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Normalize citation dictionaries into a stable UI-friendly shape."""
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip() or "Unknown source"
        page = item.get("page")
        if page in ("", None):
            page = None
        section = str(item.get("section", "")).strip() or None
        snippet = str(item.get("snippet", "")).strip() or None
        kind = str(item.get("kind", "document")).strip() or "document"
        label = str(item.get("label", "")).strip() or None

        entry: dict[str, Any] = {
            "source": source,
            "page": page,
            "section": section,
            "snippet": snippet,
            "kind": kind,
        }
        if label:
            entry["label"] = label

        dedupe_key = (source, page, section, snippet, kind, label)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(entry)

    return normalized


def build_visible_reasoning_trace(
    *,
    module: str,
    message: str,
    answer_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    existing: str | None = None,
) -> str | None:
    """Guarantee a user-visible reasoning summary for LocalBuddy turns."""
    if existing:
        return existing
    if module != "localbuddy":
        return None

    steps: list[str] = [f"Goal: {_compact_text(message, 160)}"]
    tool_calls = tool_calls or []
    citations = citations or []

    if tool_calls:
        tool_names = ", ".join(
            str(call.get("name", "tool")) for call in tool_calls if call.get("name")
        )
        steps.append(f"Approach: I inspected local data with {tool_names}.")
    elif citations:
        steps.append("Approach: I grounded the answer in the local document evidence.")
    else:
        steps.append("Approach: I used the conversation context and generated a direct answer.")

    if citations:
        steps.append(f"Evidence: {len(citations)} source reference(s) are attached below.")

    if answer_text:
        steps.append(f"Deliverable: {_compact_text(answer_text, 180)}")

    return "\n".join(f"- {step}" for step in steps)


def _compact_text(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."
