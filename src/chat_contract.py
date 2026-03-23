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


# Matches tool calls like `{"name": "search_web", "arguments": ...}` with or without markdown
_TOOL_JSON_RE = re.compile(
    r"(?:```(?:json)?\s*)?\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}\s*(?:```)?",
    re.IGNORECASE | re.DOTALL,
)

# Broader pattern: bare JSON with "name"/"arguments" keys (no markdown fences)
_BARE_TOOL_JSON_RE = re.compile(
    r"\{[^{}]*\"name\"\s*:\s*\"(?:search_web|get_weather|write_vfs_note|read_vfs_note)[^{}]*\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)

# Boilerplate audit instructions that leak from GroundingAuditor
_AUDIT_BOILERPLATE_RE = re.compile(
    r"Output the (?:connected|corrected),?\s*citation-?(?:backed|rooted)\s*final answer[^.]*\.?\s*"
    r"(?:If the search results do not answer the query,?\s*say so honestly\.?)?\s*",
    re.IGNORECASE,
)


def sanitize_output(text: str, is_stream: bool = False) -> str:
    """Rigorous project-wide guard to strip protocol tags and stray tool JSON from user-facing text."""
    if not text:
        return ""
    # 1. Strip any lingering protocol tags (handle partials like </think)
    text = re.sub(r"</?think\s*>? ?", "", text, flags=re.IGNORECASE)
    
    # 2. Strip explicit tool JSON payloads that might have leaked
    text = _TOOL_JSON_RE.sub("", text)
    # 2b. Strip bare tool JSON (no markdown fences)
    text = _BARE_TOOL_JSON_RE.sub("", text)
    # 2c. Strip leaked audit boilerplate
    text = _AUDIT_BOILERPLATE_RE.sub("", text)
    
    # 3. Cleanup spacing and noise (PRESERVE NEWLINES AND STREAM CONTEXT)
    if is_stream:
        # In stream mode, we ONLY strip protocol tags. 
        # We MUST NOT strip spaces because they might be needed between chunks.
        return text
    
    # Only collapse multiple horizontal spaces, not vertical ones
    text = re.sub(r"[ \t]+", " ", text)
    # Strip any leading/trailing spaces on each line, but keep the lines
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def split_reasoning_trace(text: str) -> tuple[str | None, str]:
    """Extract a visible reasoning block from <think>...</think> output, returning sanitized answer."""
    open_tag = "<think>"
    close_tags = ["</think>", "</think"] # Support partial tags

    start = text.find(open_tag)
    if start == -1:
        # If no explicit tags, check if the model just started reasoning without the tag
        # (common with some small models that skip the prefix)
        return None, sanitize_output(text)

    before = text[:start].strip() # type: ignore
    after_open = text[start + len(open_tag):] # type: ignore
    
    # Look for any matching closing tag
    end = -1
    matched_tag_len = 0
    for tag in close_tags:
        idx = after_open.find(tag) # type: ignore
        if idx != -1:
            end = idx
            matched_tag_len = len(tag)
            break

    if end == -1:
        # If no closing tag, everything after the opening tag is currently reasoning
        reasoning = after_open.strip() or None # type: ignore
        return reasoning, sanitize_output(before)

    reasoning = after_open[:end].strip() or None # type: ignore
    
    # Extract what remains after the closing tag
    remaining = after_open[end + matched_tag_len:] # type: ignore
    # If the tag was partial e.g. "</think", it might be follow by ">"
    if remaining.startswith(">"):
        remaining = remaining[1:] # type: ignore
        
    # Crucial: If 'before' and 'remaining' are both empty/whitespace, then
    # the entire message was reasoning. Retain the reasoning for the UI.
    answer = f"{before}\n{remaining}"
    return reasoning, sanitize_output(answer)


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
    return collapsed[: limit - 3].rstrip() + "..."  # type: ignore[misc]
