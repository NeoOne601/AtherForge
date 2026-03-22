# tests/test_cognitiverag.py
# ─────────────────────────────────────────────────────────────────
# Tests for CognitiveRAG extract_cot and ThinkingTrace parsing.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import re


def extract_cot(text: str) -> tuple[str, str]:
    """
    Separate chain-of-thought reasoning from the final answer.

    Tries these patterns in order:
      A) <think>...</think> block (most reliable, from CognitiveRAG prompts)
      B) <answer>...</answer> block for the clean answer
      C) **Reasoning:**... followed by a blank line
      D) Lines starting with "Step N:" until a blank line separator
      E) Fallback: first 60% is thinking, last 40% is answer

    Returns:
        (thinking, clean_answer)
    """
    # ── Pattern A: <think>...</think> tags ──────────────────────
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        # Check for <answer> block
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            clean_answer = answer_match.group(1).strip()
        else:
            # Everything after </think> is the answer
            parts = re.split(r"</think>", text, flags=re.IGNORECASE)
            clean_answer = parts[-1].strip() if len(parts) > 1 else ""
        return thinking, clean_answer

    # ── Pattern B: **Reasoning:** prefix ────────────────────────
    reasoning_match = re.match(r"\*\*Reasoning:\*\*\s*(.*?)(?:\n\n|\Z)(.*)", text, re.DOTALL)
    if reasoning_match:
        thinking = reasoning_match.group(1).strip()
        clean_answer = reasoning_match.group(2).strip()
        if thinking:
            return thinking, clean_answer

    # ── Pattern C: Step N: lines ────────────────────────────────
    step_match = re.match(r"((?:Step\s+\d+:.*\n?)+)\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if step_match:
        thinking = step_match.group(1).strip()
        clean_answer = step_match.group(2).strip()
        if thinking and clean_answer:
            return thinking, clean_answer

    # ── Pattern D: 60/40 fallback ───────────────────────────────
    if len(text) > 100:
        split_point = int(len(text) * 0.6)
        # Try to split at a sentence boundary near the 60% mark
        boundary = text.rfind(". ", max(0, split_point - 50), min(len(text), split_point + 50))
        if boundary > 0:
            split_point = boundary + 1
        return text[:split_point].strip(), text[split_point:].strip()

    # Short text — no separation needed
    return "", text.strip()


# ═════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════

def test_extract_cot_think_tags():
    """Pattern A: <think>...</think> with <answer>...</answer>."""
    text = (
        "<think>Step 1: I looked at the hydrostatic table.\n"
        "Step 2: Found displacement at 8.17m.\n"
        "Step 3: Interpolated between 8.1 and 8.2.</think>\n"
        "<answer>The displacement at 8.17m is **25,050 tonnes** [1].</answer>"
    )
    thinking, answer = extract_cot(text)
    assert "Step 1" in thinking
    assert "hydrostatic" in thinking
    assert "25,050" in answer
    assert "<think>" not in answer
    assert "<answer>" not in answer


def test_extract_cot_think_tags_no_answer_block():
    """Pattern A fallback: <think>...</think> but no explicit <answer> tags."""
    text = (
        "<think>Checking evidence chunks for relevance.</think>\n"
        "The ship's displacement is 25,839 tonnes."
    )
    thinking, answer = extract_cot(text)
    assert "evidence chunks" in thinking
    assert "25,839" in answer


def test_extract_cot_reasoning_prefix():
    """Pattern B: **Reasoning:** prefix followed by blank line."""
    text = (
        "**Reasoning:** I examined the table on page 42 of the stability booklet.\n\n"
        "The displacement at draft 8.17m is 25,050 tonnes."
    )
    thinking, answer = extract_cot(text)
    assert "page 42" in thinking
    assert "25,050" in answer
    assert "**Reasoning:**" not in answer


def test_extract_cot_step_lines():
    """Pattern C: Step N: lines followed by answer."""
    text = (
        "Step 1: Located the hydrostatic table.\n"
        "Step 2: Cross-referenced with displacement column.\n"
        "Step 3: Applied interpolation.\n"
        "\n"
        "The displacement is 25,050 tonnes at 8.17m draft."
    )
    thinking, answer = extract_cot(text)
    assert "Step 1" in thinking
    assert "Step 3" in thinking
    assert "25,050" in answer


def test_extract_cot_plain_text_fallback():
    """Pattern D: plain text with no markers — 60/40 split."""
    text = "A" * 60 + ". " + "B" * 40 + ". " + "C" * 30
    thinking, answer = extract_cot(text)
    # Both parts should be non-empty
    assert len(thinking) > 0
    assert len(answer) > 0
    # Total should reconstruct (modulo whitespace)
    assert len(thinking) + len(answer) + 1 <= len(text) + 2  # +1 for space


def test_extract_cot_short_text_no_split():
    """Very short text should not be split."""
    text = "Hello world."
    thinking, answer = extract_cot(text)
    assert thinking == ""
    assert answer == "Hello world."


if __name__ == "__main__":
    test_extract_cot_think_tags()
    test_extract_cot_think_tags_no_answer_block()
    test_extract_cot_reasoning_prefix()
    test_extract_cot_step_lines()
    test_extract_cot_plain_text_fallback()
    test_extract_cot_short_text_no_split()
    print("✅ All 6 extract_cot tests passed!")
