from src.chat_contract import (
    build_visible_reasoning_trace,
    extract_attachment_names,
    normalize_citations,
    resolve_session_id,
    split_reasoning_trace,
)
from src.core.grammar import GrammarGenerator


def test_split_reasoning_trace_complete_block():
    reasoning, answer = split_reasoning_trace(
        "<think>\nstep 1\nstep 2\n</think>\n\nFinal answer."
    )

    assert reasoning == "step 1\nstep 2"
    assert answer == "Final answer."


def test_split_reasoning_trace_without_block():
    reasoning, answer = split_reasoning_trace("Final answer only.")

    assert reasoning is None
    assert answer == "Final answer only."


def test_resolve_session_id_prefixes_module_and_preserves_real_id():
    session_id, is_new = resolve_session_id("abc123", "localbuddy")

    assert session_id == "localbuddy:abc123"
    assert is_new is False


def test_generate_tool_grammar_restricts_name_to_registered_tools():
    grammar = GrammarGenerator.generate_tool_grammar(
        [
            {"name": "search_web", "parameters": {"type": "object", "properties": {}}},
            {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        ]
    )

    assert 'tool_call ::= "{" ws "\\"name\\"" ws ":" ws tool_name' in grammar
    assert 'tool_name ::= "\\"search_web\\"" | "\\"get_weather\\""' in grammar


def test_extract_attachment_names_deduplicates_embedded_tags():
    attachments = extract_attachment_names(
        "See [attachment:chart.png] and [attachment:report.pdf] plus [attachment:chart.png]"
    )

    assert attachments == ["chart.png", "report.pdf"]


def test_build_visible_reasoning_trace_backfills_localbuddy_summary():
    reasoning = build_visible_reasoning_trace(
        module="localbuddy",
        message="Explain the PDF in the LiveFolder to a 10 year old.",
        answer_text="It explains how attention helps a model focus on important words.",
        tool_calls=[{"name": "analyze_data"}],
        citations=[{"source": "paper.pdf"}],
        existing=None,
    )

    assert reasoning is not None
    assert "Goal:" in reasoning
    assert "analyze_data" in reasoning
    assert "source reference" in reasoning


def test_normalize_citations_deduplicates_and_shapes_entries():
    citations = normalize_citations(
        [
            {"source": "paper.pdf", "page": 2, "section": "Intro", "snippet": "Hello"},
            {"source": "paper.pdf", "page": 2, "section": "Intro", "snippet": "Hello"},
        ]
    )

    assert citations == [
        {
            "source": "paper.pdf",
            "page": 2,
            "section": "Intro",
            "snippet": "Hello",
            "kind": "document",
        }
    ]
