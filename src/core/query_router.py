# AetherForge v1.0 — src/core/query_router.py
# ─────────────────────────────────────────────────────────────────
# Deterministic Query Router
#
# Pure keyword + regex routing — no subprocess, no LLM, no network.
# Classifies queries BEFORE any LLM invocation so calculation
# queries bypass the LLM entirely and go to the CalcEngine.
#
# Routes:
#   TABLE_LOOKUP     → single-column interpolation (e.g. "displacement at 8.17m")
#   MULTI_LOOKUP     → all columns at one draft
#   INTERPOLATE      → explicit interpolation request
#   UNIT_CONVERT     → FW/dock-water density correction
#   EXPLAIN          → conceptual question (→ RAG)
#   PROCEDURE        → procedural question (→ RAG)
#   SYNTHESIS        → default full RAG pipeline
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger("aetherforge.core.query_router")

# ── Draft extractor ──────────────────────────────────────────────
_DRAFT_RE = re.compile(
    r'\b(\d+\.\d{1,3})\s*(?:m\b|metres?\b|meters?\b)',
    re.IGNORECASE,
)


class QueryRoute(str, Enum):
    TABLE_LOOKUP = "table_lookup"
    MULTI_LOOKUP = "multi_lookup"
    INTERPOLATE = "interpolate"
    UNIT_CONVERT = "unit_convert"
    EXPLAIN = "explain"
    PROCEDURE = "procedure"
    SYNTHESIS = "synthesis"


# ── Keyword sets ─────────────────────────────────────────────────
_CALC_KEYWORDS = frozenset([
    "displacement", "tpc", "tonnes per cm", "mtc", "moment to change",
    "km", "lcb", "lcf", "deadweight",
])
_CONVERT_KEYWORDS = frozenset([
    "fresh water", "dock water", "fw", "dw", "density", "rd", "sg",
    "relative density", "specific gravity",
])
_ALL_PARTICULARS = frozenset([
    "all particulars", "all stability", "all hydrostatic",
    "stability particulars", "hydrostatic data", "hydrostatic particulars",
])
_EXPLAIN_KEYWORDS = frozenset([
    "what is", "what does", "explain", "define", "meaning of",
    "describe", "difference between", "why is", "concept",
])
_PROCEDURE_KEYWORDS = frozenset([
    "how do i", "how to", "procedure", "steps to", "process for",
    "operate", "instructions",
])


def route_query(query: str) -> QueryRoute:
    """
    Classify a user query into a route. Pure function — no I/O, no subprocess.
    Called BEFORE any LLM invocation.
    """
    q = query.lower().strip()
    has_draft = bool(_DRAFT_RE.search(query))
    has_calc = any(kw in q for kw in _CALC_KEYWORDS)
    has_conv = any(kw in q for kw in _CONVERT_KEYWORDS)
    has_all = any(kw in q for kw in _ALL_PARTICULARS)
    has_expl = any(kw in q for kw in _EXPLAIN_KEYWORDS)
    has_proc = any(kw in q for kw in _PROCEDURE_KEYWORDS)

    # Conversion queries (fresh water / dock water)
    if has_draft and has_conv:
        return QueryRoute.UNIT_CONVERT
    if has_conv and not has_expl:
        return QueryRoute.UNIT_CONVERT

    # All stability particulars at one draft
    if has_draft and has_all:
        return QueryRoute.MULTI_LOOKUP

    # Explicit interpolation request
    if has_draft and "interpolat" in q:
        return QueryRoute.INTERPOLATE

    # Single column table lookup
    if has_draft and has_calc:
        return QueryRoute.TABLE_LOOKUP

    # Explanation requests (no draft number = conceptual question)
    if has_expl and not has_draft:
        return QueryRoute.EXPLAIN

    # Procedure requests
    if has_proc:
        return QueryRoute.PROCEDURE

    # Default: full RAG pipeline
    return QueryRoute.SYNTHESIS


def extract_draft(query: str) -> Optional[float]:
    """Extract draft value in metres from a query string."""
    m = _DRAFT_RE.search(query)
    return float(m.group(1)) if m else None


def extract_column(query: str) -> str:
    """Extract which hydrostatic column is being asked for. Default: displacement."""
    q = query.lower()
    if "tpc" in q or "tonnes per cm" in q:
        return "tpc"
    if "mtc" in q or "moment to change trim" in q:
        return "mtc"
    if " km" in q or "km " in q:
        return "km"
    if "lcb" in q:
        return "lcb"
    if "lcf" in q:
        return "lcf"
    return "displacement"  # default


def extract_sg(query: str) -> Optional[float]:
    """Extract specific gravity (RD/SG) from a query string."""
    m = re.search(r'(?:rd|sg)\s*[=:]?\s*(\d+\.\d+)', query, re.IGNORECASE)
    return float(m.group(1)) if m else None
