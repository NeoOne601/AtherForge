# AetherForge v1.0 — src/guardrails/coherence_gate.py
# ─────────────────────────────────────────────────────────────────
# Coherence Gate — Number Verification for CalcEngine responses.
#
# Core invariant: LLMs explain. Deterministic engines calculate.
# Every number in an LLM explanation must trace back to the calc
# engine's output. Any invented number gets blocked.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from typing import Any


class NumberVerificationError(Exception):
    """Raised when the LLM response contains numbers not in the calc trace."""

    def __init__(self, unauthorized: set[str], response: str):
        self.unauthorized = unauthorized
        super().__init__(
            f"Response contains {len(unauthorized)} untraced number(s): {unauthorized}. "
            f"Response excerpt: {response[:200]}"
        )


def extract_significant_numbers(text: str) -> set[str]:
    """
    Extract all numeric values from text that could represent measurements.
    Excludes: page numbers, years (1900-2099), single digits 0-9.
    Includes: decimals, numbers with commas, numbers > 9.
    """
    raw = re.findall(r'\b\d+(?:[,]\d{3})*(?:\.\d+)?\b', text)
    significant: set[str] = set()
    for n in raw:
        clean = n.replace(',', '')
        try:
            val = float(clean)
            if val > 9 and not (1900 <= val <= 2099):
                significant.add(clean)
        except ValueError:
            pass
    return significant


def numbers_from_trace(trace: dict | list | Any) -> set[str]:
    """Recursively extract all numeric values from a trace dict."""
    if isinstance(trace, dict):
        nums: set[str] = set()
        for v in trace.values():
            nums |= numbers_from_trace(v)
        return nums
    elif isinstance(trace, (list, tuple)):
        nums = set()
        for item in trace:
            nums |= numbers_from_trace(item)
        return nums
    elif isinstance(trace, (int, float)):
        clean = str(trace).replace(',', '')
        return {clean}
    elif isinstance(trace, str):
        return extract_significant_numbers(trace)
    return set()


def verify_calc_response(
    llm_response: str,
    calc_trace: dict,
    tolerance: float = 0.01,
) -> None:
    """
    Verify that every significant number in llm_response exists in calc_trace.
    Raises NumberVerificationError if any unauthorized number is found.

    tolerance: allow numbers within this % of a trace value to pass.
    """
    response_numbers = extract_significant_numbers(llm_response)
    trace_numbers = numbers_from_trace(calc_trace)

    # Build a set of all acceptable values (exact + within tolerance)
    acceptable: set[str] = set()
    for tn in trace_numbers:
        acceptable.add(tn)
        try:
            val = float(tn)
            # Allow slight rounding differences
            acceptable.add(str(round(val, 1)))
            acceptable.add(str(round(val, 2)))
            acceptable.add(str(int(val)))
        except ValueError:
            pass

    # Check each response number
    unauthorized: set[str] = set()
    for rn in response_numbers:
        if rn in acceptable:
            continue
        # Check within tolerance
        try:
            rval = float(rn)
            within_tolerance = any(
                abs(rval - float(tn)) / max(abs(float(tn)), 1) < tolerance
                for tn in trace_numbers
                if tn.replace('.', '').replace('-', '').isdigit()
            )
            if not within_tolerance:
                unauthorized.add(rn)
        except ValueError:
            unauthorized.add(rn)

    if unauthorized:
        raise NumberVerificationError(unauthorized, llm_response)


def is_calc_route(route: str) -> bool:
    """Returns True for routes that go through the calc engine."""
    return route in {"table_lookup", "multi_lookup", "interpolate", "unit_convert"}
