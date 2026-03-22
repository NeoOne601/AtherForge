# tests/test_coherence_gate.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for the Coherence Gate (SA-06)
# ─────────────────────────────────────────────────────────────────
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.guardrails.coherence_gate import (
    extract_significant_numbers,
    numbers_from_trace,
    verify_calc_response,
    NumberVerificationError,
    is_calc_route,
)


class TestExtractSignificantNumbers:
    def test_basic_extraction(self):
        text = "The displacement is 25,839 tonnes at draft 8.17m"
        nums = extract_significant_numbers(text)
        assert "25839" in nums
        assert "8.17" in nums

    def test_excludes_single_digits(self):
        text = "Step 1: found 5 rows"
        nums = extract_significant_numbers(text)
        assert not nums  # 1 and 5 are <=9

    def test_excludes_years(self):
        text = "Published in 2024 with displacement 25839 tonnes"
        nums = extract_significant_numbers(text)
        assert "25839" in nums
        assert "2024" not in nums


class TestNumbersFromTrace:
    def test_dict_trace(self):
        trace = {"value": 25839.5, "draft_low": 8.0, "draft_high": 8.2}
        nums = numbers_from_trace(trace)
        assert "25839.5" in nums
        assert "8.0" in nums

    def test_nested_trace(self):
        trace = {"results": {"disp": {"value": 25839}, "tpc": {"value": 43.2}}}
        nums = numbers_from_trace(trace)
        assert "25839" in nums
        assert "43.2" in nums

    def test_string_trace(self):
        trace = "displacement at 8.17m = 25,839 tonnes"
        nums = numbers_from_trace(trace)
        assert "25839" in nums


class TestVerifyCalcResponse:
    def test_clean_response_passes(self):
        """Response using only trace numbers → no error."""
        response = "The displacement at 8.17m draft is 25,839 tonnes."
        trace = {"value": 25839, "draft": 8.17, "unit": "tonnes"}
        # Should NOT raise
        verify_calc_response(response, trace)

    def test_invented_number_blocked(self):
        """Response with a made-up number → raises error."""
        response = "The displacement is 99,999 tonnes at 8.17m draft."
        trace = {"value": 25839, "draft": 8.17, "unit": "tonnes"}
        with pytest.raises(NumberVerificationError) as exc_info:
            verify_calc_response(response, trace)
        assert "99999" in exc_info.value.unauthorized

    def test_tolerance_allowed(self):
        """Response with rounded version of trace number → passes."""
        response = "The displacement is approximately 25,840 tonnes."
        trace = {"value": 25839.5, "draft": 8.17, "unit": "tonnes"}
        # 25840 is within tolerance of 25839.5
        verify_calc_response(response, trace)

    def test_non_calc_route_skipped(self):
        """SYNTHESIS route → verify not called (is_calc_route returns False)."""
        assert is_calc_route("synthesis") is False
        assert is_calc_route("explain") is False
        assert is_calc_route("procedure") is False

    def test_calc_routes_detected(self):
        """All calc routes should be detected."""
        assert is_calc_route("table_lookup") is True
        assert is_calc_route("multi_lookup") is True
        assert is_calc_route("interpolate") is True
        assert is_calc_route("unit_convert") is True


class TestEdgeCases:
    def test_empty_response(self):
        """Empty response should not raise."""
        verify_calc_response("", {"value": 100})

    def test_empty_trace(self):
        """Empty trace with number in response → raises error."""
        with pytest.raises(NumberVerificationError):
            verify_calc_response("The value is 500 tonnes.", {})

    def test_integer_rounding(self):
        """Integer version of float should be acceptable."""
        response = "The TPC is 43 tonnes per cm."
        trace = {"value": 43.2, "unit": "t/cm"}
        verify_calc_response(response, trace)


if __name__ == "__main__":
    # Simple runner for non-pytest environments
    test_classes = [
        TestExtractSignificantNumbers,
        TestNumbersFromTrace,
        TestVerifyCalcResponse,
        TestEdgeCases,
    ]
    passed = 0
    failed = 0
    for cls in test_classes:
        for method_name in dir(cls):
            if method_name.startswith("test_"):
                try:
                    getattr(cls(), method_name)()
                    print(f"  ✓ {cls.__name__}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {cls.__name__}.{method_name}: {e}")
                    failed += 1
    print(f"\n{'✅' if failed == 0 else '❌'} {passed} passed, {failed} failed")
