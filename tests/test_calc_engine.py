# AetherForge — tests/test_calc_engine.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for the deterministic CalcEngine.
# Verifies: exact lookup, interpolation, FW correction, SG correction.
# ─────────────────────────────────────────────────────────────────
import sqlite3
import pytest
from pathlib import Path

from src.core.calc_engine import CalcEngine


@pytest.fixture
def calc_engine(tmp_path):
    """Create a CalcEngine with sample hydrostatic data."""
    db_path = tmp_path / "test_structured.db"
    engine = CalcEngine(db_path=str(db_path))

    # Seed with sample hydrostatic data
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hydrostatic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vessel_id TEXT NOT NULL,
            draft REAL NOT NULL,
            displacement REAL,
            tpc REAL,
            mtc REAL,
            km REAL,
            lcb REAL,
            lcf REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Insert sample rows (approximate Primrose Ace data)
    sample_data = [
        ("HA", 7.00, 19500.0, 45.2, 180.0, 10.5, 85.0, 83.0),
        ("HA", 7.50, 21800.0, 46.1, 185.0, 10.3, 85.5, 83.5),
        ("HA", 8.00, 24200.0, 47.0, 190.0, 10.1, 86.0, 84.0),
        ("HA", 8.20, 25200.0, 47.4, 192.0, 10.0, 86.2, 84.2),
        ("HA", 8.50, 26800.0, 48.0, 195.0, 9.9, 86.5, 84.5),
        ("HA", 9.00, 29500.0, 49.1, 200.0, 9.7, 87.0, 85.0),
    ]
    for row in sample_data:
        conn.execute(
            "INSERT INTO hydrostatic (vessel_id, draft, displacement, tpc, mtc, km, lcb, lcf) "
            "VALUES (?,?,?,?,?,?,?,?)",
            row,
        )
    conn.commit()
    conn.close()

    return engine


class TestCalcEngine:
    """Tests for the deterministic CalcEngine."""

    def test_exact_draft_match(self, calc_engine):
        """Draft exactly equals a table row → returns that row's value."""
        result = calc_engine.lookup_hydrostatic("HA", 8.00, "displacement")
        assert result["value"] == 24200.0
        assert result["unit"] == "tonnes"
        assert result["trace"]["method"] == "exact_match"
        assert result["trace"]["draft"] == 8.00

    def test_interpolation(self, calc_engine):
        """Draft between two rows → correct linear interpolation."""
        # 8.17m is between 8.00 (24200) and 8.20 (25200)
        result = calc_engine.lookup_hydrostatic("HA", 8.17, "displacement")
        # Expected: 24200 + (8.17-8.00)/(8.20-8.00) * (25200-24200)
        #         = 24200 + 0.85 * 1000 = 25050
        assert result["trace"]["method"] == "linear_interpolation"
        assert result["value"] == 25050.0
        assert result["unit"] == "tonnes"
        assert "formula" in result["trace"]
        assert result["trace"]["lower_row"]["draft_m"] == 8.00
        assert result["trace"]["upper_row"]["draft_m"] == 8.20

    def test_fw_correction(self, calc_engine):
        """Fresh water correction: Δ_FW = Δ_SW × (1.000 / 1.025)"""
        result = calc_engine.apply_fw_correction(25839.0)
        # Expected: 25839 × 0.97561... ≈ 25208.78-25208.79 (float rounding)
        expected = round(25839.0 * (1.000 / 1.025), 2)
        assert abs(result["value"] - expected) < 1.0
        assert result["unit"] == "tonnes"
        assert result["trace"]["method"] == "fresh_water_sg_correction"
        assert result["trace"]["sg_salt_water"] == 1.025
        assert result["trace"]["sg_fresh_water"] == 1.000

    def test_sg_correction(self, calc_engine):
        """Dock water correction: Δ_dock = Δ_SW × (sg_dock / 1.025)"""
        result = calc_engine.apply_sg_correction(25839.0, 1.010)
        # Expected: 25839 × (1.010 / 1.025) ≈ 25460.87-25460.88 (float rounding)
        expected = round(25839.0 * (1.010 / 1.025), 2)
        assert abs(result["value"] - expected) < 1.0
        assert result["unit"] == "tonnes"
        assert result["trace"]["method"] == "dock_water_sg_correction"
        assert result["trace"]["sg_dock_water"] == 1.010

    def test_out_of_range_draft(self, calc_engine):
        """Draft outside data range raises ValueError."""
        with pytest.raises(ValueError, match="outside the table range"):
            calc_engine.lookup_hydrostatic("HA", 15.0, "displacement")

    def test_no_data_vessel(self, calc_engine):
        """Unknown vessel raises ValueError."""
        with pytest.raises(ValueError, match="No hydrostatic data"):
            calc_engine.lookup_hydrostatic("UNKNOWN", 8.0, "displacement")

    def test_lookup_all_hydrostatic(self, calc_engine):
        """lookup_all_hydrostatic returns all columns at once."""
        result = calc_engine.lookup_all_hydrostatic("HA", 8.00)
        assert "results" in result
        assert "displacement" in result["results"]
        assert result["results"]["displacement"]["value"] == 24200.0
        assert result["draft_m"] == 8.00

    def test_extract_draft_from_query(self, calc_engine):
        """Extract draft value from natural language query."""
        assert calc_engine.extract_draft_from_query("displacement at 8.17m") == 8.17
        assert calc_engine.extract_draft_from_query("draft of 7.00m") == 7.00
        assert calc_engine.extract_draft_from_query("no draft here") is None

    def test_extract_sg_from_query(self, calc_engine):
        """Extract specific gravity from natural language query."""
        assert calc_engine.extract_sg_from_query("RD 1.015") == 1.015
        assert calc_engine.extract_sg_from_query("sg 1.010") == 1.010
        assert calc_engine.extract_sg_from_query("no sg here") is None
