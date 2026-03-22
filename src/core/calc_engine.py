# AetherForge v1.0 — src/core/calc_engine.py
# ─────────────────────────────────────────────────────────────────
# Deterministic Calculation Engine
#
# ALL arithmetic happens here. The LLM receives the result and trace,
# then writes an explanation. The LLM never performs the calculation.
#
# Core invariant: LLMs EXPLAIN. Deterministic engines CALCULATE.
# Citations PROVE. Any number that reaches the user must trace to:
# (a) a row in this structured SQLite data store, OR
# (b) a verbatim value in a retrieved document chunk.
# Otherwise: BLOCK. Never warn-and-deliver.
# ─────────────────────────────────────────────────────────────────
import sqlite3
import re
from pathlib import Path


class CalcEngine:
    """
    Deterministic calculation engine.
    ALL arithmetic happens here. The LLM receives the result and trace,
    then writes an explanation. The LLM never performs the calculation.
    """

    def __init__(self, db_path: str) -> None:
        self.db = str(Path(db_path).resolve())

    # ── Core math ────────────────────────────────────────────────────

    def linear_interpolate(
        self, d: float, d1: float, d2: float, v1: float, v2: float
    ) -> float:
        """Pure linear interpolation. No LLM. No ambiguity."""
        if d2 == d1:
            return round(v1, 3)
        frac = (d - d1) / (d2 - d1)
        return round(v1 + frac * (v2 - v1), 3)

    # ── Table lookups ────────────────────────────────────────────────

    def lookup_hydrostatic(
        self,
        vessel_id: str,
        draft: float,
        column: str = "displacement",
    ) -> dict:
        """
        Return interpolated value for any hydrostatic column at the given draft.
        Returns: {"value": float, "unit": str, "trace": dict}
        Raises: ValueError if draft is outside the table range.
        """
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT draft, {column} FROM hydrostatic "
            f"WHERE vessel_id=? AND {column} IS NOT NULL ORDER BY draft",
            (vessel_id,)
        ).fetchall()
        conn.close()

        if not rows:
            raise ValueError(
                f"No hydrostatic data for vessel '{vessel_id}' column '{column}'"
            )

        rows = [(r["draft"], r[column]) for r in rows]

        # Exact match
        exact = [r for r in rows if r[0] == draft]
        if exact:
            return {
                "value": exact[0][1],
                "unit": _unit_for(column),
                "trace": {
                    "method": "exact_match",
                    "draft": draft,
                    column: exact[0][1],
                    "density": "salt_water_1.025",
                }
            }

        # Find bracketing rows
        lower_candidates = [r for r in rows if r[0] <= draft]
        upper_candidates = [r for r in rows if r[0] >= draft]

        if not lower_candidates or not upper_candidates:
            min_d = rows[0][0]
            max_d = rows[-1][0]
            raise ValueError(
                f"Draft {draft}m is outside the table range "
                f"({min_d}m – {max_d}m) for vessel '{vessel_id}'"
            )

        lower = max(lower_candidates, key=lambda r: r[0])
        upper = min(upper_candidates, key=lambda r: r[0])

        fraction = round((draft - lower[0]) / (upper[0] - lower[0]), 6)
        result = self.linear_interpolate(draft, lower[0], upper[0], lower[1], upper[1])

        return {
            "value": result,
            "unit": _unit_for(column),
            "trace": {
                "method": "linear_interpolation",
                "target_draft_m": draft,
                "lower_row": {"draft_m": lower[0], column: lower[1]},
                "upper_row": {"draft_m": upper[0], column: upper[1]},
                "fraction": fraction,
                "formula": f"{lower[1]} + {fraction} × ({upper[1]} - {lower[1]}) = {result}",
                "density_assumption": "salt_water_1.025",
            }
        }

    def lookup_all_hydrostatic(self, vessel_id: str, draft: float) -> dict:
        """Return all hydrostatic columns at once for a given draft."""
        columns = ["displacement", "tpc", "mtc", "km", "lcb", "lcf"]
        results = {}
        errors = {}
        for col in columns:
            try:
                results[col] = self.lookup_hydrostatic(vessel_id, draft, col)
            except Exception as e:
                errors[col] = str(e)
        return {"results": results, "errors": errors, "draft_m": draft}

    # ── Corrections ──────────────────────────────────────────────────

    def apply_fw_correction(self, displacement_sw: float) -> dict:
        """Fresh water correction: Δ_FW = Δ_SW × (1.000 / 1.025)"""
        ratio = round(1.000 / 1.025, 6)
        displacement_fw = round(displacement_sw * ratio, 2)
        return {
            "value": displacement_fw,
            "unit": "tonnes",
            "trace": {
                "method": "fresh_water_sg_correction",
                "input_sw_tonnes": displacement_sw,
                "sg_salt_water": 1.025,
                "sg_fresh_water": 1.000,
                "ratio": ratio,
                "formula": f"{displacement_sw} × (1.000 / 1.025) = {displacement_fw}",
            }
        }

    def apply_sg_correction(self, displacement_sw: float, sg_dock: float) -> dict:
        """Dock water correction: Δ_dock = Δ_SW × (sg_dock / 1.025)"""
        ratio = round(sg_dock / 1.025, 6)
        displacement_dock = round(displacement_sw * ratio, 2)
        return {
            "value": displacement_dock,
            "unit": "tonnes",
            "trace": {
                "method": "dock_water_sg_correction",
                "input_sw_tonnes": displacement_sw,
                "sg_salt_water": 1.025,
                "sg_dock_water": sg_dock,
                "ratio": ratio,
                "formula": f"{displacement_sw} × ({sg_dock} / 1.025) = {displacement_dock}",
            }
        }

    # ── Utility ──────────────────────────────────────────────────────

    def extract_draft_from_query(self, query: str) -> float | None:
        """Extract a decimal draft value (e.g. 8.17m) from a query string."""
        pattern = re.compile(r'\b(\d+\.\d{1,3})\s*m\b', re.IGNORECASE)
        match = pattern.search(query)
        return float(match.group(1)) if match else None

    def extract_sg_from_query(self, query: str) -> float | None:
        """Extract a specific gravity value (e.g. RD 1.015) from a query string."""
        pattern = re.compile(r'\b(?:rd|sg|r\.d\.|s\.g\.)\s*[=:]?\s*(\d+\.\d+)\b', re.IGNORECASE)
        match = pattern.search(query)
        return float(match.group(1)) if match else None


def _unit_for(column: str) -> str:
    units = {
        "displacement": "tonnes",
        "tpc": "t/cm",
        "mtc": "t·m/cm",
        "km": "m",
        "lcb": "m",
        "lcf": "m",
    }
    return units.get(column, column)
