# AetherForge v1.0 — src/modules/ragforge/calc_engine.py
# ─────────────────────────────────────────────────────────────────
# Deterministic Calculation Engine
# 
# Parses structured table data, stores it in SQLite, and provides
# strict linear interpolation lookups to prevent LLM hallucination.
# ─────────────────────────────────────────────────────────────────
import sqlite3
import pandas as pd
from pathlib import Path
import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger("aetherforge.ragforge.calc_engine")

class CalcEngine:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the structured SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hydrostatic_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vessel_id TEXT,
                    draft REAL,
                    displacement REAL,
                    tpc REAL,
                    mtc REAL,
                    km REAL,
                    lcb REAL,
                    lcf REAL,
                    UNIQUE(vessel_id, draft)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generic_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT,
                    table_name TEXT,
                    headers_json TEXT,
                    rows_json TEXT
                )
            """)

    def ingest_hydrostatic_table(self, vessel_id: str, df: pd.DataFrame):
        """
        Parses a pandas DataFrame extracted by Docling.
        Maps columns heuristically to hydrostatic parameters and inserts into SQLite.
        """
        if df.empty:
            return

        # Normalize column names for heuristic matching
        cols = [str(c).lower().replace('\n', '').replace(' ', '') for c in df.columns]
        
        # We need at least a draft and displacement column to be useful
        draft_col = next((c for c in cols if "draft" in c or "mean" in c or "d(m)" in c), None)
        disp_col = next((c for c in cols if "displ" in c or "volume" in c or "w(t)" in c), None)

        if not draft_col or not disp_col:
            logger.debug("Table does not look like a hydrostatic table, skipping structured ingest.")
            return

        tpc_col = next((c for c in cols if "tpc" in c), None)
        mtc_col = next((c for c in cols if "mtc" in c), None)
        km_col = next((c for c in cols if "km" in c), None)
        lcb_col = next((c for c in cols if "lcb" in c), None)
        lcf_col = next((c for c in cols if "lcf" in c), None)

        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            for _, row in df.iterrows():
                try:
                    # Clean and parse numerical values
                    def _parse(val):
                        if pd.isna(val): return None
                        s = str(val).replace(',', '').strip()
                        try: return float(s)
                        except: return None

                    draft = _parse(row.iloc[cols.index(draft_col)])
                    disp = _parse(row.iloc[cols.index(disp_col)])

                    if draft is None or disp is None:
                        continue

                    tpc = _parse(row.iloc[cols.index(tpc_col)]) if tpc_col else None
                    mtc = _parse(row.iloc[cols.index(mtc_col)]) if mtc_col else None
                    km = _parse(row.iloc[cols.index(km_col)]) if km_col else None
                    lcb = _parse(row.iloc[cols.index(lcb_col)]) if lcb_col else None
                    lcf = _parse(row.iloc[cols.index(lcf_col)]) if lcf_col else None

                    conn.execute("""
                        INSERT OR REPLACE INTO hydrostatic_table 
                        (vessel_id, draft, displacement, tpc, mtc, km, lcb, lcf)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (vessel_id, draft, disp, tpc, mtc, km, lcb, lcf))
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to parse table row: {e}")
        
        logger.info(f"Ingested {inserted} hydrostatic rows for vessel {vessel_id}")

    def linear_interpolate(self, d: float, d1: float, d2: float, v1: float, v2: float) -> float:
        """Standard linear interpolation formula."""
        if d1 == d2:
            return v1
        return v1 + ((d - d1) / (d2 - d1)) * (v2 - v1)

    def lookup_table(self, vessel_id: str, draft: float) -> Dict[str, Any]:
        """
        Table lookup function mapped for Tiny Dancer routes.
        Finds exact or interpolates values at the given draft.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 1. Check exact match
            cur = conn.execute("SELECT * FROM hydrostatic_table WHERE vessel_id = ? AND draft = ?", (vessel_id, draft))
            exact = cur.fetchone()
            if exact:
                return dict(exact)

            # 2. Find lower bound
            cur = conn.execute("SELECT * FROM hydrostatic_table WHERE vessel_id = ? AND draft < ? ORDER BY draft DESC LIMIT 1", (vessel_id, draft))
            lower = cur.fetchone()

            # 3. Find upper bound
            cur = conn.execute("SELECT * FROM hydrostatic_table WHERE vessel_id = ? AND draft > ? ORDER BY draft ASC LIMIT 1", (vessel_id, draft))
            upper = cur.fetchone()

            if not lower or not upper:
                return {"error": f"Draft {draft} is out of bounds for vessel {vessel_id}"}
            
            # Interpolate all numerical columns
            result = {"vessel_id": vessel_id, "draft": draft, "interpolated": True, "bounds": {"lower": lower["draft"], "upper": upper["draft"]}}
            
            d1, d2 = lower["draft"], upper["draft"]
            for col in ["displacement", "tpc", "mtc", "km", "lcb", "lcf"]:
                if lower[col] is not None and upper[col] is not None:
                    result[col] = round(self.linear_interpolate(draft, d1, d2, lower[col], upper[col]), 3)

            return result
