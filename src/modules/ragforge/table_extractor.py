# AetherForge v1.0 — src/modules/ragforge/table_extractor.py
# ─────────────────────────────────────────────────────────────────
# Structured Table Extractor
#
# Extracts table data from Docling-parsed documents and writes
# typed rows to SQLite. This is the foundational layer: without
# structured data, the CalcEngine has nothing to query.
#
# Pipeline: Docling table objects → parse → classify → SQLite rows
# ─────────────────────────────────────────────────────────────────
import sqlite3
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def classify_table_domain(headers: list[str]) -> str:
    """Determine which domain a table belongs to based on its column headers."""
    header_text = " ".join(headers).lower()
    if any(w in header_text for w in ["draft", "displacement", "tpc", "mtc"]):
        return "hydrostatic"
    if any(w in header_text for w in ["tank", "capacity", "volume", "ullage"]):
        return "tank_capacity"
    if any(w in header_text for w in ["angle", "gz", "righting"]):
        return "gz_curve"
    if any(w in header_text for w in ["kn", "sin"]):
        return "kn_table"
    return "generic"


def parse_docling_table(table_obj: Any) -> tuple[list[str], list[dict]]:
    """
    Extract headers and typed row dicts from a Docling table object.
    Docling table objects expose: table.header_cells and table.data_rows.
    Each cell has a .text attribute.
    Returns (headers_list, rows_list) where each row is a dict.
    """
    headers = [cell.text.strip() for cell in table_obj.header_cells]
    # Normalise header names to valid column identifiers
    col_names = [
        h.lower()
         .replace(" ", "_")
         .replace("(", "")
         .replace(")", "")
         .replace("/", "_per_")
         .replace(".", "_")
        for h in headers
    ]
    rows = []
    for data_row in table_obj.data_rows:
        row_dict: dict[str, Any] = {}
        for i, cell in enumerate(data_row):
            if i >= len(col_names):
                continue
            raw = cell.text.strip()
            try:
                row_dict[col_names[i]] = float(raw.replace(",", ""))
            except ValueError:
                row_dict[col_names[i]] = raw
        rows.append(row_dict)
    return headers, rows


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create all structured data tables if they don't exist."""
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tank_capacity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vessel_id TEXT NOT NULL,
            tank_name TEXT,
            ullage REAL,
            volume REAL,
            mass REAL,
            lcg REAL,
            vcg REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gz_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vessel_id TEXT NOT NULL,
            displacement REAL,
            angle REAL,
            gz REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hydrostatic_vessel ON hydrostatic(vessel_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hydrostatic_draft ON hydrostatic(vessel_id, draft)")
    conn.commit()


def extract_tables_to_sqlite(
    doc: Any,
    vessel_id: str,
    db_path: str,
) -> dict[str, int]:
    """
    Main entry point. Called from ragforge_indexer.py after Docling parses a document.

    Iterates all pages in doc. When a page has .tables (Docling table objects),
    extracts them to typed SQLite rows.

    Returns a summary dict: {"hydrostatic": N, "tank_capacity": M, "generic": K}
    """
    db_path = str(Path(db_path).resolve())
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    summary: dict[str, int] = {}

    for page in doc.pages:
        if not hasattr(page, "tables") or not page.tables:
            continue
        for table_obj in page.tables:
            try:
                headers, rows = parse_docling_table(table_obj)
                if not headers or not rows:
                    continue
                domain = classify_table_domain(headers)
                summary[domain] = summary.get(domain, 0) + len(rows)

                if domain == "hydrostatic":
                    _insert_hydrostatic(conn, vessel_id, rows)
                elif domain == "tank_capacity":
                    _insert_tank_capacity(conn, vessel_id, rows)
                elif domain == "gz_curve":
                    _insert_gz_curve(conn, vessel_id, rows)
                # generic tables: skip structured storage, let text chunker handle them
            except Exception as e:
                logger.warning(
                    "Table extraction failed on page: %s", e
                )
                continue

    conn.commit()
    conn.close()
    return summary


def _insert_hydrostatic(conn: sqlite3.Connection, vessel_id: str, rows: list[dict]) -> None:
    for row in rows:
        draft = row.get("draft") or row.get("draft_m") or row.get("mean_draft")
        if draft is None:
            continue
        conn.execute(
            """INSERT INTO hydrostatic
               (vessel_id, draft, displacement, tpc, mtc, km, lcb, lcf)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                vessel_id,
                float(draft),
                row.get("displacement") or row.get("displacement_tonnes"),
                row.get("tpc") or row.get("tonnes_per_cm"),
                row.get("mtc") or row.get("moment_to_change_trim"),
                row.get("km") or row.get("km_m"),
                row.get("lcb") or row.get("lcb_m"),
                row.get("lcf") or row.get("lcf_m"),
            )
        )


def _insert_tank_capacity(conn: sqlite3.Connection, vessel_id: str, rows: list[dict]) -> None:
    for row in rows:
        conn.execute(
            "INSERT INTO tank_capacity (vessel_id, tank_name, ullage, volume, mass, lcg, vcg) VALUES (?,?,?,?,?,?,?)",
            (
                vessel_id,
                row.get("tank_name") or row.get("tank") or "unknown",
                row.get("ullage"),
                row.get("volume") or row.get("volume_m3"),
                row.get("mass") or row.get("mass_tonnes"),
                row.get("lcg"),
                row.get("vcg"),
            )
        )


def _insert_gz_curve(conn: sqlite3.Connection, vessel_id: str, rows: list[dict]) -> None:
    for row in rows:
        angle = row.get("angle") or row.get("angle_deg") or row.get("heel")
        gz = row.get("gz") or row.get("gz_m") or row.get("righting_lever")
        if angle is None or gz is None:
            continue
        conn.execute(
            "INSERT INTO gz_curve (vessel_id, displacement, angle, gz) VALUES (?,?,?,?)",
            (vessel_id, row.get("displacement"), float(angle), float(gz))
        )
