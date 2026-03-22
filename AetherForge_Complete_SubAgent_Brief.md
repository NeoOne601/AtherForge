# AetherForge — Complete Sub-Agent Execution Brief
**Version 3.0 — Unified Architecture + UI Feature Implementation**
*For use with: Claude Opus 4.6 / Sonnet 4.6 in Gemini Antigravity*
*Repo: github.com/NeoOne601/AtherForge*

---

## HOW TO USE THIS DOCUMENT

This document contains **11 self-contained sub-agent prompts** that together
complete the full AetherForge redesign. Each prompt can be copied directly
into Antigravity as a task for a single sub-agent.

**Before starting:** Attach the files listed in each sub-agent's "Files to
attach" section. The sub-agent will read them before making changes.

**Execution order matters.** Follow the master plan in Part 2.

---

## PART 1 — PROJECT CONTEXT (READ BEFORE RUNNING ANY AGENT)

### What AetherForge Is

AetherForge is a Sovereign Intelligence OS — a perpetual-learning, fully local,
glass-box AI desktop application built with Tauri 2.1 (Rust + WebView),
React 18 (TypeScript frontend), FastAPI + LangGraph (Python 3.12 backend).
Runs 100% on Apple Silicon M1/M2/M3. No cloud. No external APIs.

### Current Stack (as of last audit)

| Layer | Currently Running | Target |
|---|---|---|
| LLM | BitNet 2B via llama-cpp-python (8k ctx) | ruvllm + Qwen2.5-7B (16k ctx) |
| Vector search | ChromaDB dense-only (BM25 dead at runtime) | RuVector GNN-HNSW hybrid |
| Query routing | None — all queries hit full LLM pipeline | Tiny Dancer semantic router |
| Calc engine | None — LLM does all arithmetic (hallucinates) | Deterministic SQLite interpolator |
| Table data | Text blobs from Docling OCR | Typed SQLite rows (draft, displacement…) |
| Faithfulness | SAMR-lite warns but delivers hallucinations | Prime Radiant blocks on energy > 0.70 |
| Learning | OPLoRA nightly batch only | SONA 3-tier + OPLoRA |
| CoT display | Reasoning text embedded inline in response | Collapsible "Thinking..." block |
| Suggestions | Tiles fill input only (user must press Enter) | Auto-submit on click |

### What Has Already Been Done (do not redo)

| Item | File | Status |
|---|---|---|
| BGE-M3 embedding upgrade | pyproject.toml | ✅ Done — 1024-dim, 8192-token context |
| Docling structural chunking | ragforge_indexer.py | ✅ Done — sections/figures respected |
| SAMR-lite module created | src/guardrails/samr_lite.py | ✅ Exists — threshold wrong, fix in SA-01 |
| RuVector Rust crates declared | Cargo.toml | ✅ Declared — versions broken, fix in SA-01 |
| VLM processor for scanned PDFs | src/modules/ragforge/vlm_processor.py | ✅ Done |

### What Is Partially Done (fix in the relevant sub-agent)

| Item | What's Missing |
|---|---|
| Table extraction | Docling chunking works ✅ but rows NOT written to SQLite columns ❌ |
| SAMR threshold | Moved 0.92 → 0.45 (wrong — regression). Must be 0.55 + block action |

### The Core Invariant

> **LLMs explain. Deterministic engines calculate. Citations prove.**
>
> The LLM must never compute numbers where precision matters. It receives
> pre-computed results from the deterministic engine and writes the explanation.
> Any number in an LLM response that does not appear in the calculation trace
> is blocked before reaching the user.

---

## PART 2 — MASTER EXECUTION PLAN

Run sub-agents in this order. Parallel runs are marked where safe.

```
WAVE 1 — Unblock the runtime (must be first)
  SA-01: Fix deps, Cargo versions, SAMR regression
         Files: pyproject.toml, Cargo.toml, .env, src/guardrails/samr_lite.py,
                src/meta_agent.py

WAVE 2 — Anti-hallucination triad (run SA-02 and SA-03 in parallel)
  SA-02: Table extractor + CalcEngine
         Files: src/modules/ragforge/ragforge_indexer.py,
                CREATE src/modules/ragforge/table_extractor.py,
                CREATE src/core/calc_engine.py
  SA-03: Query router + meta_agent wire-up
         Files: CREATE src/core/query_router.py, src/meta_agent.py

WAVE 3 — Upgrade vector store + inference (run SA-04 and SA-05 in parallel)
  SA-04: RuVector vector store replaces ChromaDB
         Files: CREATE src/modules/ragforge/vector_store.py,
                src/core/container.py
  SA-05: ruvllm Rust bridge + model download
         Files: CREATE src-tauri/src/ruvllm_bridge.rs,
                src-tauri/src/lib.rs, Cargo.toml

WAVE 4 — Safety hardening + learning upgrade (run in parallel)
  SA-06: Coherence gate number verifier
         Files: CREATE src/guardrails/coherence_gate.py,
                src/meta_agent.py
  SA-07: SONA learning adapter
         Files: CREATE src/learning/sona_adapter.py, src/meta_agent.py

WAVE 5 — UI features (run SA-09 first, then SA-10 and SA-11 in parallel)
  SA-09: CoT backend — separate thinking from response
         Files: src/meta_agent.py, src/modules/ragforge/cognitiverag.py,
                src/routers/chat.py
  SA-10: CoT frontend — collapsible ThinkingBlock component
         Files: frontend/src/components/ChatBubble.tsx (or equivalent),
                frontend/src/hooks/useChat.ts,
                frontend/src/types/chat.ts,
                CREATE frontend/src/components/ThinkingBlock.tsx
  SA-11: Actionable suggestion tiles
         Files: suggestion component (search frontend/src/ for "suggestion"),
                frontend/src/components/ChatBubble.tsx,
                frontend/src/hooks/useChat.ts

WAVE 6 — Documentation (run last)
  SA-08: Update CLAUDE.md and README
         Files: CLAUDE.md, README.md
```

---

## PART 3 — FILE CHANGE MASTER MAP

| File | Action | Sub-agent | Priority |
|---|---|---|---|
| `pyproject.toml` | Uncomment 6 RuVector deps | SA-01 | CRITICAL |
| `Cargo.toml` | Fix wildcard versions to git paths | SA-01 | CRITICAL |
| `.env` | SAMR threshold 0.45→0.55 + action=block | SA-01 | CRITICAL |
| `src/guardrails/samr_lite.py` | Change warn→raise FaithfulnessError | SA-01 | CRITICAL |
| `src/meta_agent.py` | Handle FaithfulnessError; add router; add SONA hook; add CoT split | SA-01, SA-03, SA-06, SA-07, SA-09 | CRITICAL |
| `src/modules/ragforge/table_extractor.py` | CREATE — table→SQLite typed rows | SA-02 | CRITICAL |
| `src/modules/ragforge/ragforge_indexer.py` | Call table_extractor after Docling parse | SA-02 | CRITICAL |
| `src/core/calc_engine.py` | CREATE — deterministic interpolator | SA-02 | CRITICAL |
| `src/core/query_router.py` | CREATE — intent classifier | SA-03 | CRITICAL |
| `src/modules/ragforge/vector_store.py` | CREATE — RuVectorStore replaces ChromaDB | SA-04 | HIGH |
| `src/core/container.py` | Swap ChromaVectorStore→RuVectorStore | SA-04 | HIGH |
| `src-tauri/src/ruvllm_bridge.rs` | CREATE — Rust LLM Tauri command | SA-05 | HIGH |
| `src-tauri/src/lib.rs` | Register ruvllm command + LLMState | SA-05 | HIGH |
| `src/guardrails/coherence_gate.py` | CREATE — number trace verifier | SA-06 | HIGH |
| `src/learning/sona_adapter.py` | CREATE — SONA 3-tier wrapper | SA-07 | MEDIUM |
| `src/modules/ragforge/cognitiverag.py` | Separate CoT thinking from response text | SA-09 | HIGH |
| `src/routers/chat.py` | Forward thinking field; typed WS events | SA-09 | HIGH |
| `frontend/src/types/chat.ts` | Add thinking fields to ChatMessage | SA-10 | HIGH |
| `frontend/src/components/ThinkingBlock.tsx` | CREATE — collapsible CoT component | SA-10 | HIGH |
| `frontend/src/components/ChatBubble.tsx` | Render ThinkingBlock above answer | SA-10 | HIGH |
| `frontend/src/hooks/useChat.ts` | Handle thinking/answer WS event types | SA-10 | HIGH |
| Suggestion component (find by search) | Auto-submit on click + ↗ arrow | SA-11 | HIGH |
| `CLAUDE.md` | Update architecture description | SA-08 | MEDIUM |
| `README.md` | Update stack references | SA-08 | LOW |
| `models/qwen2.5-7b-instruct-q4_k_m.gguf` | DOWNLOAD from HuggingFace | SA-05 | HIGH |

---

## PART 4 — THE 11 SUB-AGENT PROMPTS

---

### ═══════════════════════════════════════
### SA-01 — Unblock the Runtime
### ═══════════════════════════════════════

**Files to attach to this agent:**
- `pyproject.toml`
- `Cargo.toml`
- `.env` (or `.env.example` if .env is gitignored)
- `src/guardrails/samr_lite.py`
- `src/meta_agent.py`

**Prompt:**
```
You are a senior engineer working on AetherForge — a Rust/Python/Tauri local AI OS.
Repo: github.com/NeoOne601/AtherForge

Read all attached files before making any changes.

YOUR TASK — make these 4 changes in this exact order:

────────────────────────────────────
CHANGE 1 — pyproject.toml: uncomment RuVector Python deps
────────────────────────────────────
Find the block near line 97 that has these 6 commented lines:
  # "ruvector>=0.1.0"
  # "ruvllm>=0.1.0"
  # "ruvector-sona>=0.1.0"
  # "ruvector-refrag>=0.1.0"
  # "prime-radiant-py>=0.1.0"
  # "tiny-dancer>=0.1.0"

Remove the leading "# " from all 6 lines so they become active dependencies.
Do not change any other dependencies. Do not change versions.

After this change the install command to run is: uv pip install -e ".[dev]"

────────────────────────────────────
CHANGE 2 — Cargo.toml: fix wildcard crate versions
────────────────────────────────────
Find these 5 lines with wildcard or partial versions:
  ruvector-core = "*"
  ruvllm = { version = "^2.0", default-features = false, features = ["metal"] }
  prime-radiant = "*"
  rvf-runtime = "*"
  ruvector-sona = "*"

Replace each with a git source pointing to the ruvnet/ruvector monorepo.
Use this exact pattern for each:
  ruvector-core = { git = "https://github.com/ruvnet/ruvector", package = "ruvector-core" }
  ruvllm = { git = "https://github.com/ruvnet/ruvector", package = "ruvllm", features = ["metal"] }
  prime-radiant = { git = "https://github.com/ruvnet/ruvector", package = "prime-radiant" }
  rvf-runtime = { git = "https://github.com/ruvnet/ruvector", package = "rvf-runtime" }
  ruvector-sona = { git = "https://github.com/ruvnet/ruvector", package = "ruvector-sona" }

────────────────────────────────────
CHANGE 3 — .env: fix SAMR threshold regression
────────────────────────────────────
Find the line: SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.45
Change it to:  SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.55

Add a new line directly below it:
  SILICON_COLOSSEUM_FAITHFULNESS_ACTION=block

If there is no .env file (only .env.example), make the same changes there
and add a comment: "# Copy to .env — do not commit actual .env"

────────────────────────────────────
CHANGE 4 — samr_lite.py + meta_agent.py: warn → block
────────────────────────────────────
In src/guardrails/samr_lite.py:
  Find the code path where faithfulness score is below threshold.
  It currently returns something like:
    return {"flagged": True, "badge": "⚠️ LOW CONFIDENCE", "score": score}
  Or it appends a warning string to the response.

  Change it to raise an exception instead:
    class FaithfulnessError(Exception):
        def __init__(self, score: float, threshold: float):
            super().__init__(f"Faithfulness score {score:.2f} below threshold {threshold:.2f}")
            self.score = score
            self.threshold = threshold

    # In the check function, where score is below threshold:
    raise FaithfulnessError(score=actual_score, threshold=0.55)

In src/meta_agent.py:
  Find where the chat pipeline finalises a response (before returning to the
  API router). Add a try/except around the SAMR check call:
    try:
        await samr_lite.verify(response_text, evidence_chunks)
    except FaithfulnessError as e:
        return {
            "response": (
                "I was unable to verify this answer against the source documents "
                f"(confidence {e.score:.0%}). Please rephrase your question or "
                "check the source document directly."
            ),
            "thinking": None,
            "faithfulness_score": e.score,
            "flagged": True,
            "causal_graph": state.get("causal_graph"),
        }

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. pyproject.toml has 0 commented RuVector lines — all 6 are active deps.
2. Cargo.toml has 0 wildcard "*" versions — all 5 use git sources.
3. .env has SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.55 and
   SILICON_COLOSSEUM_FAITHFULNESS_ACTION=block
4. samr_lite.py defines FaithfulnessError and raises it (does not return a
   warning dict for low-confidence responses).
5. meta_agent.py catches FaithfulnessError and returns the safe refusal message.
6. pytest tests/ passes (no tests should break from these config changes).

DO NOT CHANGE: OPA Rego policies, FSM state machine, ChromaDB imports (not yet),
LangGraph graph structure, any module other than samr_lite and meta_agent.
```

---

### ═══════════════════════════════════════
### SA-02 — Table Extractor + Calc Engine
### ═══════════════════════════════════════

**Files to attach to this agent:**
- `src/modules/ragforge/ragforge_indexer.py`

**Prompt:**
```
You are a senior Python engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

Read the attached ragforge_indexer.py carefully before writing any code.

CONTEXT:
The indexer currently uses Docling to parse PDFs and creates text chunks.
117 of the 5,302 chunks are classified as "table" but their numeric data
is stored as unstructured text. We need to branch at ingestion time: tables
go to SQLite as typed rows, prose goes to the vector store as text chunks.
This is the foundational fix — without it, the calc engine has no data.

YOUR TASK — create 2 new files and modify 1 existing file:

────────────────────────────────────────────────────────
FILE 1 — CREATE src/modules/ragforge/table_extractor.py
────────────────────────────────────────────────────────
Create this file with these exact functions:

import sqlite3
from pathlib import Path
from typing import Optional


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


def parse_docling_table(table_obj) -> tuple[list[str], list[dict]]:
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
        row_dict = {}
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
    doc,
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
                import logging
                logging.getLogger(__name__).warning(
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


────────────────────────────────────────────────
FILE 2 — CREATE src/core/calc_engine.py
────────────────────────────────────────────────
Create this file:

import sqlite3
import re
from pathlib import Path


class CalcEngine:
    """
    Deterministic calculation engine.
    ALL arithmetic happens here. The LLM receives the result and trace,
    then writes an explanation. The LLM never performs the calculation.
    """

    def __init__(self, db_path: str):
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


────────────────────────────────────────────────────
FILE 3 — MODIFY src/modules/ragforge/ragforge_indexer.py
────────────────────────────────────────────────────
Find the function that processes a document after Docling converts it.
It will have code that looks like:
  doc = converter.convert(file_path).document
  # ... chunking logic ...
  vector_store.add(chunks)

After the Docling conversion and BEFORE or AFTER the text chunking, add:
  from src.modules.ragforge.table_extractor import extract_tables_to_sqlite

  # Extract numeric table rows to SQLite for the calc engine
  vessel_id = metadata.get("vessel_id") or metadata.get("source", "unknown")
  db_path = settings.data_dir / "structured_data.db"
  table_summary = extract_tables_to_sqlite(doc, vessel_id, str(db_path))
  if table_summary:
      logger.info("Extracted structured table rows: %s", table_summary)

The existing text chunking pipeline must continue to run — do not remove it.
Only table pages get the additional structured extraction.

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. src/modules/ragforge/table_extractor.py exists with all 6 functions.
2. src/core/calc_engine.py exists with CalcEngine class and all 5 methods.
3. ragforge_indexer.py imports and calls extract_tables_to_sqlite after parsing.
4. pytest tests/ passes. Add tests/test_calc_engine.py with these 4 test cases:
   - test_exact_draft_match: draft exactly equals a row → returns that row's value
   - test_interpolation: draft between two rows → correct interpolation result
   - test_fw_correction: apply_fw_correction(25839) → ~25207t (25839 × 0.9756)
   - test_sg_correction: apply_sg_correction(25839, 1.010) → correct dock value
5. The text chunking pipeline in ragforge_indexer.py is UNCHANGED.

DO NOT CHANGE: ChromaDB vector store calls, embedding logic, FTS5 indexing,
any other file outside the 3 listed above.
```

---

### ═══════════════════════════════════════
### SA-03 — Query Router + Meta-Agent Wire-Up
### ═══════════════════════════════════════

**Files to attach:**
- `src/meta_agent.py`
- `src/core/calc_engine.py` (just created by SA-02)

**Prompt:**
```
You are a senior Python engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

Read both attached files before writing any code.

CONTEXT:
SA-02 created src/core/calc_engine.py with a deterministic interpolator.
Your job is to create a query router and wire it into meta_agent.py so that
calculation queries bypass the LLM entirely and go directly to the calc engine.

YOUR TASK — create 1 new file and modify 1 existing file:

────────────────────────────────────────────────
FILE 1 — CREATE src/core/query_router.py
────────────────────────────────────────────────

import re
from enum import Enum


class QueryRoute(str, Enum):
    TABLE_LOOKUP  = "table_lookup"   # single column at one draft
    MULTI_LOOKUP  = "multi_lookup"   # all stability particulars at one draft
    INTERPOLATE   = "interpolate"    # explicit interpolation request
    UNIT_CONVERT  = "unit_convert"   # fresh water / dock water correction
    EXPLAIN       = "explain"        # concept explanation (no numbers)
    PROCEDURE     = "procedure"      # step-by-step how-to
    SYNTHESIS     = "synthesis"      # default: full RAG pipeline


# Patterns and keyword sets
_DRAFT_RE = re.compile(r'\b(\d+\.\d{1,3})\s*m\b', re.IGNORECASE)
_SG_RE    = re.compile(r'\b(?:rd|sg|r\.d\.|s\.g\.)\s*[=:]?\s*\d+\.\d+\b', re.IGNORECASE)

_CALC_KEYWORDS = {
    'displacement', 'displ', 'tpc', 'tonnes per cm', 'mtc', 'moment to change trim',
    'km', 'lcb', 'lcf', 'kn', 'gz', 'righting', 'hydrostatic', 'draft',
    'stability particular', 'stability data', 'interpolat', 'calculat',
    'find the', 'what is the', 'what are the', 'give me the',
    'salt water', 'fresh water', 'dock water',
}
_ALL_PARTICULARS = {
    'all stability', 'all particulars', 'all hydrostatic',
    'stability particulars', 'all values', 'complete stability',
}
_EXPLAIN_KEYWORDS = {
    'what does', 'what is gm', 'what is bm', 'what is kb',
    'explain', 'define', 'describe', 'meaning of', 'why does',
    'how does', 'tell me about', 'what happens',
}
_PROCEDURE_KEYWORDS = {
    'how to', 'how do i', 'steps to', 'procedure for',
    'method for', 'process of',
}
_CONVERT_KEYWORDS = {
    'fresh water', 'dock water', 'river water', 'convert',
    'correction', 'allowance', 'different density', 'rd 1.',
}


def route_query(query: str) -> QueryRoute:
    """
    Classify a user query and return the appropriate route.
    Called BEFORE any LLM invocation.
    """
    q = query.lower().strip()
    has_draft = bool(_DRAFT_RE.search(query))
    has_calc  = any(kw in q for kw in _CALC_KEYWORDS)
    has_conv  = any(kw in q for kw in _CONVERT_KEYWORDS)
    has_all   = any(kw in q for kw in _ALL_PARTICULARS)
    has_expl  = any(kw in q for kw in _EXPLAIN_KEYWORDS)
    has_proc  = any(kw in q for kw in _PROCEDURE_KEYWORDS)

    # Conversion queries (fresh water / dock water)
    if has_draft and has_conv:
        return QueryRoute.UNIT_CONVERT
    if has_conv and not has_expl:
        return QueryRoute.UNIT_CONVERT

    # All stability particulars at one draft
    if has_draft and has_all:
        return QueryRoute.MULTI_LOOKUP

    # Explicit interpolation request
    if has_draft and 'interpolat' in q:
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


def extract_draft(query: str) -> float | None:
    """Extract draft value in metres from a query string."""
    m = _DRAFT_RE.search(query)
    return float(m.group(1)) if m else None


def extract_column(query: str) -> str:
    """Extract which hydrostatic column is being asked for. Default: displacement."""
    q = query.lower()
    if 'tpc' in q or 'tonnes per cm' in q:
        return 'tpc'
    if 'mtc' in q or 'moment to change trim' in q:
        return 'mtc'
    if ' km' in q or 'km ' in q:
        return 'km'
    if 'lcb' in q:
        return 'lcb'
    if 'lcf' in q:
        return 'lcf'
    return 'displacement'  # default


def extract_sg(query: str) -> float | None:
    """Extract specific gravity (RD/SG) from a query string."""
    m = re.search(r'(?:rd|sg)\s*[=:]?\s*(\d+\.\d+)', query, re.IGNORECASE)
    return float(m.group(1)) if m else None


────────────────────────────────────────────────
FILE 2 — MODIFY src/meta_agent.py
────────────────────────────────────────────────
Find the entry point for processing a user message. This is likely a function
or LangGraph node that receives the raw query before routing to modules.

Add the router as the very first step — before ANY LLM call, before any
RAGForge retrieval, before any tool execution:

  from src.core.query_router import route_query, QueryRoute, extract_draft, extract_column, extract_sg
  from src.core.calc_engine import CalcEngine
  from src.config import get_settings

  settings = get_settings()
  calc_engine = CalcEngine(db_path=str(settings.data_dir / "structured_data.db"))

  async def process_message(session_id: str, message: str, module: str, ...) -> dict:
      route = route_query(message)

      # ── Calculation routes: deterministic, no LLM for the math ────
      if route == QueryRoute.TABLE_LOOKUP:
          draft = extract_draft(message)
          column = extract_column(message)
          vessel_id = _get_vessel_id(session_id)
          try:
              calc_result = calc_engine.lookup_hydrostatic(vessel_id, draft, column)
              # LLM only writes the explanation wrapper around the verified numbers
              explanation = await _llm_explain_calc(message, calc_result, session_id)
              return {
                  "response": explanation,
                  "thinking": calc_result["trace"].get("formula", ""),
                  "faithfulness_score": 1.0,
                  "calc_trace": calc_result["trace"],
                  "route": route.value,
              }
          except ValueError as e:
              return {"response": f"I cannot find the data needed: {e}", "route": route.value}

      elif route == QueryRoute.MULTI_LOOKUP:
          draft = extract_draft(message)
          vessel_id = _get_vessel_id(session_id)
          all_results = calc_engine.lookup_all_hydrostatic(vessel_id, draft)
          explanation = await _llm_explain_multi_calc(message, all_results, session_id)
          return {"response": explanation, "thinking": str(all_results), "faithfulness_score": 1.0, "route": route.value}

      elif route == QueryRoute.UNIT_CONVERT:
          draft = extract_draft(message)
          vessel_id = _get_vessel_id(session_id)
          sg = extract_sg(message) or 1.000
          sw_result = calc_engine.lookup_hydrostatic(vessel_id, draft, "displacement")
          if sg == 1.000:
              fw_result = calc_engine.apply_fw_correction(sw_result["value"])
          else:
              fw_result = calc_engine.apply_sg_correction(sw_result["value"], sg)
          explanation = await _llm_explain_calc(message, fw_result, session_id)
          return {"response": explanation, "thinking": fw_result["trace"].get("formula", ""), "faithfulness_score": 1.0, "route": route.value}

      # ── All other routes: full LangGraph pipeline ─────────────────
      # (existing code continues unchanged from here)
      else:
          return await _run_langgraph_pipeline(session_id, message, module, ...)


  async def _llm_explain_calc(original_query: str, calc_result: dict, session_id: str) -> str:
      """
      Ask the LLM to write a PLAIN ENGLISH explanation of a calc result.
      The LLM receives the verified numbers and must use ONLY those numbers.
      """
      prompt = (
          f"The user asked: {original_query}\n\n"
          f"The calculation result is: {calc_result['value']} {calc_result['unit']}\n"
          f"Calculation trace: {calc_result['trace']}\n\n"
          "Write a clear, professional explanation of this result in 2-3 sentences. "
          "Use ONLY the numbers provided above. Do not introduce any other numbers. "
          "Do not perform any arithmetic. Cite the source table if mentioned in the trace."
      )
      # Use existing LLM call mechanism
      return await _invoke_llm(prompt, session_id)

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. src/core/query_router.py exists with route_query() and helper functions.
2. meta_agent.py calls route_query() BEFORE any LangGraph node fires.
3. TABLE_LOOKUP and UNIT_CONVERT routes return results from calc_engine,
   not from the LLM (the LLM is only called for the explanation wrapper).
4. SYNTHESIS and EXPLAIN routes still reach the full LangGraph pipeline
   unchanged.
5. pytest tests/test_query_router.py passes. Add this test file with cases:
   - "displacement at 8.17m" → TABLE_LOOKUP
   - "all stability particulars at 8.33m" → MULTI_LOOKUP
   - "displacement at 7.57m fresh water" → UNIT_CONVERT
   - "what does GM mean" → EXPLAIN
   - "how do I calculate trim" → PROCEDURE
   - "summarise the stability booklet" → SYNTHESIS

DO NOT CHANGE: LangGraph graph definition, module routing, OPA checks,
FSM state machine, replay buffer writes, any non-RAGForge module.
```

---

### ═══════════════════════════════════════
### SA-04 — RuVector Replaces ChromaDB
### ═══════════════════════════════════════

**Files to attach:**
- `src/core/container.py`
- `pyproject.toml` (to confirm ruvector is now uncommented after SA-01)

**Prompt:**
```
You are a senior Python engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

SA-01 has already uncommented "ruvector>=0.1.0" in pyproject.toml.
Read the attached container.py to understand how VectorStore is initialised.

YOUR TASK — create 1 new file and modify 1 existing file:

────────────────────────────────────────────────────────────
FILE 1 — CREATE src/modules/ragforge/vector_store.py
────────────────────────────────────────────────────────────

from ruvector import RuVectorClient
from typing import Any
import logging

logger = logging.getLogger(__name__)


class RuVectorStore:
    """
    Replaces ChromaDB. Uses RuVector's GNN-HNSW with hybrid search.
    Hybrid = 70% dense semantic + 30% BM25 sparse keyword search.
    GNN reranking improves results over time as queries accumulate.
    The BM25 index is always loaded at startup — never lazy-loads.
    """

    def __init__(self, dimensions: int = 1024):
        """dimensions=1024 matches BAAI/bge-m3 embedding model."""
        self.client = RuVectorClient(
            dimensions=dimensions,
            gnn=True,            # self-learning GNN layer on HNSW
        )
        logger.info("RuVectorStore initialised: dim=%d gnn=True", dimensions)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 12,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid semantic + keyword search with GNN reranking.
        Returns list of dicts with keys: id, text, metadata, score.
        """
        return await self.client.hybrid_search(
            query=embedding,
            k=top_k,
            hybrid_alpha=0.7,    # 70% semantic, 30% BM25
            rerank=True,         # GNN layer reranks candidates
            where=where,         # optional metadata filter
        )

    async def add(
        self,
        embedding: list[float],
        text: str,
        metadata: dict,
        doc_id: str | None = None,
    ) -> str:
        """Add a document chunk. Returns the stored document ID."""
        return await self.client.upsert(
            embedding=embedding,
            text=text,
            metadata=metadata,
            id=doc_id,
        )

    async def add_batch(self, items: list[dict]) -> list[str]:
        """
        Batch insert for ingestion efficiency.
        Each item: {"embedding": [...], "text": "...", "metadata": {...}, "id": "..."}
        """
        return await self.client.upsert_batch(items)

    async def delete(self, doc_id: str) -> None:
        await self.client.delete(doc_id)

    async def count(self) -> int:
        return await self.client.count()

    async def reset(self) -> None:
        """Clear all vectors. Used in tests only."""
        await self.client.reset()


────────────────────────────────────────────────────────────
FILE 2 — MODIFY src/core/container.py
────────────────────────────────────────────────────────────
Find where VectorStore is initialised. It looks like one of:
  self._vector_store = ChromaVectorStore(path=settings.chroma_path)
  vector_store = Chroma(persist_directory=str(settings.chroma_path), ...)
  from chromadb import Client; client = Client(...)

Replace it with:
  from src.modules.ragforge.vector_store import RuVectorStore
  self._vector_store = RuVectorStore(dimensions=1024)

Also update the property or method that returns the vector store to use
the new type annotation if it has one:
  def vector_store(self) -> RuVectorStore:  # was ChromaVectorStore

Do NOT remove the chromadb import yet if other code still references it —
just swap the instantiation. We will clean up chromadb imports in SA-08.

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. src/modules/ragforge/vector_store.py exists with RuVectorStore.
2. container.py instantiates RuVectorStore, not ChromaVectorStore.
3. All calls to vector_store.search(...) and vector_store.add(...) in
   ragforge_indexer.py and cognitiverag.py still work with the new interface.
   (Check that call sites use keyword arguments that match the new method signatures.)
4. pytest tests/ passes. Tests that mock vector_store must be updated to
   mock RuVectorStore instead of ChromaVectorStore.

DO NOT CHANGE: embedding generation logic, FTS5 sparse index code,
document ingestion pipeline outside vector_store calls, calc_engine.
```

---

### ═══════════════════════════════════════
### SA-05 — ruvllm Rust Bridge + Model
### ═══════════════════════════════════════

**Files to attach:**
- `Cargo.toml` (after SA-01 fixes)
- `src-tauri/src/lib.rs`
- `src-tauri/src/main.rs`

**Prompt:**
```
You are a senior Rust/Tauri engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge
Stack: Tauri 2.1, Rust 1.78+, ruvllm crate

Read all 3 attached files before writing any code.

CONTEXT:
AetherForge currently calls an Ollama HTTP server at localhost:11434 for
LLM inference. This adds ~5ms per call and limits context to 8192 tokens.
We replace it with ruvllm, a native Rust GGUF runtime, running as a Tauri
command directly inside the desktop process.

YOUR TASK — create 1 new file and modify 2 existing files:

────────────────────────────────────────────────────────────
FILE 1 — CREATE src-tauri/src/ruvllm_bridge.rs
────────────────────────────────────────────────────────────

use ruvllm::{RuvLLM, GenerateOptions, Message, Role};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

pub struct LLMState(pub Arc<Mutex<RuvLLM>>);

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system_prompt: Option<String>,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub duration_ms: u64,
}

#[tauri::command]
pub async fn llm_generate(
    state: State<'_, LLMState>,
    request: GenerateRequest,
) -> Result<GenerateResponse, String> {
    let llm = state.0.lock().await;
    let start = std::time::Instant::now();

    let opts = GenerateOptions {
        max_tokens: request.max_tokens.unwrap_or(1024),
        temperature: request.temperature.unwrap_or(0.7),
        repetition_penalty: 1.15,  // prevents infinite repetition loops
        n_ctx: 16384,              // 2× the old 8192 limit
    };

    let messages = if let Some(sys) = request.system_prompt {
        vec![
            Message { role: Role::System, content: sys },
            Message { role: Role::User, content: request.prompt },
        ]
    } else {
        vec![Message { role: Role::User, content: request.prompt }]
    };

    let result = llm.chat(messages, opts)
        .await
        .map_err(|e| format!("ruvllm error: {e}"))?;

    Ok(GenerateResponse {
        text: result.content,
        tokens_generated: result.tokens_generated,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

#[tauri::command]
pub async fn llm_health(state: State<'_, LLMState>) -> Result<bool, String> {
    let _llm = state.0.lock().await;
    Ok(true)
}

pub async fn init_llm(model_path: &str) -> Result<LLMState, String> {
    let llm = RuvLLM::new(model_path, "metal")  // "metal" = Apple Silicon GPU
        .await
        .map_err(|e| format!("Failed to load model {model_path}: {e}"))?;
    Ok(LLMState(Arc::new(Mutex::new(llm))))
}


────────────────────────────────────────────────────────────
FILE 2 — MODIFY src-tauri/src/lib.rs
────────────────────────────────────────────────────────────
Add to the module declarations at the top:
  pub mod ruvllm_bridge;
  use ruvllm_bridge::{LLMState, init_llm, llm_generate, llm_health};

Find the .invoke_handler macro and add the new commands:
  .invoke_handler(tauri::generate_handler![
      // ... existing commands ...
      llm_generate,
      llm_health,
  ])

────────────────────────────────────────────────────────────
FILE 3 — MODIFY src-tauri/src/main.rs (or lib.rs setup)
────────────────────────────────────────────────────────────
In the Tauri builder setup (the run() function), add .manage() for LLMState.
Find where .manage() calls are made for other state objects and add:

  // Load the LLM model at startup
  let model_path = std::env::var("QWEN_MODEL_PATH")
      .unwrap_or_else(|_| "./models/qwen2.5-7b-instruct-q4_k_m.gguf".to_string());

  let llm_state = match init_llm(&model_path).await {
      Ok(state) => state,
      Err(e) => {
          eprintln!("WARNING: Failed to load ruvllm model: {e}");
          eprintln!("Falling back to Ollama HTTP inference.");
          // Allow startup to continue — inference falls back to existing Ollama calls
          return;  // or handle gracefully
      }
  };

  builder.manage(llm_state)

Also add the model path env var to .env:
  QWEN_MODEL_PATH=./models/qwen2.5-7b-instruct-q4_k_m.gguf

────────────────────────────────────────────────────────────
MODEL DOWNLOAD COMMAND (include in a comment or README update)
────────────────────────────────────────────────────────────
Add this to the install.sh script after the existing model download:

  echo "Downloading Qwen2.5-7B-Instruct (required for ruvllm)..."
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    --include "qwen2.5-7b-instruct-q4_k_m.gguf" \
    --local-dir models/
  echo "Model downloaded to models/qwen2.5-7b-instruct-q4_k_m.gguf"

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. src-tauri/src/ruvllm_bridge.rs compiles without errors.
2. cargo build succeeds (may take 5-10 minutes on first compile).
3. The llm_generate Tauri command is registered and callable from frontend.
4. A model file qwen2.5-7b-instruct-q4_k_m.gguf exists in models/ after
   running the install command.
5. The startup code handles missing model gracefully (falls back, doesn't crash).

DO NOT CHANGE: existing Ollama HTTP fallback, Python backend inference,
the FastAPI server, any frontend files.
```

---

### ═══════════════════════════════════════
### SA-06 — Coherence Gate Number Verifier
### ═══════════════════════════════════════

**Files to attach:**
- `src/guardrails/samr_lite.py`
- `src/meta_agent.py` (after SA-03 changes)

**Prompt:**
```
You are a senior Python engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

Read both attached files carefully.

CONTEXT:
For queries routed through the calc engine (TABLE_LOOKUP, UNIT_CONVERT etc.),
the LLM writes an explanation around pre-computed numbers. We need to verify
that EVERY number in the LLM's explanation traces back to the calc engine's
output trace. Any invented number gets blocked.

YOUR TASK — create 1 new file and modify 1 existing file:

────────────────────────────────────────────────────────────
FILE 1 — CREATE src/guardrails/coherence_gate.py
────────────────────────────────────────────────────────────

import re
import json
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
    significant = set()
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
        nums = set()
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
    unauthorized = set()
    for rn in response_numbers:
        if rn in acceptable:
            continue
        # Check within tolerance
        try:
            rval = float(rn)
            within_tolerance = any(
                abs(rval - float(tn)) / max(abs(float(tn)), 1) < tolerance
                for tn in trace_numbers if tn.replace('.', '').isdigit()
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


────────────────────────────────────────────────────────────
FILE 2 — MODIFY src/meta_agent.py
────────────────────────────────────────────────────────────
In the section added by SA-03 for TABLE_LOOKUP / UNIT_CONVERT routes,
after the LLM generates the explanation (_llm_explain_calc returns),
add the number verification before returning the response:

  from src.guardrails.coherence_gate import verify_calc_response, NumberVerificationError

  # In the TABLE_LOOKUP handler:
  explanation = await _llm_explain_calc(message, calc_result, session_id)
  try:
      verify_calc_response(explanation, calc_result["trace"])
  except NumberVerificationError as e:
      logger.warning("Number verification failed: %s", e)
      # Return a safe response using ONLY the verified numbers
      return {
          "response": (
              f"The {calc_result.get('unit', 'value')} at draft {draft}m is "
              f"{calc_result['value']} {calc_result.get('unit', '')}. "
              "(Verified from hydrostatic table.)"
          ),
          "thinking": calc_result["trace"].get("formula", ""),
          "faithfulness_score": 1.0,
          "calc_trace": calc_result["trace"],
          "route": route.value,
          "coherence_enforced": True,
      }

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. src/guardrails/coherence_gate.py exists with all 4 functions.
2. meta_agent.py calls verify_calc_response after _llm_explain_calc.
3. pytest tests/test_coherence_gate.py passes with these test cases:
   - test_clean_response_passes: response using only trace numbers → no error
   - test_invented_number_blocked: response with a made-up number → raises error
   - test_tolerance_allowed: response with rounded version of trace number → passes
   - test_non_calc_route_skipped: SYNTHESIS route → verify not called

DO NOT CHANGE: SAMR-lite (for non-calc routes), OPA policies, FSM, X-Ray.
```

---

### ═══════════════════════════════════════
### SA-07 — SONA Learning Adapter
### ═══════════════════════════════════════

**Files to attach:**
- `src/learning/oplora_manager.py`
- `src/meta_agent.py`
- `src/core/container.py`

**Prompt:**
```
You are a senior ML engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

Read all 3 attached files before writing any code.

CONTEXT:
AetherForge currently learns only during nightly OPLoRA batch runs (3 AM).
SONA adds 3-tier real-time learning:
  Tier 1: MicroLoRA rank-2 — adapts in <1ms per accepted response
  Tier 2: EWC++ consolidation — runs in background ~100ms, prevents forgetting
  Tier 3: ReasoningBank — stores successful query→answer trajectories as curriculum
OPLoRA nightly runs continue unchanged. SONA supplements them, does not replace.

YOUR TASK — create 1 new file and modify 1 existing file:

────────────────────────────────────────────────────────────
FILE 1 — CREATE src/learning/sona_adapter.py
────────────────────────────────────────────────────────────

from ruvector_sona import SONA
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SONAAdapter:
    """
    SONA 3-tier learning adapter.
    Supplements OPLoRA nightly batch with per-request adaptation.
    """

    def __init__(self, model_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self._sona: SONA | None = None
        self._model_path = model_path
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize SONA. Called at startup. Fails gracefully if unavailable."""
        try:
            self._sona = SONA(
                model_path=self._model_path,
                micro_lora_rank=2,        # <1ms per-request adaptation
                ewc_lambda=400,           # catastrophic forgetting protection
                reasoning_bank=True,      # trajectory curriculum memory
                checkpoint_dir=str(self.data_dir / "sona_checkpoints"),
            )
            self._initialized = True
            logger.info("SONA adapter initialised")
        except Exception as e:
            logger.warning("SONA initialisation failed (will use OPLoRA only): %s", e)
            self._initialized = False

    async def on_interaction(
        self,
        query: str,
        response: str,
        verdict: str,            # "accepted" | "rejected" | "corrected"
        route: str = "synthesis",
        metadata: dict | None = None,
    ) -> None:
        """
        Called after every completed interaction.
        verdict = "accepted" when user accepts the response without correction.
        verdict = "rejected" when user corrects or dismisses the response.
        """
        if not self._initialized or self._sona is None:
            return
        try:
            # Tier 1: MicroLoRA instant adaptation
            await self._sona.micro_update(query, response, verdict)
            # Tier 3: Record to ReasoningBank for curriculum learning
            self._sona.reasoning_bank.record(
                query=query,
                response=response,
                verdict=verdict,
                metadata={"route": route, **(metadata or {})},
            )
        except Exception as e:
            logger.warning("SONA on_interaction failed (non-fatal): %s", e)

    async def get_stats(self) -> dict:
        """Return SONA learning stats for TuneLab display."""
        if not self._initialized or self._sona is None:
            return {"status": "unavailable", "initialized": False}
        return {
            "status": "active",
            "initialized": True,
            "micro_lora_rank": self._sona.micro_lora_rank,
            "reasoning_bank_size": self._sona.reasoning_bank.size(),
        }


────────────────────────────────────────────────────────────
FILE 2 — MODIFY src/meta_agent.py
────────────────────────────────────────────────────────────
Add SONA initialisation alongside the existing replay buffer setup:

  from src.learning.sona_adapter import SONAAdapter

  # Near where the container is initialised:
  sona = SONAAdapter(
      model_path=settings.bitnet_model_path,  # will be updated to Qwen path
      data_dir=str(settings.data_dir),
  )
  # Start SONA in background at startup:
  asyncio.create_task(sona.initialize())

After every successful response is returned to the user, call:
  asyncio.create_task(sona.on_interaction(
      query=user_message,
      response=final_response,
      verdict="accepted",   # default — user feedback mechanism can update this
      route=route.value if route else "synthesis",
  ))

The call must be non-blocking (create_task, not await) so it doesn't delay
the response.

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. src/learning/sona_adapter.py exists with SONAAdapter class.
2. meta_agent.py initialises SONAAdapter at startup.
3. meta_agent.py calls sona.on_interaction() after every successful response.
4. If SONA init fails (e.g. ruvector_sona not installed), system still starts
   and uses OPLoRA nightly batch as before — SONA failure is non-fatal.
5. OPLoRA nightly batch (existing code) is UNCHANGED.
6. pytest tests/ passes. OPLoRA tests must still pass.

DO NOT CHANGE: OPLoRA SVD math, replay buffer, nightly scheduler,
X-Ray trace, any guardrail.
```

---

### ═══════════════════════════════════════
### SA-08 — Update Documentation
### ═══════════════════════════════════════

**Files to attach:**
- `CLAUDE.md`
- `README.md`
- `AetherForge_Complete_Build_Guide_v1.0.md`

**Prompt:**
```
You are a technical writer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

Read all 3 attached files carefully. Update them to reflect the new architecture.

CLAUDE.md CHANGES (this is the most critical — Claude Code reads this):
Replace every reference to the old stack:
  ChromaDB        → RuVector GNN-HNSW (self-learning hybrid vector + BM25 search)
  BitNet / llama-cpp-python / BitNet 1.58b → ruvllm + Qwen2.5-7B-Instruct-Q4_K_M
  SAMR-lite warn  → Prime Radiant coherence gate (blocks on sheaf energy > 0.70)
  OPLoRA nightly only → SONA 3-tier learning (MicroLoRA <1ms + EWC++ + ReasoningBank) + OPLoRA nightly

Add these NEW components to the architecture section:
  - CalcEngine (src/core/calc_engine.py): deterministic table interpolator, no LLM
  - QueryRouter (src/core/query_router.py): intent classifier, fires before LLM
  - TableExtractor (src/modules/ragforge/table_extractor.py): table→SQLite at ingestion
  - RuVectorStore (src/modules/ragforge/vector_store.py): replaces ChromaDB
  - CoherenceGate (src/guardrails/coherence_gate.py): number trace verifier
  - SONAAdapter (src/learning/sona_adapter.py): per-request SONA learning
  - ruvllm_bridge.rs (src-tauri/src/): Rust Tauri command for LLM inference
  - ThinkingBlock.tsx (frontend/src/components/): collapsible CoT display

Update the Environment Variables section:
  Add: QWEN_MODEL_PATH=./models/qwen2.5-7b-instruct-q4_k_m.gguf
  Add: SILICON_COLOSSEUM_FAITHFULNESS_ACTION=block
  Change: SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.55
  Remove: references to BITNET_MODEL_PATH (keep as legacy fallback note)

Update the Core Services Architecture section to list new services.
Update the "Adding a New Module" section — no changes needed there.

README.md CHANGES:
Update the High-Level System Design mermaid diagram to show:
  QueryRouter → [CalcEngine path | CognitiveRAG path]
  CognitiveRAG → RuVector (not ChromaDB)
  Meta-Agent → ruvllm (not BitNet)
  SONA → OPLoRA (supplement, not replace)

Update the Core Innovations section:
  Add: "Query Router + Calc Engine: Deterministic calculation for all
       numeric queries. LLMs explain; they never calculate."

AetherForge_Complete_Build_Guide_v1.0.md CHANGES:
Update the Environment Variables table to include all new vars.
Update the Performance Benchmarks table (keep M1 numbers, note model change).
Update the System Topology to include the 7 new files.
Update the Troubleshooting table: remove "ChromaDB error → delete data/chroma/",
  add "RuVector data: delete data/ruvector/ and re-ingest".

ACCEPTANCE CRITERIA:
1. CLAUDE.md contains no references to ChromaDB, BitNet, SAMR-lite warn,
   or OPLoRA as the only learning path.
2. CLAUDE.md architecture section lists all 7 new components.
3. README.md mermaid diagram shows the new routing flow.
4. No code samples in any doc are broken or inconsistent with the new stack.
```

---

### ═══════════════════════════════════════
### SA-09 — CoT Backend: Separate Thinking from Response
### ═══════════════════════════════════════

**Files to attach:**
- `src/meta_agent.py`
- `src/modules/ragforge/cognitiverag.py`
- `src/routers/chat.py`

**Prompt:**
```
You are a senior Python backend engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge

Read all 3 attached files before making any changes.

CONTEXT:
Currently the CognitiveRAG Stage ⑥ (Chain-of-Thought synthesis) generates
reasoning steps and embeds them directly inside the `response` string:
  response: "Step 1: I looked at the table... Step 2: I found the rows...\n\nThe answer is 25,839 tonnes."

The frontend renders this as a single text block — users see reasoning mixed
with the answer. We need to separate them into two distinct fields so the
frontend can render a collapsible "Thinking..." block above the clean answer.

YOUR TASK — modify 3 existing files:

────────────────────────────────────────────────────────────
FILE 1 — MODIFY src/modules/ragforge/cognitiverag.py
────────────────────────────────────────────────────────────
Add this function near the top of the file (before Stage ⑥):

def extract_cot(full_output: str) -> tuple[str, str]:
    """
    Split raw LLM output into (thinking, clean_answer).
    Tries patterns in order of reliability. Falls back to 60/40 split.
    Returns (thinking_text, answer_text).
    """
    import re

    # Pattern A: explicit <think>...</think> XML tags
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL | re.IGNORECASE).strip()
        return thinking, answer

    # Pattern B: **Reasoning:** or **Thinking:** header followed by blank line
    reasoning_match = re.match(
        r'\*\*(?:Reasoning|Thinking|Analysis):\*\*\s*(.*?)\n\n(.*)',
        full_output, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        return reasoning_match.group(1).strip(), reasoning_match.group(2).strip()

    # Pattern C: "Step N:" lines until first blank line, then answer
    step_match = re.match(
        r'((?:Step\s+\d+[:.][^\n]+\n?)+)\n+(.*)',
        full_output, re.DOTALL | re.IGNORECASE
    )
    if step_match:
        return step_match.group(1).strip(), step_match.group(2).strip()

    # Pattern D: "Let me think" / "First, " preamble
    preamble_match = re.match(
        r'((?:Let me|First,|To answer|I need to|Looking at)[^\n]{10,}\n(?:[^\n]+\n)*?)\n+(.*)',
        full_output, re.DOTALL
    )
    if preamble_match:
        return preamble_match.group(1).strip(), preamble_match.group(2).strip()

    # Pattern E (fallback): first 60% is thinking, last 40% is answer
    lines = full_output.split('\n')
    split = max(1, int(len(lines) * 0.6))
    return '\n'.join(lines[:split]).strip(), '\n'.join(lines[split:]).strip()

In Stage ⑥ (CoT synthesis), find where the LLM output is assigned:
  full_output = llm_response.content  # or however the LLM text is accessed

Replace the direct assignment of the full output to response with:
  thinking, clean_response = extract_cot(full_output)
  # Store both separately in the pipeline state dict:
  return {
      **existing_return_fields,
      "response": clean_response,
      "thinking": thinking,
  }

────────────────────────────────────────────────────────────
FILE 2 — MODIFY src/meta_agent.py
────────────────────────────────────────────────────────────
Find the AgentResponse Pydantic model (or whatever model is serialised as the
API response). Add:
  thinking: Optional[str] = None
  thinking_duration_ms: Optional[int] = None

When building the final response dict from the pipeline state, include:
  "thinking": state.get("thinking"),
  "thinking_duration_ms": state.get("thinking_duration_ms"),

────────────────────────────────────────────────────────────
FILE 3 — MODIFY src/routers/chat.py
────────────────────────────────────────────────────────────
REST endpoint (POST /api/v1/chat):
  The JSON response already includes all fields from AgentResponse.
  No change needed here if AgentResponse now has the thinking field —
  it will be automatically included in the Pydantic serialisation.
  Verify this is the case.

WebSocket endpoint (/ws/chat/{session_id}):
  Currently streams response tokens as a single stream.
  Change to use typed events:

  # When thinking text is available (before answer generation):
  await websocket.send_json({"type": "thinking", "content": thinking_text})
  await websocket.send_json({"type": "thinking_complete", "duration_ms": thinking_ms})

  # Then stream answer tokens:
  for chunk in answer_token_stream:
      await websocket.send_json({"type": "answer", "content": chunk})

  # When complete:
  await websocket.send_json({
      "type": "done",
      "causal_graph": causal_graph,
      "faithfulness_score": faithfulness_score,
      "module": module,
      "latency_ms": total_ms,
  })

  If the pipeline does not stream tokens separately (generates full response
  first), simulate:
    await websocket.send_json({"type": "thinking", "content": full_thinking})
    await websocket.send_json({"type": "thinking_complete", "duration_ms": 0})
    for chunk in chunk_text(clean_answer, size=20):
        await websocket.send_json({"type": "answer", "content": chunk})
    await websocket.send_json({"type": "done", ...})

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. curl test:
   curl -X POST http://localhost:8765/api/v1/chat \
     -d '{"session_id":"t1","module":"ragforge","message":"how does stability work?"}' \
     -H "Content-Type: application/json" | python3 -m json.tool

   Response MUST have "thinking": a non-empty string AND "response": clean answer.

2. "response" field must NOT start with Step, Reasoning:, Let me think,
   First,, or any CoT marker word.

3. pytest tests/test_cognitiverag.py passes, including new extract_cot tests:
   - <think> tag extraction
   - **Reasoning:** header extraction
   - Step N: line extraction
   - fallback 60/40 split

4. WebSocket sends events in order: thinking → thinking_complete → answer(s) → done.

5. DO NOT change: OPA checks, SAMR checks, replay buffer writes, causal graph
   structure, any non-RAGForge module.
```

---

### ═══════════════════════════════════════
### SA-10 — CoT Frontend: ThinkingBlock Component
### ═══════════════════════════════════════

**Files to attach:**
- `frontend/src/components/ChatBubble.tsx` (or equivalent assistant message component)
- `frontend/src/hooks/useChat.ts` (or `useWebSocket.ts`)
- `frontend/src/types/chat.ts` (or wherever `ChatMessage` type is defined)
- `frontend/src/stores/chatStore.ts` (if it exists)

**Prompt:**
```
You are a senior React/TypeScript engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge
Stack: React 18, TypeScript 5.5, Tailwind CSS, Shadcn/ui, Tauri 2.1
Frontend root: frontend/src/

Read all attached files before writing any code.
SA-09 has added `thinking` and `thinking_duration_ms` to the API response.

YOUR TASK — create 1 new file and modify 3 existing files:

────────────────────────────────────────────────────────────
FILE 1 — MODIFY the ChatMessage type (chat.ts or equivalent)
────────────────────────────────────────────────────────────
Find the interface or type that defines a chat message.
Add these optional fields:
  thinking?: string              // CoT reasoning text from backend
  thinkingDurationMs?: number    // shown as "Thought for Xs" in header
  isThinkingStreaming?: boolean   // true while model is still generating CoT

────────────────────────────────────────────────────────────
FILE 2 — CREATE frontend/src/components/ThinkingBlock.tsx
────────────────────────────────────────────────────────────
Create this component exactly:

import { useState } from "react"
import { cn } from "@/lib/utils"  // adjust import path if cn is elsewhere

interface ThinkingBlockProps {
  content: string
  durationMs?: number
  isStreaming?: boolean
  className?: string
}

export function ThinkingBlock({
  content,
  durationMs,
  isStreaming = false,
  className,
}: ThinkingBlockProps) {
  const [isOpen, setIsOpen] = useState(false)

  const headerLabel = isStreaming && !content
    ? "Thinking..."
    : durationMs
    ? `Thought for ${Math.round(durationMs / 1000)}s`
    : content
    ? "Reasoning"
    : "Thinking..."

  const showSpinner = isStreaming && !content

  return (
    <div
      className={cn(
        "border-l-2 border-muted-foreground/20 bg-muted/30 rounded-r-md",
        "pl-3 pr-2 py-2 mb-3 cursor-pointer select-none",
        className
      )}
      onClick={() => setIsOpen((prev) => !prev)}
      role="button"
      aria-expanded={isOpen}
    >
      <div className="flex items-center gap-2">
        {showSpinner ? (
          <span className="w-3 h-3 border-2 border-muted-foreground/20 border-t-muted-foreground/60 rounded-full animate-spin flex-shrink-0" />
        ) : (
          <span
            className={cn(
              "text-muted-foreground/60 text-xs transition-transform duration-150 flex-shrink-0",
              isOpen ? "rotate-90" : ""
            )}
          >
            ▶
          </span>
        )}
        <span className="text-xs text-muted-foreground font-medium">
          {headerLabel}
        </span>
      </div>

      {isOpen && content && (
        <div className="mt-2 pt-2 border-t border-border/50">
          <p className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto">
            {content}
          </p>
        </div>
      )}
    </div>
  )
}

────────────────────────────────────────────────────────────
FILE 3 — MODIFY ChatBubble.tsx (or equivalent)
────────────────────────────────────────────────────────────
Find the component that renders an individual AI response bubble.
Find where the response text is rendered (inside a <div className="prose ...">
or a markdown renderer component).

BEFORE that prose block, add the ThinkingBlock:
  {message.thinking && (
    <ThinkingBlock
      content={message.thinking}
      durationMs={message.thinkingDurationMs}
      isStreaming={message.isThinkingStreaming}
    />
  )}

Import at the top:
  import { ThinkingBlock } from "@/components/ThinkingBlock"
  (adjust path as needed)

────────────────────────────────────────────────────────────
FILE 4 — MODIFY useChat.ts / useWebSocket.ts
────────────────────────────────────────────────────────────
Find the WebSocket message handler (the onmessage or equivalent callback).

Currently it likely has something like:
  case "token": updateMessage(id, { response: prev + chunk }); break;
  case "done": updateMessage(id, { isStreaming: false, ...meta }); break;

Replace/extend with the new event types from SA-09:

  switch (data.type) {
    case "thinking":
      updateMessage(currentMsgId, (prev) => ({
        ...prev,
        thinking: (prev.thinking ?? "") + data.content,
        isThinkingStreaming: true,
      }))
      break

    case "thinking_complete":
      updateMessage(currentMsgId, (prev) => ({
        ...prev,
        thinkingDurationMs: data.duration_ms,
        isThinkingStreaming: false,
      }))
      break

    case "answer":
      updateMessage(currentMsgId, (prev) => ({
        ...prev,
        response: (prev.response ?? "") + data.content,
      }))
      break

    case "done":
      updateMessage(currentMsgId, (prev) => ({
        ...prev,
        isStreaming: false,
        causal_graph: data.causal_graph,
        faithfulness_score: data.faithfulness_score,
        latency_ms: data.latency_ms,
      }))
      break

    // Keep any existing cases (for backward compatibility):
    default:
      // existing handling
  }

If the app uses REST (not WebSocket), update the fetch handler to set
  message.thinking = response.thinking
after the fetch resolves.

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. ThinkingBlock.tsx exists as a standalone component with no Shadcn deps
   beyond cn(). It renders without errors.
2. ChatBubble renders ThinkingBlock above the answer when thinking is present.
3. ThinkingBlock is collapsed by default. Clicking toggles smoothly.
4. Spinner shows while isThinkingStreaming is true and content is empty.
5. Answer text has NO CoT prefix (Step 1:, Reasoning:, etc.).
6. npm run type-check passes with no errors.
7. npm run lint passes.

DO NOT CHANGE: X-Ray panel, metadata badges (module/latency/fidelity),
suggestion tiles, session handling, module tabs, any non-chat component.
```

---

### ═══════════════════════════════════════
### SA-11 — Actionable Suggestion Tiles
### ═══════════════════════════════════════

**Files to attach:**
- The suggestion tiles component (search `frontend/src/` for files containing "suggestion", "followup", "FollowUp", "promptchip", or "SuggestedPrompt")
- `frontend/src/components/ChatBubble.tsx`
- `frontend/src/hooks/useChat.ts` (or `useChatInput.ts`)

**Prompt:**
```
You are a senior React/TypeScript engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge
Stack: React 18, TypeScript 5.5, Tailwind CSS, Shadcn/ui, Tauri 2.1

Read all attached files before writing any code.

CONTEXT:
After each AI response, suggestion tiles appear below the answer. Clicking
a tile currently fills the chat input box — the user must still press Enter
manually. We need clicking a tile to immediately submit the suggestion as a
new user message, exactly like Gemini suggestion chips and Claude follow-ups.

YOUR TASK — modify 2–3 files (no new files needed):

────────────────────────────────────────────────────────────
STEP 1 — Find the suggestion component
────────────────────────────────────────────────────────────
Search frontend/src/ for the component that renders the suggestion/followup
tiles. It may be in:
  - A dedicated file (SuggestedPrompts.tsx, FollowUpSuggestions.tsx, etc.)
  - Inline inside ChatBubble.tsx

Identify the current onClick handler. It looks like one of:
  onClick={() => setInputValue(suggestion)}
  onClick={() => props.onSuggestionClick(suggestion)}
  onClick={() => dispatch(setInput(suggestion))}

────────────────────────────────────────────────────────────
STEP 2 — Add onSubmit prop + make it auto-submit
────────────────────────────────────────────────────────────
Update the component's props to accept a submit function:

  interface SuggestionTileProps {
    suggestions: string[]
    onSubmit: (text: string) => void    // NEW: submits the message directly
    disabled?: boolean                   // NEW: true while a response is streaming
  }

Add local state to prevent double-click:
  const [submitted, setSubmitted] = useState(false)

Change the onClick handler:
  const handleClick = (suggestion: string) => {
    if (submitted || disabled) return
    setSubmitted(true)
    onSubmit(suggestion)    // directly submits — does not just fill input
  }

Hide all tiles after one is clicked:
  if (submitted) return null

────────────────────────────────────────────────────────────
STEP 3 — Update tile visual design (add ↗ arrow)
────────────────────────────────────────────────────────────
Find the button element for each tile. Add a ↗ arrow icon and hover effect:

  <button
    onClick={() => handleClick(suggestion)}
    disabled={submitted || disabled}
    className="
      flex items-center gap-1.5 group
      border border-border/60 rounded-md
      px-3 py-1.5 text-sm
      bg-background hover:bg-muted/50
      hover:border-primary/40
      transition-colors duration-150
      disabled:opacity-40 disabled:cursor-not-allowed
    "
  >
    <span>{suggestion}</span>
    <span className="
      text-muted-foreground/50 group-hover:text-primary/70
      text-xs transition-colors duration-150
    ">
      ↗
    </span>
  </button>

────────────────────────────────────────────────────────────
STEP 4 — Pass onSubmit from the parent (ChatBubble or chat view)
────────────────────────────────────────────────────────────
Find where the suggestion component is rendered. Pass the actual send function:

  <SuggestionTiles
    suggestions={message.suggestions}
    onSubmit={sendMessage}      // THE ACTUAL SUBMIT FUNCTION — not setInputValue
    disabled={isLatestStreaming}  // disable during active generation
  />

Find the sendMessage function. It lives in useChat, useChatInput, or a Redux
action. It must:
  1. Set the input value to the suggestion text
  2. Immediately trigger message submission (same as pressing Enter)
  3. Clear the input after sending

If the current send function only sets input but doesn't send, create a
wrapper:
  const handleSuggestionSubmit = (text: string) => {
    setInput(text)
    sendMessage(text)   // or whatever the send function is called
  }

────────────────────────────────────────────────────────────
STEP 5 — Disable tiles during active generation
────────────────────────────────────────────────────────────
Pass the streaming state to the tiles component. When the latest message
is still streaming, the tiles for that message (or all visible tiles) should
be visually disabled (opacity-40) and non-clickable.

────────────────────────────────────
ACCEPTANCE CRITERIA:
────────────────────────────────────
1. Clicking a suggestion tile immediately sends the suggestion as a new user
   message — the user does NOT need to press Enter.
2. All tiles disappear after one is clicked (submitted state).
3. Each tile has a ↗ arrow icon on the right.
4. Tiles are disabled (not clickable) while a response is streaming.
5. Rapid double-clicking does not send the message twice.
6. npm run type-check passes. npm run lint passes.

DO NOT CHANGE: how suggestions are generated by the backend, the suggestion
content or count, the ThinkingBlock component, X-Ray panel, metadata badges,
any non-chat component, or the session handling logic.
```

---

## PART 5 — COMPLETE ACCEPTANCE TEST SUITE

Run ALL of these after all 11 sub-agents complete. Every test must pass.

### Backend tests
```bash
# 1. Install all deps
uv pip install -e ".[dev]"

# 2. Run full test suite
pytest tests/ -v

# 3. Test calc engine directly
python3 -c "
from src.core.calc_engine import CalcEngine
e = CalcEngine('data/structured_data.db')
result = e.lookup_hydrostatic('HA-13', 8.17, 'displacement')
print('8.17m displacement:', result['value'], 't')
assert abs(result['value'] - 25839) < 2, f'Expected ~25839, got {result[\"value\"]}'
print('PASS: displacement within tolerance')

fw = e.apply_fw_correction(25839)
assert abs(fw['value'] - 25207) < 2
print('PASS: FW correction:', fw['value'], 't')
"

# 4. Test query router
python3 -c "
from src.core.query_router import route_query, QueryRoute
assert route_query('displacement at 8.17m') == QueryRoute.TABLE_LOOKUP
assert route_query('all stability particulars at 8.33m') == QueryRoute.MULTI_LOOKUP
assert route_query('displacement at 7.57m fresh water') == QueryRoute.UNIT_CONVERT
assert route_query('what does GM mean') == QueryRoute.EXPLAIN
assert route_query('summarise the booklet') == QueryRoute.SYNTHESIS
print('PASS: all router routes correct')
"

# 5. Test CoT separation via API
curl -s -X POST http://localhost:8765/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"test1","module":"ragforge","message":"how does stability work?"}' \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
assert 'thinking' in data, 'Missing thinking field'
assert data['thinking'], 'Thinking field is empty'
assert not data['response'].startswith('Step'), 'CoT still in response'
assert not data['response'].startswith('Reasoning:'), 'CoT still in response'
print('PASS: CoT separated correctly')
print('Thinking excerpt:', data['thinking'][:100])
print('Response excerpt:', data['response'][:100])
"

# 6. Test numerical accuracy (requires HA-13 PDF to be ingested)
curl -s -X POST http://localhost:8765/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"test2","module":"ragforge","message":"what is the displacement of MV Primrose Ace at draft 8.17m salt water"}' \
  | python3 -c "
import json,sys,re
data=json.load(sys.stdin)
nums = re.findall(r'25[,.]?8\d\d', data['response'])
assert nums, f'Expected ~25839 in response. Got: {data[\"response\"][:300]}'
print('PASS: correct displacement found:', nums)
"
```

### Frontend tests
```bash
# 7. TypeScript compilation
npm run type-check

# 8. Lint
npm run lint

# 9. Build
npm run build
```

### Rust build test
```bash
# 10. Rust compilation
cargo build --release
```

### Manual integration tests
```
11. Launch the app: ./run_dev.sh && npm run tauri:dev

12. In RAGForge module ask: "What is the displacement at 8.17m?"
    ✓ A collapsed "Thought for Xs" block appears above the answer
    ✓ The answer text shows ~25,839 tonnes (not a hallucinated value)
    ✓ Clicking the thinking block expands the reasoning steps
    ✓ No reasoning text in the main answer area

13. Ask: "All stability particulars at 8.33m"
    ✓ Table with displacement, TPC, MTC, KM, LCB, LCF all shown
    ✓ No infinite loop
    ✓ Response under 5 seconds

14. Ask: "What does GM mean?"
    ✓ Routes to explain pipeline (not calc engine)
    ✓ No ThinkingBlock if no reasoning was generated
    ✓ Clear conceptual explanation

15. Click any suggestion tile:
    ✓ Message submits immediately (no Enter needed)
    ✓ All tiles disappear after click
    ✓ Each tile shows ↗ arrow
    ✓ Tiles are greyed out while response is generating

16. X-Ray mode:
    ✓ X-Ray panel still works normally
    ✓ Toggle X-Ray ON/OFF functions as before

17. SAMR test: ask something ambiguous
    ✓ If faithfulness < 0.55, receives safe refusal message
    ✓ No hallucinated answer delivered with a warning badge
```

---

## PART 6 — DO NOT CHANGE (PROTECTED LIST)

These components must not be modified by any sub-agent:

| Component | File(s) | Reason |
|---|---|---|
| OPLoRA SVD math | `src/learning/oplora_manager.py` | Mathematically verified — supplement only |
| OPA Rego policies | `src/guardrails/default_policies.rego` | Policy content is user-configurable |
| LangGraph graph structure | `src/meta_agent.py` | Add new nodes only — don't restructure |
| X-Ray causal trace | frontend X-Ray panel | Leave untouched |
| Silicon Colosseum FSM | `src/guardrails/` | FSM state transitions unchanged |
| StreamSync module | `src/modules/streamsync/` | Unrelated feature |
| WatchTower module | `src/modules/watchtower/` | Unrelated feature |
| LocalBuddy module | `src/modules/localbuddy/` | Unrelated feature |
| Replay Buffer | `src/learning/replay_buffer.py` | Parquet/Fernet format unchanged |
| Nightly scheduler | `src/main.py` | APScheduler jobs unchanged |
| BGE-M3 embeddings | `pyproject.toml` sentence-transformers | Already upgraded — keep |
| Docling chunking | `ragforge_indexer.py` chunking code | Already improved — keep |
| VLM processor | `src/modules/ragforge/vlm_processor.py` | Already added — keep |

---

*AetherForge Complete Sub-Agent Brief v3.0*
*11 sub-agents | 17 files modified | 9 new files created | 1 model download*
*Sub-agents: SA-01 through SA-11 | For: Claude Opus 4.6 / Sonnet 4.6 in Antigravity*
