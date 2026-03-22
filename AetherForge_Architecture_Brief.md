# AetherForge — Complete Architectural Design & Enhancement Brief
**Version 2.0 | Status: Active Re-architecture**
*Authored by: Elite Staff AI Engineer + AI Research Scientist*

---

## PART 1 — WHAT IS AETHERFORGE (CURRENT STATE)

AetherForge v1.2 is a **Sovereign Intelligence OS** — a perpetual-learning, fully local, glass-box AI system built as a Tauri desktop app (macOS-first, Apple Silicon optimised). It runs 100% on-device with no cloud dependency.

### Current Technology Stack

| Layer | Component | Technology |
|---|---|---|
| Desktop Shell | Tauri 2.1 | Rust + WebView |
| Frontend | React 18 + Shadcn/Tailwind | TypeScript 5.5 |
| Backend | FastAPI + LangGraph | Python 3.12 |
| LLM Inference | llama-cpp-python (Metal) | BitNet 1.58-bit 2B |
| Vector Search | ChromaDB (dense only) | Python |
| Sparse Search | SQLite FTS5 | **BROKEN** — never loads at runtime |
| Embeddings | all-MiniLM-L6-v2 → BGE-M3 | 384-dim → 1024-dim |
| Faithfulness | SAMR-lite (cosine similarity) | Warn-and-deliver |
| Guardrails | OPA Rego + FSM | Python/Rego |
| Learning | OPLoRA (SVD-based) | Nightly batch only |
| PDF Parsing | Docling | Upgraded (structural chunking) |
| Storage | SQLite + ChromaDB + Parquet | Encrypted |

### Current Architecture Flow

```
User Query
  └─► Silicon Colosseum pre-flight (OPA + FSM)
      └─► LangGraph Meta-Agent Supervisor
          └─► CognitiveRAG 7-Stage Pipeline
              ① Query Understanding (intent: Factual/Synthesis/Comparative)
              ② Decomposition (DAG of sub-questions)
              ③ HyDE (hypothetical document generation)
              ④ Hybrid Search (ChromaDB dense + FTS5 sparse [BROKEN])
              ⑤ Evidence Scoring (relevance ranking)
              ⑥ Chain-of-Thought synthesis (BitNet 2B)
              ⑦ Self-Verification (SAMR-lite cosine ≥ 0.45 → warn badge)
          └─► Response to UI + X-Ray trace
              └─► Replay Buffer (async, encrypted Parquet)
                  └─► OPLoRA Nightly Batch (3 AM, SVD projection)
```

---

## PART 2 — ROOT CAUSE FAILURES (DIAGNOSED)

These failures were observed in a live session where a Navy Chief Officer queried the HA - 13 LOADING AND STABILITY INFORMATION BOOKLET.pdf:

| Query | What Happened | Root Cause |
|---|---|---|
| "Calculate all stability particulars at 8.33m" | Repeated "Let's proceed..." 6× — no answer | LLM looped; no calc engine; no router |
| "Displacement at 8.17m salt water" | Invented 647,850 kg (correct: 25,839 t) | No structured table data; BM25 dead; BitNet 2B too small |
| "Interpolate displacement at 8.33m" | Called unregistered tool `get_data` — crash | Tool hallucinated; no deterministic engine |
| "Displacement at 7.57m fresh water" | Invented 48m draft, 8.9 trillion kg | BitNet 2B numerical confabulation |
| "Find displacement at 8.17m" (corrected) | "2500 tons" repeated 13× — infinite loop | No repetition guard; wrong by 10× |
| "Check hydrostatic table for 8.17m" | "Please share the relevant sections" — gave up | Table indexed as text blob; SQL rows empty |

### Three Core Architecture Defects

1. **No deterministic calculation engine.** All queries — including pure table lookups — route through the LLM. The LLM cannot reliably interpolate from a table it cannot structurally read.

2. **Table data not extracted as structured data.** Docling indexes 5,302 text chunks including 117 "table" chunks. These are text strings, not typed SQL rows. The hydrostatic table `(draft REAL, displacement REAL, tpc REAL...)` does not exist in queryable form.

3. **Hybrid search is dense-only.** BM25 (FTS5) returns 0 results at query time because the sparse index is not loaded when queries arrive. Exact-match retrieval for "8.17" or "25551" never fires.

---

## PART 3 — TARGET ARCHITECTURE (POST-ENHANCEMENT)

### 3.1 Complete Target Stack

| Layer | Old | New |
|---|---|---|
| LLM Runtime | llama-cpp-python + BitNet 2B | ruvllm + Qwen2.5-7B-Q4_K_M |
| Vector Search | ChromaDB (dense-only) | RuVector GNN-HNSW (self-learning hybrid) |
| Query Router | None | Tiny Dancer semantic router (<1ms) |
| Calc Engine | None | Deterministic Python interpolator |
| Table Storage | Text chunks | SQLite typed rows (draft, displacement…) |
| Faithfulness | SAMR-lite warn | Prime Radiant coherence gate (block) |
| Learning | OPLoRA nightly batch | SONA 3-tier (MicroLoRA + EWC++ + ReasoningBank) |
| Document Graph | None | Cypher graph engine (RuVector graph) |
| RAG Speed | Full 7-stage every query | RefRag (~30× on cache hits) |
| Knowledge Packaging | Scattered files | RVF cognitive containers (.rvf) |
| Rust Hot Path | reqwest HTTP to Ollama | Native ruvllm + ruvector-core in Tauri |

### 3.2 Target Architecture Flow

```
User Query
  └─► Tiny Dancer Semantic Router (<1ms, pre-LLM)
      │
      ├─► [table_lookup / interpolate / unit_convert]
      │   └─► SQLite Calc Engine (DETERMINISTIC — LLM never invoked)
      │       └─► linear_interpolate(d, d1, d2, v1, v2)
      │           └─► Prime Radiant number trace verification
      │               └─► LLM writes explanation only (not numbers)
      │
      ├─► [compliance_check / procedure]
      │   └─► Document graph Cypher query + RAG pipeline
      │
      └─► [explain / synthesis / compare]
          └─► RefRag wrapper
              └─► CognitiveRAG 7-Stage (unchanged logic)
                  ④ RuVector GNN-HNSW (dense + BM25 + Cypher, always live)
                  ⑦ Prime Radiant sheaf energy (BLOCK if energy > 0.70)
          └─► ruvllm (Qwen2.5-7B, 16k ctx, Metal, rep_penalty=1.15)
          └─► Response to UI + X-Ray trace
              └─► SONA feedback loop
                  ├─► MicroLoRA rank-2 (<1ms, per-request)
                  ├─► EWC++ consolidation (~100ms background)
                  └─► ReasoningBank (trajectory curriculum)
                      └─► RVF cognitive container export (snapshot)
```

### 3.3 The Core Principle (Unchanged)

> **LLMs explain. Deterministic engines calculate. Citations prove.**
>
> The LLM must never perform arithmetic where precision matters. It receives pre-computed results from the deterministic engine and writes the explanation. Any number in the LLM response that does not appear verbatim in the calculation trace is blocked before delivery.

---

## PART 4 — THE 8 INTEGRATIONS (FULL SPECIFICATION)

### Integration 1 — RuVector GNN-HNSW replaces ChromaDB
**File:** `src/modules/ragforge/vector_store.py`
**Status in repo:** COMMENTED OUT — not active
**What it solves:** Dead BM25 index, dense-only search, no query-level learning

```python
# REMOVE: "chromadb>=0.5.0" from pyproject.toml dependencies
# ADD (uncomment): "ruvector>=0.1.0"

from ruvector import RuVectorClient

class RuVectorStore:
    def __init__(self):
        self.client = RuVectorClient(dimensions=1024, gnn=True)  # BGE-M3 = 1024-dim

    async def search(self, embedding, top_k=12):
        return await self.client.hybrid_search(
            query=embedding, k=top_k,
            hybrid_alpha=0.7,  # 70% dense, 30% BM25
            rerank=True        # GNN reranking
        )

    async def add(self, embedding, text, metadata):
        return await self.client.upsert(embedding, text, metadata)
```

**Also update** `src/core/container.py`:
```python
# Replace: VectorStore = ChromaVectorStore(path=CHROMA_PATH)
VectorStore = RuVectorStore()
```

---

### Integration 2 — Prime Radiant replaces SAMR-lite
**File:** `src/guardrails/samr_lite.py`
**Status in repo:** COMMENTED OUT — SAMR-lite still active with wrong threshold (0.45)
**What it solves:** Hallucinations delivered with warning badge; threshold regression

```python
# REMOVE: "# prime-radiant-py>=0.1.0" comment — uncomment it
# CRITICAL: also raise threshold back to 0.55

from prime_radiant import CoherenceGate

gate = CoherenceGate(gpu=True)  # Metal on M1

async def verify_response(response_text: str, evidence_chunks: list) -> float:
    energy = await gate.sheaf_energy(response_text, evidence_chunks)
    if energy > 0.70:
        raise CoherenceViolation(f"Sheaf energy {energy:.2f} exceeds threshold")
    elif energy > 0.40:
        extra = await retrieve_more_evidence(response_text)
        return await verify_response(response_text, evidence_chunks + extra)
    return energy  # approved
```

**Critical config fix in `.env`:**
```
# WRONG (current — allows hallucinations through):
SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.45

# CORRECT:
SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.55
SILICON_COLOSSEUM_FAITHFULNESS_ACTION=block  # not warn
```

---

### Integration 3 — SONA 3-tier learning replaces OPLoRA batch
**File:** `src/learning/sona_adapter.py` (NEW FILE)
**Status in repo:** COMMENTED OUT — OPLoRA nightly batch still only learning path
**What it solves:** No real-time learning, nightly-only adaptation, no trajectory memory

```python
# ADD (uncomment): "ruvector-sona>=0.1.0"

from ruvector_sona import SONA

sona = SONA(
    model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf",
    micro_lora_rank=2,    # <1ms per-request adaptation
    ewc_lambda=400,       # catastrophic forgetting protection
    reasoning_bank=True   # trajectory curriculum memory
)

async def on_interaction_complete(query: str, response: str, verdict: str):
    """Called after every successful response. verdict = 'accepted' | 'rejected'"""
    await sona.micro_update(query, response, verdict)
    sona.reasoning_bank.record(query, response, verdict)
```

**Also keep OPLoRA** — SONA supplements it, doesn't remove it. OPLoRA runs Sunday deep training; SONA runs every request.

---

### Integration 4 — ruvllm + Qwen2.5-7B replaces BitNet 2B + Ollama
**File:** `src-tauri/src/ruvllm_bridge.rs` (NEW FILE) + `.env` update
**Status in repo:** Cargo.toml declared but no .rs file written; Python still uses llama-cpp-python
**What it solves:** Repetition loops, malformed JSON tool calls, numerical confabulation, 8192-token context cap

```rust
// src-tauri/src/ruvllm_bridge.rs
use ruvllm::{RuvLLM, GenerateOptions};
use tauri::State;

pub struct LLMState(pub RuvLLM);

#[tauri::command]
pub async fn llm_generate(
    state: State<'_, LLMState>,
    prompt: String,
) -> Result<String, String> {
    state.0.generate(&prompt, GenerateOptions {
        max_tokens: 1024,
        repetition_penalty: 1.15,  // kills infinite loops
        n_ctx: 16384,              // was 8192
    }).await.map_err(|e| e.to_string())
}
```

**Register in `src-tauri/src/lib.rs`:**
```rust
.manage(LLMState(RuvLLM::new("models/qwen2.5-7b-instruct-q4_k_m.gguf", "metal")))
.invoke_handler(tauri::generate_handler![llm_generate])
```

**Model to download:**
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q4_k_m.gguf" \
  --local-dir models/
```

---

### Integration 5 — Structured table extraction (MOST CRITICAL)
**File:** `src/modules/ragforge/ragforge_indexer.py` (MODIFY existing)
**Status in repo:** PARTIAL — Docling structural chunking added, but no table→SQL extraction
**What it solves:** Every single numerical query failure — no data exists to query

```python
# Add to ragforge_indexer.py — runs during PDF ingestion

import sqlite3
from docling.document_converter import DocumentConverter

def extract_tables_to_sqlite(doc, vessel_id: str, db_path: str):
    """Branch at ingestion: tables → SQLite rows, not text chunks."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hydrostatic (
            vessel_id TEXT, draft REAL, displacement REAL,
            tpc REAL, mtc REAL, km REAL, lcb REAL, lcf REAL
        )
    """)

    for page in doc.pages:
        if page.page_type == "table":
            headers, rows = parse_table_page(page)  # see below
            domain = classify_table_domain(headers)  # "hydrostatic" | "tank" | etc.

            if domain == "hydrostatic":
                for row in rows:
                    conn.execute(
                        "INSERT INTO hydrostatic VALUES (?,?,?,?,?,?,?,?)",
                        (vessel_id, row['draft'], row['displacement'],
                         row.get('tpc'), row.get('mtc'), row.get('km'),
                         row.get('lcb'), row.get('lcf'))
                    )
    conn.commit()
    conn.close()


def parse_table_page(page) -> tuple[list, list]:
    """Extract headers and typed rows from Docling table page."""
    table = page.tables[0]  # Docling exposes .tables
    headers = [cell.text.strip() for cell in table.header_cells]
    rows = []
    for row in table.data_rows:
        row_dict = {}
        for i, cell in enumerate(row):
            col_name = headers[i].lower().replace(' ', '_').replace('(', '').replace(')', '')
            try:
                row_dict[col_name] = float(cell.text.strip())
            except ValueError:
                row_dict[col_name] = cell.text.strip()
        rows.append(row_dict)
    return headers, rows


def classify_table_domain(headers: list) -> str:
    header_text = ' '.join(headers).lower()
    if any(w in header_text for w in ['draft', 'displacement', 'tpc']):
        return 'hydrostatic'
    if any(w in header_text for w in ['tank', 'capacity', 'volume']):
        return 'tank_capacity'
    return 'generic'
```

---

### Integration 6 — Deterministic Calculation Engine
**File:** `src/core/calc_engine.py` (NEW FILE)
**Status in repo:** NOT STARTED
**What it solves:** LLM computing numbers; interpolation errors; SG conversion errors

```python
# src/core/calc_engine.py

import sqlite3
from typing import Optional

class CalcEngine:
    def __init__(self, db_path: str):
        self.db = db_path

    def linear_interpolate(
        self, d: float, d1: float, d2: float, v1: float, v2: float
    ) -> float:
        """Pure deterministic linear interpolation. No LLM involved."""
        if d2 == d1:
            return v1
        frac = (d - d1) / (d2 - d1)
        return round(v1 + frac * (v2 - v1), 3)

    def lookup_hydrostatic(
        self, vessel_id: str, draft: float, column: str = "displacement"
    ) -> dict:
        """Return interpolated value + trace for any hydrostatic column."""
        conn = sqlite3.connect(self.db)
        rows = conn.execute(
            "SELECT draft, ? FROM hydrostatic WHERE vessel_id=? ORDER BY draft",
            (column, vessel_id)
        ).fetchall()
        conn.close()

        # Find bracketing rows
        lower = max((r for r in rows if r[0] <= draft), key=lambda r: r[0], default=None)
        upper = min((r for r in rows if r[0] >= draft), key=lambda r: r[0], default=None)

        if not lower or not upper:
            raise ValueError(f"Draft {draft}m outside table range")

        result = self.linear_interpolate(draft, lower[0], upper[0], lower[1], upper[1])

        return {
            "value": result,
            "unit": "tonnes" if column == "displacement" else column,
            "trace": {
                "method": "linear_interpolation",
                "lower_row": {"draft": lower[0], column: lower[1]},
                "upper_row": {"draft": upper[0], column: upper[1]},
                "fraction": round((draft - lower[0]) / (upper[0] - lower[0]), 4),
                "density": "salt_water_1.025"
            }
        }

    def apply_fw_correction(self, displacement_sw: float) -> dict:
        """Fresh water correction. SG ratio is deterministic."""
        displacement_fw = round(displacement_sw * (1.000 / 1.025), 2)
        return {
            "value": displacement_fw,
            "unit": "tonnes",
            "trace": {
                "method": "sg_correction",
                "input_sw": displacement_sw,
                "sg_sw": 1.025,
                "sg_fw": 1.000,
                "ratio": round(1.000 / 1.025, 6)
            }
        }
```

---

### Integration 7 — Tiny Dancer Semantic Query Router
**File:** `src/core/query_router.py` (NEW FILE)
**Status in repo:** NOT STARTED
**What it solves:** Calculation queries reaching the LLM; 200 wasted context tokens per query; generation loops

```python
# src/core/query_router.py
# Phase 1: keyword/regex router (works without Tiny Dancer installed)
# Phase 2: replace with Tiny Dancer FastGRNN when dep is available

import re
from enum import Enum

class QueryRoute(str, Enum):
    TABLE_LOOKUP   = "table_lookup"
    INTERPOLATE    = "interpolate"
    UNIT_CONVERT   = "unit_convert"
    COMPLIANCE     = "compliance_check"
    EXPLAIN        = "explain"
    PROCEDURE      = "procedure"
    SYNTHESIS      = "synthesis"

DRAFT_PATTERN = re.compile(r'\b\d+\.\d{1,2}\s*m\b', re.IGNORECASE)
CALC_KEYWORDS = {
    'displacement', 'draft', 'tpc', 'mtc', 'km', 'lcb', 'lcf',
    'interpolat', 'calculat', 'find the', 'what is the', 'stability particulars',
    'hydrostatic', 'fresh water', 'salt water'
}
EXPLAIN_KEYWORDS = {
    'what does', 'explain', 'define', 'meaning of', 'what is gm', 'describe'
}

def route_query(query: str) -> QueryRoute:
    q_lower = query.lower()
    has_draft = bool(DRAFT_PATTERN.search(query))
    has_calc = any(kw in q_lower for kw in CALC_KEYWORDS)
    has_explain = any(kw in q_lower for kw in EXPLAIN_KEYWORDS)

    if has_draft and has_calc:
        if 'interpolat' in q_lower or 'calculat' in q_lower:
            return QueryRoute.INTERPOLATE
        return QueryRoute.TABLE_LOOKUP

    if 'convert' in q_lower or 'fresh water' in q_lower:
        return QueryRoute.UNIT_CONVERT

    if has_explain and not has_draft:
        return QueryRoute.EXPLAIN

    return QueryRoute.SYNTHESIS  # default: full RAG pipeline


# In meta_agent.py — add BEFORE any LLM invocation:
# route = route_query(user_message)
# if route in (QueryRoute.TABLE_LOOKUP, QueryRoute.INTERPOLATE, QueryRoute.UNIT_CONVERT):
#     return calc_engine.lookup_hydrostatic(vessel_id, draft, column)
```

---

### Integration 8 — Cypher Graph Engine + RefRag + RVF Containers
**Status:** NOT STARTED — Phase 3 work, lower urgency than 1–7

**Cypher graph** (`src/modules/ragforge/graph_ingester.py`): At ingestion, create graph nodes for each document section and edges for cross-references. Enables multi-hop queries like "what does section 4 reference from section 2?"

**RefRag wrapper** (`src/modules/ragforge/refrag_pipeline.py`): Wrap existing CognitiveRAG with hash-fingerprint cache. Cache hits return in ~1ms instead of running the full 7-stage pipeline. ~30× latency improvement for repeated query patterns.

**RVF containers** (`src-tauri/src/rvf.rs`): Export entire knowledge state (vectors + LoRA adapters + GNN weights + sessions) as a single signed `.rvf` file. Portable snapshot of a trained AetherForge instance.

---

## PART 5 — CURRENT CONFORMANCE STATUS

### What Has Been Done ✅

| Item | File | Verdict |
|---|---|---|
| BGE-M3 embedding upgrade | pyproject.toml | ✅ Done — 384→1024 dim, 8192-token context |
| Docling structural chunking | ragforge_indexer.py | ✅ Done — sections/figures/equations respected |
| SAMR-lite creation | samr_lite.py | ✅ Done — exists and fires |
| RuVector Rust crates declared | Cargo.toml | ✅ Declared — ruvllm, ruvector-core, prime-radiant, rvf-runtime, ruvector-sona |
| VLM processor for scanned PDFs | vlm_processor.py | ✅ Done |

### What Is Partially Done ⚠️

| Item | File | What's Missing |
|---|---|---|
| Table-to-SQL extraction | ragforge_indexer.py | Docling used for structural chunking ✅ but table rows NOT written to SQLite columns ❌ |
| SAMR threshold fix | .env / samr_lite.py | Threshold moved 0.92→0.45 (wrong direction — regression) ❌; behaviour still warn-not-block ❌ |

### What Has NOT Been Done ❌

| Item | File Needed | Blocking? |
|---|---|---|
| RuVector Python deps uncommented | pyproject.toml | **YES** — nothing works until this is done |
| Cargo.toml wildcard versions fixed | Cargo.toml | **YES** — cargo build fails |
| ChromaDB replacement | container.py | YES — BM25 dead index unchanged |
| Query router | src/core/query_router.py | YES — calculation queries still reach LLM |
| Calc engine | src/core/calc_engine.py | YES — no deterministic interpolation |
| ruvllm Rust bridge | src-tauri/src/ruvllm_bridge.rs | YES — model still BitNet 2B via Ollama |
| Qwen2.5-7B model download | models/ | YES — need new model file |
| Prime Radiant dep uncommented | pyproject.toml | YES — SAMR still active |
| SONA learning adapter | src/learning/sona_adapter.py | Medium — OPLoRA still only path |
| Cypher graph ingestion | ragforge/graph_ingester.py | Low — Phase 3 |
| RefRag wrapper | ragforge/refrag_pipeline.py | Low — Phase 3 |
| RVF container bridge | src-tauri/src/rvf.rs | Low — Phase 3 |
| CLAUDE.md updated | CLAUDE.md | Medium — Claude Code reads stale architecture |
| README.md updated | README.md | Low |

---

## PART 6 — THE COMPLETE PROMPT

This is the exact prompt to give to Claude Code (or any developer) to implement all outstanding changes:

---

```
You are working on AetherForge — a Rust/Python/Tauri local AI OS.
The repo is at github.com/NeoOne601/AtherForge.

CONTEXT: A full architectural redesign has been planned. Some parts are complete,
most are not. Your job is to implement everything that is missing,
in the priority order listed below.

════════════════════════════════════════════
PHASE 1 — CRITICAL: unblock the runtime (do these first, in order)
════════════════════════════════════════════

[P1-A] pyproject.toml — uncomment the 6 RuVector Python dependencies:
  Remove the leading '#' from these 6 lines (lines ~97-103):
    # "ruvector>=0.1.0"
    # "ruvllm>=0.1.0"
    # "ruvector-sona>=0.1.0"
    # "ruvector-refrag>=0.1.0"
    # "prime-radiant-py>=0.1.0"
    # "tiny-dancer>=0.1.0"
  After editing, run: uv pip install -e ".[dev]"

[P1-B] Cargo.toml — fix wildcard versions. Replace:
    ruvector-core = "*"
    ruvllm = { version = "^2.0", ... }
    prime-radiant = "*"
    rvf-runtime = "*"
    ruvector-sona = "*"
  With exact versions from crates.io, or use git paths:
    ruvector-core = { git = "https://github.com/ruvnet/ruvector", package = "ruvector-core" }
    (apply same pattern to all 5 crates)

[P1-C] .env and samr_lite.py — fix the SAMR regression:
  1. In .env: change SILICON_COLOSSEUM_MIN_FAITHFULNESS from 0.45 back to 0.55
  2. In src/guardrails/samr_lite.py: change the low-confidence action from
     returning a warning badge to raising an exception:
     BEFORE: return {"flagged": True, "badge": "⚠️ LOW CONFIDENCE", "score": score}
     AFTER:  raise FaithfulnessError(f"Score {score:.2f} below threshold 0.55")
  3. Handle FaithfulnessError in meta_agent.py to return a safe refusal message.

════════════════════════════════════════════
PHASE 2 — CORE: the anti-hallucination triad
════════════════════════════════════════════

[P2-A] CREATE src/modules/ragforge/table_extractor.py:
  At PDF ingestion time, when Docling classifies a page as a table:
  - Extract headers and typed rows from page.tables[0]
  - Classify domain by header keywords: "hydrostatic" | "tank_capacity" | "kn_table" | "generic"
  - For hydrostatic tables: INSERT rows into SQLite table:
      hydrostatic(vessel_id TEXT, draft REAL, displacement REAL,
                  tpc REAL, mtc REAL, km REAL, lcb REAL, lcf REAL)
  - Call extract_tables_to_sqlite(doc, vessel_id, db_path) from ragforge_indexer.py
    AFTER the existing Docling parse step.
  - The existing text-chunk pipeline still runs for non-table pages — do not remove it.

[P2-B] CREATE src/core/calc_engine.py:
  Implement CalcEngine class with these methods:
  - linear_interpolate(d, d1, d2, v1, v2) → float
    Formula: v1 + ((d - d1) / (d2 - d1)) * (v2 - v1)
    Round result to 3 decimal places. No LLM. Pure math.
  - lookup_hydrostatic(vessel_id, draft, column="displacement") → dict
    Query SQLite for two bracketing rows (lower ≤ draft ≤ upper).
    Interpolate. Return {"value": result, "unit": "tonnes", "trace": {...}}
    Raise ValueError if draft is outside the table range.
  - apply_fw_correction(displacement_sw) → dict
    Δ_FW = Δ_SW × (1.000 / 1.025). Return value + trace.
  - apply_sg_correction(displacement, sg_actual) → dict
    Δ_corrected = Δ_SW × (sg_actual / 1.025). Return value + trace.

[P2-C] CREATE src/core/query_router.py:
  Implement route_query(query: str) → QueryRoute using keyword+regex:
  - If query contains a decimal draft value (regex: \d+\.\d{1,2}\s*m) AND
    any of: displacement|draft|tpc|mtc|km|lcb|lcf|stability particulars|hydrostatic
    → return QueryRoute.TABLE_LOOKUP or INTERPOLATE
  - If query contains "fresh water" or "convert" → QueryRoute.UNIT_CONVERT
  - If query contains "explain"|"what does"|"define"|"describe" without a draft value
    → QueryRoute.EXPLAIN
  - Default: QueryRoute.SYNTHESIS (full RAG pipeline)

  In src/meta_agent.py, add the router as the FIRST node in the LangGraph,
  before any LLM call:
    route = route_query(user_message)
    if route in (TABLE_LOOKUP, INTERPOLATE, UNIT_CONVERT):
        result = calc_engine.lookup_or_convert(route, user_message)
        # LLM only writes the explanation wrapper:
        return llm.explain_result(result["value"], result["trace"])

════════════════════════════════════════════
PHASE 3 — UPGRADE: vector store and inference
════════════════════════════════════════════

[P3-A] CREATE src/modules/ragforge/vector_store.py (replace ChromaDB):
  Implement RuVectorStore class:
  - __init__: self.client = RuVectorClient(dimensions=1024, gnn=True)
  - async search(embedding, top_k=12): hybrid_search with hybrid_alpha=0.7, rerank=True
  - async add(embedding, text, metadata): client.upsert(...)
  In src/core/container.py: replace ChromaVectorStore() with RuVectorStore()
  Update CHROMA_PATH env var to RUVECTOR_PATH.

[P3-B] CREATE src-tauri/src/ruvllm_bridge.rs:
  - Import ruvllm::{RuvLLM, GenerateOptions}
  - Create LLMState(pub RuvLLM) struct
  - Implement #[tauri::command] llm_generate(prompt, options) → String
    with max_tokens=1024, repetition_penalty=1.15, n_ctx=16384
  - Register in src-tauri/src/lib.rs:
    .manage(LLMState(RuvLLM::new("models/qwen2.5-7b-instruct-q4_k_m.gguf", "metal")))
    .invoke_handler(tauri::generate_handler![llm_generate])
  Download model:
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
      --include "qwen2.5-7b-instruct-q4_k_m.gguf" --local-dir models/

[P3-C] CREATE src/guardrails/coherence_gate.py (replace samr_lite for numbers):
  For queries routed through the calc engine (TABLE_LOOKUP etc.):
    - After LLM generates the explanation, extract all numbers from the response
    - Verify each number appears in the calc trace dict
    - If any number in the response does NOT appear in the trace: block and return error
  This is simpler than full Prime Radiant — implement it as a pure Python number verifier:
    numbers_in_response = set(re.findall(r'\b\d+\.?\d*\b', response))
    numbers_in_trace = set(re.findall(r'\b\d+\.?\d*\b', str(trace)))
    unauthorized = numbers_in_response - numbers_in_trace
    if unauthorized:
        raise NumberVerificationError(f"Response contains untraced numbers: {unauthorized}")

════════════════════════════════════════════
PHASE 4 — LEARNING: upgrade OPLoRA to SONA
════════════════════════════════════════════

[P4-A] CREATE src/learning/sona_adapter.py:
  Implement SONAAdapter class wrapping ruvector_sona.SONA:
  - __init__: SONA(model_path, micro_lora_rank=2, ewc_lambda=400, reasoning_bank=True)
  - async on_interaction(query, response, verdict): micro_update + reasoning_bank.record
  - Keep existing OPLoRA nightly batch — SONA supplements it, does not replace it
  Wire into meta_agent.py: call on_interaction() after every accepted response.

════════════════════════════════════════════
PHASE 5 — DOCUMENTATION: update project context
════════════════════════════════════════════

[P5-A] UPDATE CLAUDE.md:
  Replace all references to:
    ChromaDB → RuVector GNN-HNSW
    BitNet / llama-cpp-python → ruvllm + Qwen2.5-7B
    SAMR-lite warn → Prime Radiant / coherence gate (block)
    OPLoRA nightly only → SONA 3-tier + OPLoRA
  Add new components to the architecture section:
    - CalcEngine (src/core/calc_engine.py)
    - QueryRouter (src/core/query_router.py)
    - TableExtractor (src/modules/ragforge/table_extractor.py)
    - RuVectorStore (src/modules/ragforge/vector_store.py)
    - ruvllm_bridge.rs (src-tauri/src/)

════════════════════════════════════════════
ACCEPTANCE CRITERIA
════════════════════════════════════════════

The implementation is complete when ALL of these pass:

1. pytest tests/ passes with no failures after uv pip install -e .
2. cargo build --release succeeds with no errors
3. Query "what is the displacement of Primrose Ace at draft 8.17m salt water?"
   returns 25,839 tonnes (±1t) with a calc trace showing the two bracketing rows
   and the linear interpolation fraction.
4. Query "what is the displacement at 7.57m fresh water?"
   returns the correct fresh water value with SG correction trace shown.
5. Query "explain what GM means" routes to the explain pipeline (not calc engine)
   and returns a coherent explanation without triggering the number verifier.
6. No response containing a number that wasn't in the source document
   reaches the user (all blocked by coherence_gate.py).
7. ChromaDB is not imported anywhere in the active code path.
8. llama-cpp-python is not called in the active inference path.

════════════════════════════════════════════
DO NOT CHANGE
════════════════════════════════════════════

- OPLoRA SVD math (src/learning/oplora_manager.py) — keep and supplement with SONA
- OPA Rego policies (src/guardrails/default_policies.rego) — unchanged
- LangGraph supervisor structure (src/meta_agent.py) — add router as first node only
- X-Ray causal trace system — unchanged
- Tauri frontend (frontend/) — unchanged
- Silicon Colosseum FSM — unchanged
- StreamSync, WatchTower, LocalBuddy modules — unchanged
- Replay Buffer (Parquet/Fernet) — unchanged
```

---

## PART 7 — PHASED TIMELINE

| Phase | Week | Items | Outcome |
|---|---|---|---|
| **Phase 1** | Week 1 | P1-A, P1-B, P1-C | Runtime unblocked; SAMR regression fixed |
| **Phase 2** | Week 1–2 | P2-A, P2-B, P2-C | Calculation queries answered correctly; no LLM for numbers |
| **Phase 3** | Week 2–3 | P3-A, P3-B, P3-C | ChromaDB replaced; ruvllm active; number verifier live |
| **Phase 4** | Week 4 | P4-A | Real-time learning added; OPLoRA supplemented |
| **Phase 5** | Week 4 | P5-A | CLAUDE.md correct; future Claude Code sessions give right guidance |
| **Phase 6** | Week 5–6 | Integration 8 | Graph engine, RefRag, RVF containers |

---

## PART 8 — QUICK REFERENCE: FILE CHANGES MAP

| File | Action | Priority |
|---|---|---|
| `pyproject.toml` | Uncomment 6 RuVector deps | P1 CRITICAL |
| `Cargo.toml` | Fix wildcard versions | P1 CRITICAL |
| `.env` | SAMR threshold 0.45→0.55 + action=block | P1 CRITICAL |
| `src/guardrails/samr_lite.py` | Change warn→raise exception | P1 CRITICAL |
| `src/modules/ragforge/table_extractor.py` | CREATE — table→SQLite rows | P2 CRITICAL |
| `src/core/calc_engine.py` | CREATE — deterministic interpolator | P2 CRITICAL |
| `src/core/query_router.py` | CREATE — intent classifier | P2 CRITICAL |
| `src/meta_agent.py` | ADD router as first LangGraph node | P2 CRITICAL |
| `src/modules/ragforge/vector_store.py` | REPLACE ChromaDB→RuVectorStore | P3 HIGH |
| `src/core/container.py` | SWAP ChromaVectorStore→RuVectorStore | P3 HIGH |
| `src-tauri/src/ruvllm_bridge.rs` | CREATE — Rust LLM bridge | P3 HIGH |
| `src-tauri/src/lib.rs` | REGISTER ruvllm Tauri command | P3 HIGH |
| `src/guardrails/coherence_gate.py` | CREATE — number trace verifier | P3 HIGH |
| `src/learning/sona_adapter.py` | CREATE — SONA 3-tier wrapper | P4 MEDIUM |
| `CLAUDE.md` | UPDATE — new architecture description | P5 MEDIUM |

---

*End of brief. Total files to create: 6. Files to modify: 9. Model to download: 1.*
*Estimated implementation time: 3–4 weeks for a solo developer following this brief.*
