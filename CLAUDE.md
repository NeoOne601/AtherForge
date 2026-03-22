# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AetherForge is a **Sovereign Intelligence OS** — a desktop-native AI system that learns, reasons, and calculates entirely on-device. It implements a Closed-Loop Perpetual Learning architecture with air-gapped security, bridging high-performance LLMs with edge-device privacy requirements.

### Core Design Principles
1. **"LLMs explain; they never calculate."** All numeric queries are routed to the deterministic CalcEngine before any LLM call.
2. **Glass-Box Reasoning.** Every decision exposes full `<think>` reasoning traces, auditable causal graphs, and SAMR-lite faithfulness scores.
3. **Perpetual Learning.** OPLoRA (nightly) + SONA (per-request) ensure the system learns without catastrophic forgetting.
4. **Air-Gapped Security.** Zero external API calls. SQLCipher encryption. No telemetry.

---

## Technology Stack

| Layer | Technology |
|:------|:-----------|
| Desktop Shell | Tauri 2.1 (Rust) |
| Frontend | React 18 + TypeScript 5.5 + Vite |
| Backend | Python 3.12 + FastAPI + Uvicorn |
| Orchestration | LangGraph + LangChain Core |
| LLM Inference | ruvllm (Rust GGUF) + llama-cpp-python (Python fallback) |
| Primary Vector Store | **RuVector GNN-HNSW** (NPM CLI → .rvf binary files) |
| Sparse Search | SQLite FTS5 (BM25) |
| Structured Data | SQLite (hydrostatic tables for CalcEngine) |
| Document Processing | IBM Docling + PyMuPDF + VLM (SmolVLM / Florence-2) |
| Embeddings | all-MiniLM-L6-v2 (384-dim, cosine similarity) |
| Learning | OPLoRA (SVD orthogonal projection) + SONA 3-tier |
| Guardrails | Silicon Colosseum (OPA/Rego + FSM) |
| Encryption | SQLCipher (AES-256) |

---

## Repository Structure

```
AtherForge/
├── src/                           # Python backend (FastAPI)
│   ├── core/                      # DI Container, CalcEngine, QueryRouter, Grammar
│   │   ├── container.py           # Central dependency injection & service lifecycle
│   │   ├── calc_engine.py         # Deterministic table interpolation (no LLM math)
│   │   └── query_router.py        # Intent classifier (fires BEFORE any LLM call)
│   ├── guardrails/                # Silicon Colosseum
│   │   ├── silicon_colosseum.py   # OPA/Rego policy enforcement
│   │   └── coherence_gate.py     # Post-generation number trace verification
│   ├── learning/                  # Continual Learning
│   │   ├── oplora_trainer.py      # Orthogonal Projection LoRA (nightly batch)
│   │   ├── sona_adapter.py        # SONA 3-tier real-time learning (optional)
│   │   ├── replay_buffer.py       # Encrypted interaction storage (Parquet/Fernet)
│   │   ├── history_manager.py     # Conversation history management
│   │   └── evolution.py           # AetherResearcher iterative experiment loop
│   ├── modules/                   # Plugin Modules
│   │   ├── ragforge/              # CognitiveRAG™ pipeline
│   │   │   ├── ruvector_store.py  # RuVector CLI bridge (LangChain VectorStore)
│   │   │   ├── cognitive_rag.py   # 7-stage reasoning pipeline
│   │   │   ├── sparse_index.py    # SQLite FTS5 BM25 search
│   │   │   ├── vlm_enrich.py      # VLM visual extraction for scanned PDFs
│   │   │   └── table_extractor.py # Tables → SQLite at ingestion time
│   │   ├── ragforge_indexer.py    # Precision Ingestion™ pipeline
│   │   ├── document_registry.py   # SQLite doc metadata + boot-sweep purge
│   │   ├── session_store.py       # SQLCipher encrypted sessions
│   │   ├── export_engine.py       # PDF/Markdown export
│   │   ├── analytics/             # Usage statistics module
│   │   ├── streamsync/            # LiveFolder watcher + RSS feeder
│   │   ├── sync/                  # P2P encrypted sync (SyncManager)
│   │   ├── tunelab/               # Learning monitor
│   │   ├── localbuddy/            # Local assistant
│   │   └── watchtower/            # System observability
│   ├── routers/                   # FastAPI route handlers
│   ├── services/                  # Business logic
│   │   ├── chat_turns.py          # Turn execution, reasoning summary, suggestions
│   │   └── document_intelligence.py # Document upload, VLM processing manager
│   ├── meta_agent.py              # LangGraph Supervisor (the brain — 2800+ lines)
│   ├── chat_contract.py           # Shared chat protocol utilities
│   ├── config.py                  # AetherForgeSettings (Pydantic)
│   └── main.py                    # Entry point + CLI
├── frontend/                      # React/Vite/TypeScript HUD
│   └── src/components/            # ThinkingBlock, X-Ray, TuneLab, DocumentPanel
├── src-tauri/src/                 # Rust Tauri shell
│   ├── ruvllm_bridge.rs           # Native GGUF inference via Tauri commands
│   └── lib.rs                     # Tauri plugin registration
├── data/                          # Persistent Storage (encrypted)
│   ├── LiveFolder/                # Drop files here for auto-ingestion
│   ├── uploads/                   # REST API uploaded files
│   ├── ruvector/                  # .rvf vector database files
│   ├── document_registry.db       # File-level metadata (SQLite)
│   ├── sparse_index.db            # FTS5 BM25 index (SQLite)
│   ├── structured_data.db         # Hydrostatic tables (SQLite)
│   └── sessions.db                # Encrypted chat sessions (SQLCipher)
├── models/                        # LLM weights
└── tests/                         # Unit and integration tests
```

---

## Development Commands

### Installation
```bash
chmod +x install.sh && ./install.sh
```

### Running Development Server
```bash
# Full stack (backend + frontend + Tauri)
./run_dev.sh

# Backend only
.venv/bin/python -m uvicorn src.main:app --host 127.0.0.1 --port 8765 --reload

# Frontend only (web)
npm run dev

# Desktop app (Tauri)
npm run tauri:dev
```

### Testing
```bash
pytest                                          # All tests
pytest tests/test_chat_contract.py              # Specific file
pytest --cov=src                                # With coverage
pytest -v -k "test_reasoning"                   # Pattern match
```

### Linting & Formatting
```bash
ruff check src           # Lint
ruff check --fix src     # Auto-fix
mypy src                 # Type checking
black src                # Format
```

### Building
```bash
uv build                 # Python package
npm run tauri:build       # Tauri desktop app
npm run build             # Frontend assets
```

---

## Architecture Deep Dive

### Request Flow (Chat Turn)

```
User Message → frontend → POST /api/v1/ragforge/chat
  → chat_turns.execute_turn()
    → state.meta_agent.run(MetaAgentInput)
      → _run_sync()
        → 1. Silicon Colosseum preflight (OPA policy check)
        → 2. QueryRouter classifies intent
        │     ├── TABLE_LOOKUP / MULTI_LOOKUP / INTERPOLATE / UNIT_CONVERT
        │     │     → CalcEngine (deterministic SQLite interpolation)
        │     │     → CoherenceGate (verify every number)
        │     │     → Return formatted result
        │     └── General / RAG query
        │           → _hybrid_search() [branches on vector store type]
        │           │   ├── RuVectorStore: GNN-HNSW unified search
        │           │   └── Fallback: dense + FTS5 sparse fusion
        │           → CognitiveRAG 7-stage pipeline
        │           → _run_llm_sync() (ruvllm or llama-cpp-python)
        │           → SAMR-lite faithfulness check
        → 3. Post-flight: build reasoning trace, citations, suggestions
  → ChatResponse (with ThinkingBlock, citations, faithfulness_score)
```

### Vector Store Architecture (RuVector)

**RuVectorStore** (`src/modules/ragforge/ruvector_store.py`) implements the LangChain `VectorStore` interface, bridging Python to the RuVector NPM CLI:

| Method | Implementation |
|:-------|:---------------|
| `add_texts()` | Writes texts + metadata to a `.jsonl` temp file, calls `npx ruvector rvf ingest <.rvf> -d <.jsonl>` |
| `similarity_search()` | Embeds query, calls `npx ruvector rvf query <.rvf> -q <embedding> -k <k>`, parses JSON results |
| `get()` | Returns empty — RuVector CLI doesn't support metadata-only queries (debug log, no warning) |
| `delete()` | Writes IDs to temp JSON, calls `npx ruvector rvf delete <.rvf> -i <ids.json>` |
| `_rvf_path()` | Returns `<persist_directory>/main.rvf` |

**Key design decision**: The `_hybrid_search()` method in `meta_agent.py` checks `type(self.vector_store).__name__` at runtime. If the store is `RuVectorStore`, it uses RuVector's unified GNN-HNSW search (which internally fuses dense + sparse). Otherwise, it falls back to the dense + FTS5 hybrid path with Reciprocal Rank Fusion.

### Ingestion Pipeline (ragforge_indexer.py)

```
File → Idempotency Guard (check mtime vs registry)
  → Dedup: vector_store.get(where={source}) → .delete(ids)
  → Smart Loading:
  │   ├── Digital PDF: IBM Docling (tables, equations, reading order)
  │   ├── Scanned PDF: PyMuPDF page images → VLM visual extraction
  │   └── Text/MD/CSV: Direct load
  → Semantic Chunking (section/table/equation boundaries)
  → Progressive Commit:
  │   ├── vector_store.add_documents(batch) → RuVector .rvf
  │   ├── sparse_index.add_documents(batch) → FTS5 SQLite
  │   └── TableExtractor → structured_data.db (for CalcEngine)
  → Document Registry update (status, chunk_count, mtime)
```

### Learning Architecture

**Replay Buffer** → stores every accepted response (Parquet + Fernet encryption)

**OPLoRA Nightly** (3 AM batch):
1. Read day's replay buffer
2. Compute SVD of current weight subspace
3. Build orthogonal projector: P = I − UₖUₖᵀ
4. Apply projected gradient updates: ΔW_safe = P_left · ΔW · P_right
5. Merge into model weights

**SONA Per-Request** (optional, requires `ruvector-sona`):
- Tier 1: MicroLoRA rank-2 (<1ms adaptation)
- Tier 2: EWC++ consolidation (prevent forgetting between tiers)
- Tier 3: ReasoningBank (store successful trajectories as curriculum)

### Boot-Sweep (Startup Housekeeping)

In `app_factory.py` lifespan → `document_registry.purge_missing_files()`:
1. Reads all records from `document_registry.db`
2. Cross-references each `source_path` against `data/LiveFolder/` and `data/uploads/`
3. Deletes records where the file no longer exists on disk
4. Logs count of purged ghost documents

### CalcEngine & QueryRouter

**QueryRouter** (`src/core/query_router.py`):
- Uses regex patterns to classify query intent: `TABLE_LOOKUP`, `MULTI_LOOKUP`, `INTERPOLATE`, `UNIT_CONVERT`, or `GENERAL`
- `extract_draft()`: Extracts draft value in metres from query text
- `extract_column()`: Maps keywords to column names (e.g., "TPC" → "tpc", "TPC and MTC" → "multi")
- `extract_sg()`: Extracts specific gravity for dock water corrections

**CalcEngine** (`src/core/calc_engine.py`):
- `lookup_hydrostatic(vessel_id, draft, column)`: Exact lookup or linear interpolation from SQLite
- `lookup_all_hydrostatic(vessel_id, draft)`: All columns at once
- `apply_sg_correction(sw_value, sg)`: Salt water → dock water correction
- `apply_fw_correction(sw_value)`: Salt water → fresh water correction

---

## Core Services Architecture

The Container (`src/core/container.py`) manages service lifecycle:

| Service | Class | Purpose |
|:--------|:------|:--------|
| `vector_store` | `RuVectorStore` | Primary vector store (GNN-HNSW) |
| `sparse_index` | `SparseIndex` | FTS5 BM25 keyword search |
| `meta_agent` | `MetaAgent` | LangGraph supervisor (the brain) |
| `document_registry` | `DocumentRegistry` | File-level metadata + boot-sweep |
| `document_intelligence` | `DocumentIntelligenceService` | Upload + VLM processing manager |
| `colosseum` | `SiliconColosseum` | OPA/Rego policy engine |
| `session_store` | `SessionStore` | SQLCipher encrypted sessions |
| `export_engine` | `ExportEngine` | PDF/Markdown export |
| `replay_buffer` | `ReplayBuffer` | Encrypted interaction storage |
| `history_manager` | `HistoryManager` | Conversation history |
| `sync_manager` | `SyncManager` | P2P device sync |

### Module Plugins
Located in `src/modules/`, extending `BaseModule`:
- **CoreModule**: Essential tools and utilities
- **WatchTowerModule**: System monitoring and observability
- **StreamSyncModule**: LiveFolder watcher + RSS feeder
- **AnalyticsModule**: Usage statistics and data analysis
- **RagForgeModule**: CognitiveRAG™ retrieval pipeline
- **TuneLabModule**: Learning monitoring and visualisation
- **LocalBuddyModule**: Local AI assistant
- **SyncModule**: P2P synchronisation tools

---

## Common Development Tasks

### Adding a New Module
1. Create module in `src/modules/<module_name>/`
2. Extend `BaseModule` class
3. Implement `initialize()`, `register_tools()`, and `execute_tool()` methods
4. Register in `src/core/container.py`

### Adding a New API Endpoint
1. Create router in `src/routers/<feature>.py`
2. Register in `src/app_factory.py`

### Adding a New Tool
1. Implement in appropriate module
2. Add to module's `get_tool_definitions()`
3. Register in module's `register_tools()`

### Working with Tests
- Tests use extensive mocking to avoid heavy dependencies
- Fixtures in `conftest.py` provide consistent test environments
- Use `tmp_path` fixture for temporary data directories

---

## Environment Configuration

Key environment variables (`.env`):

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `QWEN_MODEL_PATH` | `/Volumes/Apple/AI Model/qwen2.5-7b-instruct-q4_k_m.gguf` | GGUF model path |
| `DATA_DIR` | `data` | Persistent storage root |
| `SQLCIPHER_KEY_FILE` | `data/.sqlcipher_key` | Session encryption key |
| `SILICON_COLOSSEUM_MIN_FAITHFULNESS` | `0.55` | Min faithfulness score |
| `SILICON_COLOSSEUM_FAITHFULNESS_ACTION` | `block` | Action on low faithfulness |
| `HF_HOME` | `/Volumes/Apple/AI Model/hf_cache` | HuggingFace cache directory |
| `AETHERFORGE_HOST` | `127.0.0.1` | Server host |
| `AETHERFORGE_PORT` | `8765` | Server port |

### Background Services
- **RSS Poller**: Periodically checks RSS feeds for updates
- **Directory Watcher**: Monitors `data/LiveFolder/` for file changes
- **Scheduler**: APScheduler for nightly OPLoRA training (3 AM)
- **Sync Manager**: Handles P2P device synchronisation