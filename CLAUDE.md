# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AetherForge is a Sovereign Intelligence OS that implements a Closed-Loop Perpetual Learning architecture with air-gapped security. It bridges high-performance LLMs with edge-device privacy requirements using local inference and continual learning without catastrophic forgetting.

Key innovations:
- OPLoRA + SONA: Perpetual learning with 3-tier real-time adaptation and nightly consolidation
- CognitiveRAG™: 7-stage reasoning pipeline with self-verification
- Silicon Colosseum: Deterministic policy engine using OPA/Rego
- Query Router + CalcEngine: Deterministic calculation for numeric queries (LLMs explain; they never calculate)
- ruvllm: Native Rust GGUF runtime (Qwen2.5-7B-Instruct) via Tauri commands
- Coherence Gate: Post-generation number verification against calc traces

## Repository Structure

```
AtherForge/
├── src/              # Python backend (FastAPI)
│   ├── core/         # DI Container, Orchestrator, Grammar
│   ├── guardrails/   # Silicon Colosseum (OPA policies)
│   ├── learning/     # OPLoRA, Replay Buffer, History Manager
│   ├── modules/      # Plugin modules (RAGForge, StreamSync, etc.)
│   ├── routers/      # FastAPI route handlers
│   └── main.py       # Application entry point
├── frontend/         # React/Tauri desktop UI
├── data/             # Persistent storage (encrypted)
├── models/           # Ternary BitNet weights (GGUF)
├── tests/            # Unit and integration tests
└── Test_related/     # Additional test resources
```

## Development Commands

### Installation
```bash
# One-time setup (installs all deps, creates venv, downloads model)
chmod +x install.sh && ./install.sh
```

### Running Development Server
```bash
# Start full dev stack (backend + frontend + Tauri)
./run_dev.sh

# Start only backend
./run_dev.sh --backend-only

# Start only frontend
./run_dev.sh --frontend-only

# Start without Docker services
./run_dev.sh --no-docker
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chat_contract.py

# Run tests with coverage
pytest --cov=src

# Run specific test function
pytest tests/test_chat_contract.py::test_split_reasoning_trace_complete_block

# Run tests in verbose mode
pytest -v

# Run tests with output capturing disabled
pytest -s

# Run tests matching a pattern
pytest -k "test_reasoning"
```

### Linting and Formatting
```bash
# Run linter
ruff check src

# Auto-fix lint issues
ruff check --fix src

# Type checking
mypy src

# Format code
black src
```

### Building
```bash
# Build Python package
uv build

# Build Tauri desktop app
npm run tauri:build

# Build frontend assets
npm run build
```

### Frontend Development
```bash
# Start frontend dev server
npm run dev

# Run frontend linting
npm run lint

# Run TypeScript type checking
npm run type-check

# Format frontend code
npm run format
```

### Docker Services
```bash
# Start optional self-hosted services (Langfuse, Neo4j, OPA)
docker compose up -d

# Stop Docker services
docker compose down

# View Docker service logs
docker compose logs -f
```

## Architecture Overview

### Backend (Python/FastAPI)
- **Dependency Injection**: Central `Container` manages service lifecycle
- **Modular Design**: Plugin modules extend functionality (RAGForge, StreamSync, etc.)
- **LangGraph Integration**: MetaAgent orchestrates reasoning workflows
- **Local Inference**: ruvllm native Rust GGUF runtime with Qwen2.5-7B-Instruct (Ollama fallback)
- **Storage**: RuVector GNN-HNSW (hybrid semantic + BM25 search), SQLite FTS5 (sparse), encrypted SQLite (sessions)
- **Query Router**: Deterministic routing — calculation queries bypass LLM, go directly to CalcEngine + SQLite
- **Coherence Gate**: Every number in LLM explanations verified against calc engine traces

### Frontend (React/Tauri)
- **Module-Based UI**: Separate panels for different AI capabilities
- **X-Ray Debugging**: Real-time causal tracing of reasoning processes
- **Session Management**: Persistent chat histories with export
- **Policy Editor**: Visual interface for OPA Rego policies

### Key Components
1. **Container System** (`src/core/container.py`): Manages service lifecycle and dependencies
2. **MetaAgent** (`src/meta_agent.py`): LangGraph-based supervisor for complex reasoning
3. **RAGForge** (`src/modules/ragforge/`): 7-stage cognitive retrieval pipeline
4. **OPLoRA** (`src/learning/`): Continual learning system with orthogonal projection
5. **Silicon Colosseum** (`src/guardrails/`): Deterministic alignment using OPA policies
6. **StreamSync** (`src/modules/streamsync/`): Live folder monitoring and RSS integration

### Core Services Architecture
The application uses a dependency injection container (`src/core/container.py`) that manages the lifecycle of core services:
- **ReplayBuffer**: Stores interaction history for continual learning
- **HistoryManager**: Manages conversation history
- **RuVectorStore**: RuVector GNN-HNSW hybrid vector + BM25 search (replaces ChromaDB)
- **SparseIndex**: SQLite FTS5 for sparse retrieval (handled by RuVector hybrid when available)
- **SessionStore**: Encrypted SQLite storage for chat sessions
- **SiliconColosseum**: OPA-based policy enforcement
- **MetaAgent**: LangGraph supervisor for reasoning workflows
- **DocumentIntelligence**: VLM-based document processing
- **SyncManager**: Peer-to-peer synchronization
- **CalcEngine** (`src/core/calc_engine.py`): Deterministic table interpolator — no LLM arithmetic
- **QueryRouter** (`src/core/query_router.py`): Intent classifier that fires before any LLM call
- **TableExtractor** (`src/modules/ragforge/table_extractor.py`): Tables → SQLite at ingestion time
- **CoherenceGate** (`src/guardrails/coherence_gate.py`): Number trace verifier for calc responses
- **SONAAdapter** (`src/learning/sona_adapter.py`): Per-request SONA 3-tier real-time learning
- **ruvllm_bridge** (`src-tauri/src/ruvllm_bridge.rs`): Rust Tauri command for native LLM inference
- **ThinkingBlock** (`frontend/src/components/ThinkingBlock.tsx`): Collapsible CoT display

### Module Plugins
The system is extensible through module plugins located in `src/modules/`:
- **CoreModule**: Essential tools and utilities
- **WatchTowerModule**: System monitoring and observability
- **StreamSyncModule**: Live folder and RSS monitoring
- **AnalyticsModule**: Usage statistics and insights
- **RagForgeModule**: Cognitive retrieval pipeline
- **TuneLabModule**: Learning monitoring and visualization
- **LocalBuddyModule**: Local AI assistant capabilities
- **SyncModule**: Peer-to-peer synchronization tools

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
- Run individual tests with `pytest tests/test_file.py::test_function_name`

### Environment Configuration
The application uses environment variables defined in `.env` for configuration:
- **Model Settings**: QWEN_MODEL_PATH, BITNET_MODEL_PATH (legacy fallback), BITNET_N_CTX, BITNET_N_GPU_LAYERS
- **Storage Paths**: DATA_DIR, CHROMA_PATH (legacy), REPLAY_BUFFER_PATH
- **Security**: SQLCIPHER_KEY_FILE for encrypted storage
- **Guardrails**: SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.55, SILICON_COLOSSEUM_FAITHFULNESS_ACTION=block
- **Telemetry**: LANGFUSE_* settings for optional self-hosted telemetry
- **Network**: AETHERFORGE_HOST, AETHERFORGE_PORT for server configuration

### Background Services
Several background services run continuously:
- **RSS Poller**: Periodically checks RSS feeds for updates
- **Directory Watcher**: Monitors LiveFolder for file changes
- **Scheduler**: Runs periodic jobs like nightly OPLoRA training
- **Sync Manager**: Handles peer-to-peer synchronization