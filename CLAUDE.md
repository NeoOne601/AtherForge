# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AetherForge is a Sovereign Intelligence OS that implements a Closed-Loop Perpetual Learning architecture with air-gapped security. It bridges high-performance LLMs with edge-device privacy requirements using local inference and continual learning without catastrophic forgetting.

Key innovations:
- OPLoRA: Orthogonal Projection LoRA for perpetual learning without forgetting
- CognitiveRAG™: 7-stage reasoning pipeline with self-verification
- Silicon Colosseum: Deterministic policy engine using OPA/Rego
- BitNet 1.58-bit ternary inference optimized for Apple Silicon

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
```

### Linting and Formatting
```bash
# Run linter
ruff check src

# Auto-fix lint issues
ruff check --fix src

# Type checking
mypy src
```

### Building
```bash
# Build Python package
uv build

# Build Tauri desktop app
npm run tauri:build
```

## Architecture Overview

### Backend (Python/FastAPI)
- **Dependency Injection**: Central `Container` manages service lifecycle
- **Modular Design**: Plugin modules extend functionality (RAGForge, StreamSync, etc.)
- **LangGraph Integration**: MetaAgent orchestrates reasoning workflows
- **Local Inference**: llama-cpp-python with BitNet GGUF models
- **Storage**: ChromaDB (dense search), SQLite FTS5 (sparse search), encrypted SQLite (sessions)

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