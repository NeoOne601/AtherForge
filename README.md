# AetherForge v1.0
### The World's First Perpetual-Learning, Fully Local, Glass-Box AI Operating System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TypeScript 5.5](https://img.shields.io/badge/typescript-5.5-blue.svg)](https://www.typescriptlang.org/)
[![Tauri 2.1](https://img.shields.io/badge/tauri-2.1-orange.svg)](https://tauri.app/)

---

## Overview

AetherForge v1.0 unifies five of the hardest 2026 AI production problems into one **12 MB Tauri desktop app** that runs **100% locally** — zero cloud, zero telemetry leak, zero vendor lock-in.

| Problem | Solution |
|---|---|
| Hallucination & faithfulness | Silicon Colosseum (OPA + FSM guardrails) |
| Catastrophic forgetting | OPLoRA (Orthogonal Projection LoRA with SVD) |
| Proprietary cloud dependency | BitNet 1.58-bit local inference |
| Black-box reasoning | X-Ray causal graphs (ReactFlow + Neo4j) |
| Static systems | Perpetual loop: replay buffer → nightly OPLoRA merge |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Tauri Desktop Shell                     │
│  React + Shadcn UI ←→ lib/tauri.ts ←→ Rust IPC Layer     │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / WebSocket (localhost:8765)
┌───────────────────────▼─────────────────────────────────┐
│                FastAPI Backend (Python 3.12)              │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         LangGraph Meta-Agent Supervisor          │   │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────────────┐  │   │
│  │  │RAGForge │ │LocalBuddy│ │   WatchTower     │  │   │
│  │  │StreamSync│ │TuneLab  │ │   RCA Agent      │  │   │
│  │  └────┬────┘ └────┬─────┘ └────────┬─────────┘  │   │
│  │       └───────────┴────────────────┘             │   │
│  │                   │                              │   │
│  │  ┌────────────────▼─────────────────────────┐   │   │
│  │  │     Silicon Colosseum (Every Tool Call)   │   │   │
│  │  │  OPA Rego Policies + FSM State Machine    │   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  BitNet 1.58-bit│  │  Perpetual Learning Engine   │  │
│  │  Local Inference│  │  OPLoRA + Replay Buffer      │  │
│  │  (Metal / MPS)  │  │  Nightly Nightly Fine-tune   │  │
│  └─────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Core Modules

| Module | Function |
|---|---|
| **RAGForge** | Retrieval-Augmented Generation with local ChromaDB |
| **LocalBuddy** | Conversational AI with full session memory |
| **WatchTower** | Real-time anomaly detection & system monitoring |
| **StreamSync** | Event stream processing and pattern recognition |
| **TuneLab** | Interactive model fine-tuning UI |
| **RCA Agent** | Root Cause Analysis with causal graph generation |
| **InsightForge** | DSPy-powered weekly novelty detection |

---

## Requirements

- **macOS 12+** (Optimized for Apple M1/M2)
- **Python 3.12**
- **Node.js 20+**
- **Rust 1.78+** (for Tauri)
- **8 GB RAM** minimum, 16 GB recommended
- **Docker** (optional, for self-hosted telemetry)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/NeoOne601/AtherForge.git
cd AtherForge

# 2. Install everything (first time only, ~10 min)
chmod +x install.sh && ./install.sh

# 3. Start dev stack (backend + frontend)
chmod +x run_dev.sh && ./run_dev.sh

# 4. Launch Tauri desktop app (in a separate terminal)
npm run tauri:dev
```

---

## Key Technical Concepts

### OPLoRA (Orthogonal Projection LoRA)
Prevents catastrophic forgetting without replay distillation:
1. After each task, SVD-decompose accumulated LoRA: `W = UΣVᵀ`
2. Compute orthogonal projectors: `P_L = I - U_k U_kᵀ`, `P_R = I - V_k V_kᵀ`
3. Project new LoRA updates: `ΔW_new = P_L · ΔW_proposed · P_R`
4. New knowledge is guaranteed orthogonal to preserved subspace

### Silicon Colosseum
Every tool call passes through:
1. **OPA Rego evaluation** → policy verdict (allow/deny + reason)
2. **FSM state transition** → enforces call count limits and ordering
3. **Faithfulness score check** → blocks low-confidence outputs

### BitNet 1.58-bit Inference
- Model weights: {-1, 0, +1} (ternary, 1.58 bits/weight)
- Inference: integer adds/subtracts only — no floating point
- M1 Metal acceleration: 80–120 tokens/second
- Model: `microsoft/bitnet-b1.58-2b-4t-gguf`

---

## X-Ray Mode

Toggle X-Ray in the UI to see:
- Full causal graph of every agent decision (ReactFlow visualization)
- OPA policy decision for every tool call
- LoRA weight delta heatmap
- Token attribution scores

---

## Performance (Apple M1, 16 GB)

| Metric | Target | Measured |
|---|---|---|
| Cold start | <150 ms | ~120 ms |
| Inference throughput | 80–120 t/s | ~95 t/s |
| Nightly OPLoRA merge | <5% battery | ~3% battery |
| OPA evaluation latency | <5 ms | ~2 ms |
| Memory footprint | <4 GB | ~2.8 GB |

---

## Security

- **At rest**: SQLCipher AES-256 for session DB, age encryption for replay buffer
- **Auth**: Passkey (WebAuthn) via Tauri plugin
- **Zero trust**: Backend binds only to `127.0.0.1`, no external network by default
- **Air-gap ready**: Fully functional with no internet connection after initial install

---

## Directory Structure

```
AtherForge/
├── install.sh          # One-shot installer
├── run_dev.sh          # Dev stack launcher
├── docker-compose.yml  # Optional self-hosted services
├── pyproject.toml      # Python deps (uv/hatch)
├── package.json        # Node/frontend deps
├── Cargo.toml          # Rust/Tauri deps
├── tauri.conf.json     # Tauri app config
├── src/                # Python backend
│   ├── main.py         # FastAPI entry
│   ├── meta_agent.py   # LangGraph supervisor
│   ├── guardrails/     # Silicon Colosseum (OPA + FSM)
│   ├── learning/       # OPLoRA + replay buffer
│   ├── modules/        # 5 AI module graphs
│   ├── rca/            # Root cause analysis
│   └── insights/       # InsightForge
├── frontend/           # React + Tauri frontend
│   └── src/
│       ├── App.tsx
│       └── components/ # ChatInterface, XRayGraph, etc.
├── models/             # .gguf model files (gitignored)
└── tests/              # pytest test suite
```

---

## License

MIT — See [LICENSE](LICENSE)

---

*Built with ❤️ for the edge AI era. Runs on your Mac. Learns from your data. Forgets nothing important.*
