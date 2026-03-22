# AetherForge v1.2 — The Sovereign Intelligence OS
## Local, Perpetual, Glass-Box AI for the Edge.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TypeScript 5.5](https://img.shields.io/badge/typescript-5.5-blue.svg)](https://www.typescriptlang.org/)
[![Tauri 2.1](https://img.shields.io/badge/tauri-2.1-orange.svg)](https://tauri.app/)

---

> [!NOTE]
> **Status: Public Beta.** AetherForge is currently in beta with an active focus on becoming fully **OS-agnostic** (optimized for Apple Silicon, with expanding support for Linux and Windows).

---

## 🏛️ Executive Summary

AetherForge is a **Sovereign Intelligence Layer** designed to bridge the gap between high-performance LLMs and the strict requirements of edge-device privacy. Unlike standard RAG frameworks, AetherForge implements a **Closed-Loop Perpetual Learning** architecture. It doesn't just retrieve; it learns from every interaction using **Orthogonal Adaptation** to prevent catastrophic forgetting, while maintaining an air-gapped security profile.

### The "Glass-Box" Philosophy
It solves the "Black Box" problem by exposing internal reasoning traces in real-time. Every decision, from query decomposition to faithfulness scoring, is auditable, traceable, and governed by deterministic policies.

---

## 🏗️ High-Level System Design

```mermaid
graph TD
    %% Frontend Layer
    subgraph "Visual Interface (HUD)"
        UI[Tauri/React/Shadcn]
        TuneLab[TuneLab: Learning Monitor]
        TraceHUD[X-Ray: Causal Trace]
        ThinkingUI[ThinkingBlock: CoT Display]
    end

    %% Orchestration Layer
    subgraph "Cognitive Engine (LangGraph)"
        Supervisor[Meta-Agent Supervisor]
        Router[QueryRouter: Intent Classifier]
        CalcEng[CalcEngine: Deterministic Math]
        RagForge[CognitiveRAG™ Pipeline]
        Logic[Logic Engine: Core Tools]
        CoherenceGate[Coherence Gate: Number Verifier]
    end

    %% Inference & Learning Layer
    subgraph "Learning Hardware (Silicon & Memory)"
        ruvllm[ruvllm: Qwen2.5-7B Instruct]
        OpLoRA[OPLoRA: Orthogonal Learning]
        SONA[SONA: 3-Tier Real-Time Learning]
        RB[Replay Buffer: Parquet/Fernet]
    end

    %% Trust & Governance Layer
    subgraph "Silicon Colosseum (Guardrails)"
        OPA[OPA: Rego Policies]
        FSM[State Enforcement]
        SAMR[SAMR-lite: Faithfulness Scorer]
    end

    %% Storage Layer
    subgraph "Storage"
        RuVector[RuVector GNN-HNSW]
        SQLite[SQLite: Structured Tables]
    end

    %% Communication
    UI <-->|IPC / WebSockets| Supervisor
    Supervisor --> Router
    Router -->|Calc Routes| CalcEng
    Router -->|RAG Routes| RagForge
    CalcEng --> SQLite
    CalcEng --> CoherenceGate
    RagForge --> RuVector
    RagForge --> SAMR
    SAMR --> OPA
    OPA --> RB
    RB --> OpLoRA
    SONA -->|Supplements| OpLoRA
    OpLoRA --> ruvllm
```

---

## 🧠 Core Innovations & Implementation

### 1. **OPLoRA: Perpetual Learning without Forgetting**
Standard fine-tuning (LoRA) on new data often destroys previously learned knowledge (Catastrophic Forgetting). AetherForge implements **Orthogonal Projection LoRA (OPLoRA)** to solve this.

**The Architect's Shortcut:**
Before training on a new task $T_k$, we compute the knowledge subspace of the existing weights using **SVD**. We then build a projector $P$ onto the **orthogonal complement** of that subspace:
$$P = I - U_k U_k^T$$
All gradient updates $\Delta W$ are projected such that:
$$\Delta W_{safe} = P_{left} \Delta W P_{right}$$
This ensures that the new learning happens only in the "null space" of previous memories, preserving 100% of past intelligence.

### 2. **CognitiveRAG™: 7-Stage Reasoning Pipeline**
AetherForge replaces "One-Shot RAG" with a deep thinking pipeline that reuses the BitNet model across multiple reasoning stages:
- **① Query Understanding**: Classifies intent (Factual, Synthesis, Comparative).
- **② Decomposition**: Breaks complex queries into a DAG of sub-questions.
- **③ HyDE**: Generates hypothetical "golden" documents to guide vector search.
- **④ Hybrid Search**: Fuses **RuVector GNN-HNSW** (Dense + BM25 Hybrid) via self-learning reranking.
- **⑤ Evidence Scoring**: Ranks chunks based on multidimensional relevance.
- **⑥ Chain-of-Thought (CoT)**: Synthesizes the final answer within a verifiable reasoning block.
- **⑦ Self-Verification**: Measures faithfulness via **SAMR-lite** before delivery.

### 3. **Silicon Colosseum: Deterministic Alignment**
We replace probabilistic safety filters with a **Deterministic Policy Engine**.
- **OPA (Open Policy Agent)**: Evaluates usage budgets and content safety using **Rego**.
- **Finite State Machine (FSM)**: Enforces lifecycle invariants (e.g., ensuring a reasoning trace exists before a response is allowed).
- **SAMR-lite**: A local semantic faithfulness scorer that computes the cosine similarity between the response and the grounded evidence. Responses below a $0.55$ threshold are automatically **blocked** (not just warned).

### 4. **ruvllm: Native Rust LLM Inference (Qwen2.5-7B)**
Replaces the Ollama HTTP round-trip with direct GGUF inference via Tauri commands. Context window expanded to 16,384 tokens (2× the old 8,192 limit). Falls back gracefully to Python backend inference if model is unavailable.

### 5. **Query Router + CalcEngine: Deterministic Calculation**
For all numeric queries (displacement, TPC, draft corrections), the **QueryRouter** classifies intent *before* any LLM call. Calculation routes bypass the LLM entirely and go straight to the **CalcEngine**, which performs deterministic table interpolation from SQLite. The LLM only writes a natural-language explanation around the verified numbers. The **Coherence Gate** then verifies every number in the explanation traces back to the calc engine output.

### 6. **SONA 3-Tier Real-Time Learning**
Supplements OPLoRA nightly batch with per-request adaptation:
- **Tier 1**: MicroLoRA rank-2 — adapts in <1ms per accepted response
- **Tier 2**: EWC++ consolidation — prevents catastrophic forgetting
- **Tier 3**: ReasoningBank — stores successful trajectories as curriculum

---

## 🛡️ Industrial Scope & Use Cases

AetherForge is designed for high-stakes environments where cloud-dependency and data-leakage are non-negotiable risks.

| Industry | Use Case | Innovation Applied |
| :--- | :--- | :--- |
| **Defense** | Air-gapped field intelligence analysis. | 100% Offline / BitNet Efficiency. |
| **Legal** | Massive case-law synthesis and discovery. | RRF Hybrid Search / Verifiable CoT. |
| **Medical** | Local clinical trial synthesis on patient records. | SQLCipher Encryption / Selective Forgetting. |
| **Hedge Funds** | Market sentiment learning without leaking alpha. | OPLoRA Perpetual Learning. |
| **IoT/Edge** | Autonomous drone/robotics policy adjustment. | Embedded OPA Guardrails / FSM. |

---

## 📂 System Topology

```text
AtherForge/
├── src/
│   ├── core/           # DI Container, Orchestrator, CalcEngine, QueryRouter
│   ├── guardrails/     # Silicon Colosseum (OPA/FSM), Coherence Gate
│   ├── learning/       # OPLoRA, SONA Adapter, Replay Buffer, SVD Math
│   ├── modules/        # CognitiveRAG, RuVector Store, Analytics, SyncManager
│   └── meta_agent.py   # LangGraph Supervisor
├── frontend/           # React/Tauri HUD (X-Ray, TuneLab, ThinkingBlock)
├── src-tauri/src/      # Rust Tauri shell + ruvllm_bridge.rs
├── data/               # Persistent Storage (Encrypted)
└── models/             # Qwen2.5-7B GGUF Weights
```

---

## 🚀 Deployment

1.  **Requirement**: Apple Silicon (M1+) or AVX2 CPU.
2.  **Environment**:
    ```bash
    ./install.sh
    ```
3.  **Run Development Stack**:
    ```bash
    ./run_dev.sh
    ```

---

MIT License | Built for the Era of Sovereign Intelligence.
*Runs on your Mac. Learns from your context. Forgets nothing important.*
