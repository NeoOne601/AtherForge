# AetherForge v1.0
## User Manual — Self-Help Guide

> *Your AI. Your data. Your machine. Always.*

---

<div align="center">

**A local AI Operating System that learns, reasons transparently, and never forgets — all without sending a single byte to the cloud.**

</div>

---

## How to Use This Manual

This manual is organised by **who you are** and **what you want to do**. Find your role below and jump directly to your section.

| Your Role | Go To |
|---|---|
| First-time user — just want to chat with AI | [Part 1 — The Everyday User](#part-1--the-everyday-user) |
| Professional or knowledge worker | [Part 2 — The Knowledge Worker](#part-2--the-knowledge-worker) |
| IT administrator or deployer | [Part 3 — The IT Administrator](#part-3--the-it-administrator) |
| Data scientist or ML practitioner | [Part 4 — The Data Scientist--ML Engineer](#part-4--the-data-scientist--ml-engineer) |
| Security officer or compliance lead | [Part 5 — The Security--Compliance Officer](#part-5--the-security--compliance-officer) |

---

## What Is AetherForge?

AetherForge is a **local AI Operating System** — a desktop application that runs a fully capable AI assistant entirely on your own computer, with no internet connection required after installation.

Think of it as **your own private ChatGPT**, but:

- 🔒 **100% private** — your conversations never leave your machine
- 🧠 **Learns from you** — gets smarter about your work every night
- 🔍 **Glass-box** — you can see exactly how it reasoned, step by step
- 🛡️ **Safe by design** — a built-in rule engine blocks harmful or unreliable outputs
- 📚 **Remembers your documents** — search your own files, PDFs, and knowledge base

AetherForge is built around **five specialist modules**, each of which you can activate from the left sidebar:

| Module | Icon | What It Does |
|---|---|---|
| **LocalBuddy** | 💬 | General conversational AI with memory |
| **RAGForge** | 🔍 | Search and Q&A over your own documents |
| **WatchTower** | 👁️ | Detect anomalies in data and system metrics |
| **StreamSync** | ⚡ | Analyse streams of events for patterns |
| **TuneLab** | 🎛️ | Control and monitor the AI's learning process |

---

## Understanding the Interface

```
┌─────────────────────────────────────────────────────────────────────┐
│  Æ AetherForge v1.0         CPU 12%  🔋85%  ● Online  [X-Ray OFF]  │  ← Header bar
├──────────────┬──────────────────────────────┬────────────────────── │
│  MODULES     │                              │  X-Ray Causal Graph   │
│  ──────────  │       Chat / Insights /      │  (appears when        │
│  🔍 RAGForge │        Policies Panel        │   X-Ray is ON)        │
│  💬 LocalBud │                              │                       │
│  👁️ WatchTow │                              │  Shows exactly how    │
│  ⚡ StreamSy │                              │  the AI reasoned      │
│  🎛️ TuneLab  │                              │  through your query   │
│  ──────────  │                              │                       │
│  💬 Chat     │                              │                       │
│  ✨ Insights │                              │                       │
│  🛡️ Policies │                              │                       │
└──────────────┴──────────────────────────────┴───────────────────────┘
```

**Three panels, always available:**
- **Chat** — Talk to the AI using any module
- **Insights** — Read the AI's weekly novelty report about what it learned
- **Policies** — View and edit the safety rules that govern the AI's behaviour

---

# Part 1 — The Everyday User

> **You are:** Someone who wants to use AI to get things done — write, research, ask questions, and get answers. You are not a developer or data scientist.

---

## Getting Started in 5 Minutes

### Step 1 — Launch AetherForge

If AetherForge is already installed on your machine, look for the **Æ** icon in your Applications folder or dock. Double-click to open.

You will see a chat window appear after a few seconds (the AI model loads in the background — the first load takes about 5–8 seconds).

### Step 2 — Check You Are Online

Look at the top-right corner of the window. You should see a green dot labelled **Online**. If it shows **Offline**, contact your IT administrator.

### Step 3 — Start Talking

1. Click on the text box at the bottom of the chat area
2. Type your question or request
3. Press **Enter** to send (or **Shift + Enter** to add a new line)
4. The AI starts typing its response immediately, word by word

That's it. You're using AetherForge.

---

## Everyday Use — What You Can Do

### Ask Any Question
Type naturally, just as you would speak:

> "Summarise the key points of servant leadership."

> "What is the difference between machine learning and deep learning?"

> "Help me write a professional apology email to a client."

### Get Writing Help
AetherForge is excellent at drafting, editing, and improving text:

> "Rewrite this paragraph to sound more formal: [paste your text]"

> "Give me 5 subject lines for a marketing email about our new product launch."

> "I need to write a performance review for a team member who consistently meets targets but struggles with deadlines. Help me phrase it constructively."

### Think Through Problems
Use AetherForge as a thinking partner:

> "Walk me through the pros and cons of switching from Oracle to PostgreSQL."

> "My team disagrees on whether to build or buy the analytics dashboard. What questions should we be asking?"

### Research and Summarise
If your IT admin has added documents to your knowledge base, AetherForge can search them:

> "Find everything we have about the Johnson account."

> "What does our employee handbook say about remote work?"
*(Switch to the **RAGForge** module for document questions — see below)*

---

## Choosing the Right Module

Select a module by clicking it in the **left sidebar**. The module name appears in the chat input placeholder.

### 💬 LocalBuddy — Your Everyday AI
**Use this for:** General questions, writing, brainstorming, learning, analysis.

LocalBuddy remembers your **entire conversation** within a session. You can refer back to earlier points:

> "Based on what we discussed earlier about the budget, draft a summary for the board."

> "Can you expand on the third point you made?"

**Memory note:** Memory resets when you close and reopen the app. Each session starts fresh. (Your IT admin can enable persistent memory — ask them.)

---

### 🔍 RAGForge — Search Your Documents
**Use this for:** Finding information in your organisation's documents, PDFs, policy files, and knowledge bases.

**How it works:** RAGForge searches documents that your IT admin has ingested into the local knowledge base. It finds the most relevant passages and uses them to answer your question accurately.

**Good questions for RAGForge:**
> "What is our refund policy for enterprise customers?"

> "Find the sections about data retention in the GDPR compliance document."

> "What did the Q3 2025 board report say about headcount?"

**Tips:**
- Be specific — name the document or topic if you know it
- RAGForge cites which documents it pulled from (shown in the metadata row below the answer)
- If no relevant document is found, it will tell you

---

### 👁️ WatchTower — Spot Anomalies
**Use this for:** Understanding unusual patterns in data or metrics.

**Example questions:**
> "Here are our daily sales figures for the past 30 days: [paste numbers]. Are there any unusual spikes or dips?"

> "Analyse this server latency data and tell me if anything looks abnormal."

WatchTower uses statistical methods to detect values that fall significantly outside normal patterns (more than 3 standard deviations from the mean), shown as **anomaly alerts**.

---

### ⚡ StreamSync — Pattern Finder
**Use this for:** Analysing sequences of events to find recurring patterns or correlations.

This module is most useful if your organisation feeds it event data. Ask your IT admin whether StreamSync has been configured for your data streams.

---

### 🎛️ TuneLab — AI Learning Dashboard
**Use this for:** Seeing how the AI is learning from your team's interactions.

You can:
- See how many conversations the AI has learned from
- Check when the last training run happened
- Read the AI's **faithfulness score** (a measure of how reliable its outputs have been)
- Click **"▶ Train Now"** to manually trigger a learning update

You do not need to change any settings here for TuneLab to do its job automatically.

---

## Understanding the AI's Response Badges

Each AI response shows small coloured badges below the text:

| Badge | Colour | Meaning |
|---|---|---|
| Module name (e.g. `localbuddy`) | Purple | Which module answered |
| Response time (e.g. `342ms`) | Grey | How fast the AI responded |
| `fidelity 94%` | Green | AI's answer is highly reliable |
| `fidelity 71%` | Red | AI was less certain — verify this answer |
| `🛡️ policy applied` | Red | A safety rule modified or blocked part of this response |

**What to do if you see a policy badge:** The AI's safety layer (Silicon Colosseum) detected something it needed to handle carefully. The response is either modified or replaced with a safe alternative. This is expected behaviour — it means the guardrails are working.

---

## The X-Ray Button — See How the AI Thinks

Click **"X-Ray OFF"** in the top-right to toggle **X-Ray mode ON**.

When X-Ray is enabled, a panel slides in from the right showing a **live diagram** of every reasoning step the AI took to answer your question:

- Each box is a reasoning **node** (e.g. "safety check", "retrieve documents", "generate response")
- Arrows show the **direction of reasoning flow**
- Colours indicate node type:
  - 🟣 **Purple** — standard reasoning step
  - 🔴 **Red** — safety/policy evaluation node
  - 🟢 **Green** — final output node
- Click any node to see the full data it processed

**When to use X-Ray:**
- When you want to verify the AI's reasoning
- When you received an unexpected answer and want to understand why
- When a safety policy was applied and you want to see what triggered it

---

## Frequently Asked Questions — Everyday Users

**Q: Is my data sent to the internet?**
No. AetherForge runs 100% on your computer. Your conversations, your documents, your queries — none of it leaves your machine. There are no cloud servers involved.

**Q: Why did the AI refuse to answer my question?**
The built-in safety system (Silicon Colosseum) blocked the request. Look for the `🛡️ policy applied` badge. Common reasons: the request matched a prohibited pattern, the AI's confidence was too low, or the query exceeded the safety parameters. If you believe this is incorrect, speak to your IT admin about adjusting policies.

**Q: The AI gave a wrong answer. What should I do?**
Check the **fidelity badge**. If it shows a low percentage, the AI flagged its own uncertainty. For critical decisions, always verify AI-generated information with authoritative sources. You can also rephrase your question and try again — a different phrasing often yields a better answer.

**Q: The AI seems to have forgotten what I said earlier.**
Memory is per-session. If you closed the app or started a new session, previous context is not available. Within the same session, the AI remembers the last 50 turns of conversation.

**Q: How do I start a fresh conversation?**
Close and reopen the app, or ask the AI: *"Please start fresh and ignore our previous conversation in this session."*

**Q: Can I use AetherForge offline?**
Yes, completely. Once installed, AetherForge requires no internet connection whatsoever.

**Q: My responses feel slow. Is this normal?**
On an Apple M1 with 16 GB RAM, you should get 80–110 words per second. If it feels slow, close other heavy applications. If the issue persists, contact your IT admin.

---

# Part 2 — The Knowledge Worker

> **You are:** A professional — analyst, consultant, researcher, writer, product manager, or specialist — who wants to use AetherForge as a serious productivity tool. You want to integrate it into how you work, not just ask it occasional questions.

---

## Advanced Conversation Techniques

### System Priming
Start your chat session by giving the AI a role or context:

> "For this session, you are a seasoned financial analyst reviewing a startup's pitch deck. I will paste sections and you will critique them as that analyst."

> "Act as a senior copywriter who has worked for luxury consumer brands. All feedback should be framed through that lens."

The AI will maintain this persona for the entire session.

### Multi-Step Projects
AetherForge handles complex, multi-turn workflows:

> **Turn 1:** "Here is our product roadmap for Q1 2026. Analyse the prioritisation logic."
> **Turn 2:** "Based on what you just analysed, what would you add or remove if we had 30% fewer engineers?"
> **Turn 3:** "Now write an executive summary of the revised roadmap."

### Structured Output Requests
Ask for specific output formats:

> "Return your analysis as a markdown table with three columns: Risk, Likelihood, Mitigation."

> "Give me a SWOT analysis in bullet-point format — no prose."

> "Provide your answer as a JSON object with keys: summary, action_items, and open_questions."

---

## RAGForge — Power User Mode

RAGForge is where AetherForge delivers maximum value for knowledge workers.

### Understanding How RAGForge Searches
RAGForge uses **semantic similarity** (not keyword matching) to find relevant passages. This means:
- You don't need to use the exact words from a document
- "employee compensation" will find passages that mention "salary", "remuneration", and "pay"
- Context matters — your full question shapes what it finds, not just one keyword

### Effective RAGForge Prompting

**Use context-rich questions:**
> ❌ "Refund policy"
> ✅ "What are the conditions under which enterprise customers can request a full refund, and what is the timeframe?"

**Combine retrieval with reasoning:**
> "Find everything about our SLA commitments and then assess whether we are currently meeting the 99.9% uptime target based on what you know."

**Cross-document questions:**
> "Compare what the 2024 and 2025 market research reports say about customer churn rates."

**Follow-up on retrieved content:**
> "Based on the contract terms you just found, draft a client email explaining our position."

---

## Using X-Ray for Professional Verification

In professional contexts, **X-Ray mode is your audit trail**:

1. Enable X-Ray before asking a sensitive question
2. After the response, click each node in the causal graph to see:
   - Which documents were retrieved (RAGForge nodes)
   - What safety checks were applied and passed
   - What confidence the AI had at each stage
3. If a policy node shows in **red**, a safety rule was evaluated — click it to see what rule and why

This gives you documented evidence of how an answer was generated — useful for compliance, legal, and audit contexts.

---

## Reading the Insights Panel

Click **Insights** in the left sidebar to see **InsightForge**'s weekly novelty report.

This report shows:
- **Total Interactions** — how many queries the AI has learned from
- **Average Faithfulness** — overall reliability score across all answers
- **Novel patterns** — new topics or question types the AI detected this week, ranked by novelty score (0–100%)
- **Topic clusters** — key terms grouped together from your team's usage

**How to use this professionally:**
- A high-novelty cluster means your team is exploring new territory the AI hasn't seen much — consider feeding it more documents on that topic
- A falling average faithfulness score may mean the AI is being asked questions outside its current knowledge base — RAGForge ingestion can help
- Novelty clusters often surface emerging team interests before anyone has explicitly named them as a priority

---

## Keyboard Shortcuts and Efficiency Tips

| Action | How |
|---|---|
| Send message | Enter |
| New line in message | Shift + Enter |
| Enable/disable streaming | Toggle in bottom-right of chat input area |
| Enable X-Ray | Click "X-Ray OFF" in header |
| Switch module | Click module name in left sidebar |
| Switch to Insights | Click ✨ Insights in bottom of sidebar |
| Switch to Policy Editor | Click 🛡️ Policies in bottom of sidebar |

**Pro tips:**
- **Disable token streaming** if you want the full response to appear at once (useful when copying output)
- **X-Ray auto-enables** when an answer comes back with a causal graph — you don't need to pre-enable it
- Long conversations produce better answers — the AI uses your full session history as context

---

## Frequently Asked Questions — Knowledge Workers

**Q: How do I get my own documents into RAGForge?**
Document ingestion is an IT admin task (see Part 3). Raise a request with your IT team to add specific files, folders, or SharePoint/Confluence exports to the local knowledge base.

**Q: Can I use AetherForge to analyse spreadsheet or numerical data?**
Paste data directly into the chat as text or CSV format. AetherForge can interpret, summarise, and reason over numerical data. For anomaly detection, use the **WatchTower** module and paste your data series.

**Q: How do I know if an answer is from my documents or from the AI's own knowledge?**
In the response metadata row (below each AI message), look for the module badge. **RAGForge** answers are drawn from your documents. **LocalBuddy** answers draw on the model's trained knowledge. The X-Ray panel shows the RAGForge retrieval node specifically.

**Q: Can I export my conversation?**
Not in v1.0 directly from the UI. Copy and paste the conversation text, or use the API (`curl http://localhost:8765/api/v1/chat`) to integrate with other tools.

**Q: The AI keeps misunderstanding a specialised term from our industry. What can I do?**
Define it at the start of your session: *"Throughout this conversation, 'TCV' means Total Contract Value, not Total Claim Value."* The AI will use your definition. Long-term, ask your IT admin to ingest domain-specific documents via RAGForge.

---

# Part 3 — The IT Administrator

> **You are:** Responsible for deploying AetherForge, managing access, ingesting documents, configuring the environment, and supporting end users.

---

## Installation Overview

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Hardware | Apple M1 8 GB RAM | Apple M1/M2/M3 16+ GB |
| OS | macOS 12.0+ | macOS 14+ |
| Disk space | 5 GB free | 10 GB free |
| Network | Not required after install | Not required |

AetherForge does **not** support Windows or Linux in v1.0. macOS on Apple Silicon (M1/M2/M3) is the only supported deployment platform.

### One-Shot Install
```bash
git clone https://github.com/NeoOne601/AtherForge.git
cd AtherForge
chmod +x install.sh && ./install.sh
```

The installer handles:
- Homebrew, Python 3.12, Node.js 20, Rust 1.78
- `llama-cpp-python` with Metal GPU acceleration enabled
- BitNet 1.58-bit model download (~1.2 GB from HuggingFace)
- Environment configuration (`.env` file)
- Data directory structure

**Estimated install time:** 10–20 minutes (varies by internet speed for model download)

### Starting the Application
```bash
# Start backend + frontend dev servers
./run_dev.sh

# In a second terminal — launch the Tauri desktop window
npm run tauri:dev
```

For production deployment on a managed device, use the compiled `.dmg` bundle (see Packaging section).

---

## Environment Configuration

All configuration lives in the `.env` file at the project root. Key settings:

```bash
# ── Core ──────────────────────────────────────────────────────────
AETHERFORGE_PORT=8765            # Backend API port
AETHERFORGE_ENV=production       # production | development | test

# ── AI Model ──────────────────────────────────────────────────────
BITNET_MODEL_PATH=./models/bitnet-b1.58-2b-4t.gguf
BITNET_N_GPU_LAYERS=-1           # -1 = all layers on Metal GPU
BITNET_CONTEXT_WINDOW=32768      # Max tokens per conversation
BITNET_N_THREADS=8               # CPU threads for non-GPU layers

# ── Safety ────────────────────────────────────────────────────────
OPA_MODE=embedded                # embedded (Python) | server (Docker OPA)
SILICON_COLOSSEUM_MAX_TOOL_CALLS=8
SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.92

# ── Learning ──────────────────────────────────────────────────────
OPLOРА_NIGHTLY_HOUR=3            # 3 AM nightly training window
OPLOРА_MIN_BATTERY_PCT=30        # Skip training if battery < 30%
OPLOРА_RANK_K=64                 # SVD rank (higher = more memory preserved)
OPLOРА_LORA_R=16                 # LoRA rank (higher = more capacity)

# ── Optional Services ─────────────────────────────────────────────
LANGFUSE_ENABLED=false           # Local Langfuse telemetry dashboard
NEO4J_ENABLED=false              # Persistent X-Ray graph storage
```

---

## Ingesting Documents into RAGForge

To add documents to the local knowledge base accessible to RAGForge:

### Via API (recommended for automation)
```bash
# Single document ingestion
curl -X POST http://localhost:8765/api/v1/ragforge/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Full text of document here..."],
    "ids": ["doc-001"],
    "metadatas": [{"source": "HR Handbook v2.pdf", "date": "2026-01-15"}]
  }'
```

### Bulk ingestion script
Create a Python script using the ingestion helper:

```python
import requests, pathlib

API = "http://localhost:8765/api/v1/ragforge/ingest"

def ingest_file(filepath: str):
    text = pathlib.Path(filepath).read_text(encoding="utf-8", errors="ignore")
    # Split into chunks of ~1000 words for better retrieval granularity
    words = text.split()
    chunks = [" ".join(words[i:i+1000]) for i in range(0, len(words), 800)]

    for idx, chunk in enumerate(chunks):
        requests.post(API, json={
            "texts": [chunk],
            "ids": [f"{filepath}-{idx}"],
            "metadatas": [{"source": filepath, "chunk": idx}]
        })
    print(f"Ingested {len(chunks)} chunks from {filepath}")

# Ingest all .txt and .md files in a folder
for f in pathlib.Path("./knowledge_base").rglob("*.txt"):
    ingest_file(str(f))
```

**Supported input:** Any plaintext-extractable format. For PDFs, use `pdfminer.six` or `pypdf` to extract text first, then ingest the extracted text.

**Storage:** Documents are stored in ChromaDB at `data/chroma/` (local, no cloud). The ChromaDB directory is gitignored and never leaves the machine.

---

## Configuring Self-Hosted Telemetry (Optional)

For usage monitoring and observability without cloud tools:

```bash
# Start the optional telemetry stack
docker compose up -d

# Services started:
# Langfuse UI:  http://localhost:3000  (usage analytics)
# OPA Server:   http://localhost:8181  (external policy engine)
# Neo4j:        http://localhost:7474  (X-Ray graph persistence)
```

Then enable in `.env`:
```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
OPA_MODE=server                  # Use Docker OPA instead of embedded Python
NEO4J_ENABLED=true
```

Set up a Langfuse account at `http://localhost:3000` and create a project. Copy the public/secret keys into `.env`.

---

## Building the Production .dmg for Distribution

```bash
# Build the optimised desktop application
npm run build        # Compile React frontend
cargo tauri build    # Build Tauri native app + installer

# Output locations:
# .app → src-tauri/target/release/bundle/macos/AetherForge.app
# .dmg → src-tauri/target/release/bundle/dmg/AetherForge_1.0.0_aarch64.dmg
```

Distribute the `.dmg` to end users. They run it like any macOS installer — drag to Applications. The model file must be placed at `~/Library/Application Support/AetherForge/models/` (or configure `BITNET_MODEL_PATH` in the bundled `.env`).

---

## User Access and Multi-User Setup

AetherForge v1.0 is a **single-user, single-machine** application. Each user runs their own instance.

For team deployments:
- Deploy via MDM (Jamf, Mosyle, etc.) with a pre-configured `.env` template
- Each machine has its own independent replay buffer and learning state
- Organizations can share RAGForge document ingestion runs via a seed script, so all machines start with the same knowledge base
- Policy files (`default_policies.rego`) can be distributed via MDM to enforce organisation-wide rules

---

## API Reference for Administrators

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Backend health check |
| `/api/v1/status` | GET | System metrics (CPU, RAM, battery) |
| `/api/v1/chat` | POST | Send a chat message and get a response |
| `/api/v1/modules` | GET | List available modules |
| `/api/v1/policies` | GET | Fetch current Rego policy text |
| `/api/v1/policies` | POST | Update and hot-reload policy |
| `/api/v1/learning/trigger` | POST | Manually trigger OPLoRA training |
| `/api/v1/replay/stats` | GET | Replay buffer statistics |
| `/api/v1/ragforge/ingest` | POST | Ingest documents into knowledge base |
| `/ws/chat/{session_id}` | WebSocket | Streaming chat connection |

Full OpenAPI documentation available at `http://localhost:8765/docs` when the backend is running.

---

## Monitoring and Logs

```bash
# All application logs
tail -f data/logs/aetherforge.log

# Silicon Colosseum policy decisions (every request, allow or deny)
tail -f data/logs/colosseum.jsonl

# OPLoRA training runs
tail -f data/logs/training.log
```

Each `colosseum.jsonl` entry looks like:
```json
{
  "timestamp": "2026-03-02T13:00:00Z",
  "session_id": "abc123",
  "module": "localbuddy",
  "allowed": true,
  "reason": "All policy checks passed",
  "latency_ms": 2.1,
  "fsm_state": "PROCESSING"
}
```

---

## Troubleshooting — IT Admin Reference

| Issue | Diagnostic | Fix |
|---|---|---|
| Backend offline (red dot) | `curl http://localhost:8765/health` | Run `./run_dev.sh` |
| Model file not found | Check `BITNET_MODEL_PATH` in `.env` | Re-run `./install.sh` or download manually |
| Metal GPU not detected | `python -c "from llama_cpp import Llama; print('ok')"` | Reinstall with `CMAKE_ARGS="-DLLAMA_METAL=on"` |
| OPA mode server but Docker not running | `docker ps` | `docker compose up -d` or set `OPA_MODE=embedded` |
| ChromaDB corruption | Check disk space | Delete `data/chroma/` and re-ingest |
| High CPU at 3 AM | Expected nightly training | Increase `OPLOРА_MIN_BATTERY_PCT` or change training hour |
| Port 8765 conflict | `lsof -i :8765` | Change `AETHERFORGE_PORT` in `.env` |
| Tauri build failure | Xcode CLI tools | `xcode-select --install` |

---

# Part 4 — The Data Scientist / ML Engineer

> **You are:** Working with the AI system at a technical level — you want to understand the learning loop, configure fine-tuning, validate the OPLoRA math, and extend the system.

---

## The OPLoRA Continual Learning Loop

AetherForge learns perpetually without forgetting previous knowledge. Here is the full pipeline:

### 1. Interaction Recording
Every user interaction is written to an **encrypted Parquet replay buffer** at `data/replay_buffer.parquet`:

- Encrypted with Fernet AES-128-CBC (key at `data/.db_key`, mode 0600)
- Schema: `id | session_id | module | prompt | response | faithfulness_score | tool_calls_json | timestamp_utc | is_used_for_training | novelty_score`
- Async append with configurable flush threshold (default: every 10 records)

### 2. Nightly Training Trigger
APScheduler fires at `OPLOРА_NIGHTLY_HOUR` (default 3 AM) **only if**:
- Battery > `OPLOРА_MIN_BATTERY_PCT` (default 30%)
- CPU load < 80%

```python
# src/main.py — the scheduler gate
@scheduler.scheduled_job("cron", hour=settings.oploра_nightly_hour)
async def nightly_oploRA_job():
    battery = psutil.sensors_battery()
    if battery and battery.percent < settings.oploра_min_battery_pct:
        logger.info("Skipping training: battery %.0f%%", battery.percent)
        return
    ...
```

### 3. OPLoRA Projection Math

The core algorithm (in `src/learning/oploRA_manager.py::compute_projectors()`):

```python
# Economy SVD of the accumulated task weight update
U, sigma, Vt = np.linalg.svd(delta_W, full_matrices=False)

# Top-k singular vectors (knowledge subspace)
U_k = U[:, :k]          # (d_out, k)
V_k = Vt[:k, :].T       # (d_in,  k)

# Orthogonal projectors onto the complement
P_L = np.eye(d_out) - (U_k @ U_k.T)   # (d_out, d_out)
P_R = np.eye(d_in)  - (V_k @ V_k.T)   # (d_in,  d_in)
```

**Mathematical guarantee:**
```
U_kᵀ @ (P_L @ ΔW_new @ P_R) ≈ 0
```
Any new update projected through `P_L` and `P_R` is provably orthogonal to the past knowledge subspace. The tests in `tests/test_oploRA_manager.py::test_projection_nullifies_past_subspace()` validate this numerically.

### 4. Checkpoint Persistence
After each training run, compressed numpy checkpoints are saved:
```
data/lora_checkpoints/nightly_YYYYMMDD_HHMMSS_adapter.npz
    ├── task_id
    ├── A           (r × d_in, projected LoRA A matrix)
    ├── B           (d_out × r, projected LoRA B matrix)
    ├── alpha       (LoRA scaling factor)
    └── samples_count
```

### 5. InsightForge Novelty Scoring (Weekly)
Runs every Sunday at 3 AM alongside the OPLoRA cycle:
1. Sample 500 interactions from replay buffer
2. Compute TF-IDF vectors (500-word vocabulary, stopword-filtered, L2-normalised)
3. Score each document by L2 distance from the centroid → novelty score in [0, 1]
4. Top-N high-novelty interactions are synthesised into insight reports
5. Persisted at `data/insights.json`

---

## Inspecting and Querying the Replay Buffer

```python
import pyarrow.parquet as pq, pandas as pd

# Read (no decryption needed if accessing directly on the machine)
df = pq.read_table("data/replay_buffer.parquet").to_pandas()

# Sample high-quality, unused records
sample = df[
    (df["faithfulness_score"] >= 0.92) &
    (~df["is_used_for_training"])
].sample(n=100)

# Module distribution
print(df.groupby("module").size())

# Average faithfulness per module
print(df.groupby("module")["faithfulness_score"].mean())
```

---

## Configuring OPLoRA Hyperparameters

Edit `.env` or pass overrides to relevant settings:

```bash
# Rank of OPLoRA subspace preservation (higher → more past knowledge kept)
OPLOРА_RANK_K=64              # Default: 64. Range: 8–256

# LoRA adapter rank (higher → more expressive new learning)
OPLOРА_LORA_R=16              # Default: 16. Range: 4–64

# LoRA alpha scaling factor
OPLOРА_LORA_ALPHA=32          # Default: 32. Typically set to 2 × r

# Training epochs per nightly cycle
OPLOРА_EPOCHS=3               # Default: 3. Range: 1–10

# Only sample interactions above this faithfulness score for training
OPLOРА_MIN_FAITHFULNESS=0.85  # Default: 0.85. Range: 0.0–1.0
```

**Trade-off guide:**
- `OPLOРА_RANK_K` high → more past knowledge preserved → less available subspace for new learning
- `OPLOРА_LORA_R` high → more expressive new learning → slightly higher VRAM during training
- `OPLOРА_MIN_FAITHFULNESS` high → higher-quality training data → fewer samples available

---

## Manually Triggering a Training Run

```bash
# Via API
curl -X POST http://localhost:8765/api/v1/learning/trigger

# Via Python
import asyncio
from src.config import get_settings
from src.learning.replay_buffer import ReplayBuffer
from src.learning.bitnet_trainer import BitNetTrainer

async def main():
    settings = get_settings()
    buf = ReplayBuffer(settings)
    await buf.initialize()
    trainer = BitNetTrainer(settings, buf)
    result = await trainer.run_oploora_cycle()
    print(result.to_dict())

asyncio.run(main())
```

---

## Extending AetherForge — Adding a New Module

1. Create `src/modules/mymodule/graph.py` following the descriptor pattern:

```python
def build_mymodule_graph() -> dict:
    return {
        "module_id": "mymodule",
        "run": run_mymodule,
    }

def run_mymodule(query: str, **kwargs) -> dict:
    return {"response": f"My module processed: {query}"}
```

2. Register in `src/modules/__init__.py`
3. Add to the module allowlist in `src/guardrails/default_policies.rego`:

```rego
valid_modules := {"ragforge","localbuddy","watchtower","streamsync","tunelab","mymodule"}
```

4. Add to the frontend sidebar in `frontend/src/components/ModuleTabs.tsx`

---

## Running the Test Suite

```bash
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run only math validation tests
pytest tests/test_oploRA_manager.py -v -k "test_projection"

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run async tests only
pytest tests/test_replay_buffer.py -v
```

Test `test_oploRA_manager.py::TestProjectorMath::test_projection_nullifies_past_subspace` is the most important — it numerically validates the core anti-forgetting guarantee.

---

# Part 5 — The Security / Compliance Officer

> **You are:** Responsible for ensuring AetherForge meets your organisation's security, privacy, audit, and compliance requirements.

---

## Data Sovereignty — What Stays on Device

| Data Type | Location | Leaves Machine? |
|---|---|---|
| User conversations | `data/replay_buffer.parquet` (encrypted) | Never |
| Documents (RAGForge) | `data/chroma/` (ChromaDB embedded) | Never |
| AI model weights | `models/` | Never (downloaded once) |
| Learning checkpoints | `data/lora_checkpoints/` | Never |
| Policy decisions log | `data/logs/colosseum.jsonl` | Never |
| Insights reports | `data/insights.json` | Never |

**No telemetry is sent externally by default.** `LANGFUSE_ENABLED=false` and `NEO4J_ENABLED=false` are the defaults. Even when enabled, these services run in local Docker containers.

The backend API binds exclusively to `127.0.0.1` — it is not accessible from the local network or the internet.

---

## Encryption

| Data at Rest | Encryption Method |
|---|---|
| Replay buffer (interactions) | Fernet (AES-128-CBC + HMAC-SHA256) |
| Key derivation | 32-byte random key stored at `data/.db_key` (mode 0600) |
| Session data (planned v1.1) | SQLCipher (AES-256) |

**Key management:** The encryption key is machine-local at `data/.db_key`. It is excluded from git via `.gitignore`. Loss of this file means the replay buffer history cannot be decrypted. Back up the key file separately if long-term replay buffer history is required.

**In transit:** All communication is over localhost loopback (`127.0.0.1`). No TLS is required for loopback — there is no network surface to attack.

---

## The Silicon Colosseum — Policy Engine

Silicon Colosseum is AetherForge's **mandatory, non-bypassable safety layer**. Every request — regardless of module or user — passes through it before execution.

### How It Works

1. **Pre-flight check:** Before any tool call or AI action, the request is submitted to the OPA policy engine
2. **OPA evaluation:** The Rego policy is evaluated against the full request context
3. **FSM state check:** The session's Finite State Machine validates that the proposed action is a legal state transition
4. **Result:** If either check fails, the request is denied and an audit log entry is written
5. **Post-output check:** After the AI generates a response, a second policy evaluation checks faithfulness and output safety

### Default Policy Rules (Auditable)

The policy is plain text in `src/guardrails/default_policies.rego`. Your compliance team can read and verify every rule:

```rego
# Tool call budget — prevents runaway agent loops
deny_reasons contains r if {
    input.tool_call_count > 8
    r := sprintf("Tool call budget exceeded: %v > 8", [input.tool_call_count])
}

# Faithfulness threshold — blocks hallucinated outputs
deny_reasons contains r if {
    input.module == "output_filter"
    input.faithfulness_score < 0.92
    r := sprintf("Faithfulness below threshold: %.2f < 0.92", [input.faithfulness_score])
}

# Prohibited content patterns
deny_reasons contains r if {
    pattern := prohibited_patterns[_]
    contains(lower(input.message), pattern)
    r := sprintf("Prohibited pattern detected: '%v'", [pattern])
}
```

### Adding Custom Compliance Rules

Use the **Policy Editor** (Policies tab in sidebar) to add organisation-specific rules. Any IT admin or compliance officer with access to the app can:

1. Click **Policies** in the left sidebar
2. View and edit the Rego policy in the Monaco editor
3. Add rules such as:

```rego
# Example: Block discussion of specific competitor names
deny_reasons contains "Competitor mention policy" if {
    contains(lower(input.message), "competitorname")
}

# Example: Restrict specific modules to admin sessions
deny_reasons contains "Module restricted" if {
    input.module == "tunelab"
    not input.context.is_admin
}
```

4. Click **Save & Reload** — the new policy takes effect immediately for all subsequent requests (no restart required)

---

## Audit Trail

Every policy decision is logged in structured JSON at `data/logs/colosseum.jsonl`:

```json
{
  "timestamp": "2026-03-02T07:45:12.341Z",
  "session_id": "1740887112-abc123",
  "module": "ragforge",
  "message_preview": "Find the contract terms for...",
  "allowed": true,
  "reason": "All policy checks passed",
  "deny_reasons": [],
  "fsm_state": "PROCESSING",
  "tool_call_count": 1,
  "faithfulness_score": null,
  "latency_ms": 2.3,
  "policy_version": "1.0.0"
}
```

**For compliance reporting:**
```bash
# Count denied requests in the last 24 hours
cat data/logs/colosseum.jsonl | \
  python3 -c "
import sys, json
from datetime import datetime, timezone, timedelta
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
denied = [l for l in map(json.loads, sys.stdin)
          if not l['allowed']
          and datetime.fromisoformat(l['timestamp']) > cutoff]
print(f'Denied requests (last 24h): {len(denied)}')
for d in denied:
    print(f'  {d[\"timestamp\"]} | {d[\"module\"]} | {d[\"deny_reasons\"]}')
"
```

---

## Air-Gap Compliance

For organisations that require complete network isolation:

1. **Pre-download all dependencies** on a connected machine:
   ```bash
   pip download -r requirements.txt -d ./offline_pkgs
   npm pack --pack-destination ./offline_pkgs
   huggingface-cli download microsoft/bitnet-b1.58-2b-4t-gguf --local-dir models/
   ```

2. **Transfer** the entire directory to the air-gapped machine (USB, secure file transfer)

3. **Install offline:**
   ```bash
   uv pip install --no-index --find-links ./offline_pkgs -e ".[dev]"
   npm install --offline
   ```

After installation, AetherForge operates with **zero network calls** of any kind.

---

## Compliance Checklist

| Requirement | AetherForge Status |
|---|---|
| Data never leaves the device | ✅ Guaranteed by architecture (loopback-only API) |
| Data encrypted at rest | ✅ Fernet AES-128-CBC on replay buffer |
| Audit log of all AI decisions | ✅ `data/logs/colosseum.jsonl` — every request logged |
| Ability to define custom safety rules | ✅ Rego policy editor with hot-reload |
| Deny-all safety default | ✅ OPA policy: `default allow := false` |
| Air-gap capable | ✅ Zero runtime network calls after installation |
| No vendor lock-in | ✅ Open source: Python, Rust, React, OPA, ChromaDB |
| Explainable AI outputs | ✅ X-Ray causal graph for every response |
| Catastrophic forgetting prevention | ✅ OPLoRA mathematical guarantee (SVD projector orthogonality) |
| Ability to disable learning | ✅ Set `OPLOРА_NIGHTLY_HOUR` to a time with `MIN_BATTERY=100` so it never fires |

---

## Frequently Asked Questions — Security Officers

**Q: Can AetherForge access files on the user's machine beyond the AetherForge directory?**
Only files explicitly ingested by an IT admin via the RAGForge ingestion API. The application has no file-system access beyond its own data directory and the specified model path.

**Q: Can users exfiltrate data by asking AetherForge to summarise sensitive documents it has access to?**
Yes, if the document has been ingested into RAGForge, a user with app access can query its contents. Control access at the document ingestion level — only ingest documents appropriate for your user base. Per-user document access controls are on the v1.1 roadmap.

**Q: What happens if the AI's safety policy is deleted or corrupted?**
If the Rego policy fails to parse or load, Silicon Colosseum falls back to the Python fallback evaluator, which has the same 9 core rules hardcoded. The system is never in a policy-free state.

**Q: Can we rotate the encryption key for the replay buffer?**
Delete `data/.db_key` — AetherForge generates a new one automatically on the next start. Note: existing replay buffer data encrypted with the old key cannot be read after rotation. Export any needed data first.

**Q: How do we prevent users from disabling the X-Ray or policy features?**
In v1.0, these are UI-level controls. Enterprise hardening (disabling UI controls for specific features) is on the v1.1 roadmap. In the interim, MDM configuration profiles can restrict `.env` modifications.

---

## Glossary

| Term | Meaning |
|---|---|
| **OPLoRA** | Orthogonal Projection LoRA — the algorithm that prevents the AI from forgetting past knowledge when it learns new things |
| **Silicon Colosseum** | AetherForge's safety engine — combines OPA Rego policies with a Finite State Machine |
| **OPA / Rego** | Open Policy Agent — an open-source policy engine used to evaluate declarative safety rules |
| **FSM** | Finite State Machine — a system that enforces which actions the AI is allowed to take at each point in a session |
| **RAGForge** | Retrieval-Augmented Generation module — searches your documents to ground AI answers in your organisation's knowledge |
| **Faithfulness score** | A measure (0–100%) of how reliably the AI's answer is grounded in known facts |
| **X-Ray mode** | A visual display of every reasoning step the AI took, shown as an interactive causal graph |
| **Replay buffer** | The encrypted database of past conversations used to train the AI each night |
| **InsightForge** | The weekly analysis engine that scores interactions by novelty and synthesises insight reports |
| **BitNet** | The 1.58-bit AI model used by AetherForge — designed to run efficiently on consumer hardware without cloud GPUs |
| **Causal graph** | A node-and-edge diagram showing the logical flow of the AI's reasoning for a specific response |
| **Metal** | Apple's GPU hardware acceleration API — used by AetherForge to run AI inference at full speed on M1/M2/M3 chips |

---

*AetherForge v1.0 User Manual — Built for privacy, transparency, and perpetual learning.*
*For technical issues: refer to the AetherForge_Complete_Build_Guide_v1.0.md or contact your IT administrator.*
