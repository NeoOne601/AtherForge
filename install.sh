#!/usr/bin/env bash
# AetherForge v1.0 — install.sh
# ─────────────────────────────────────────────────────────────────
# One-shot installer for macOS M1/M2 (Apple Silicon).
# Installs all system deps, Python env, Node modules, Rust, and
# downloads the default BitNet GGUF model.
#
# Usage:  chmod +x install.sh && ./install.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${CYAN}[AetherForge]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[✗]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log "AetherForge v1.0 Installer — Apple Silicon Edition"
log "=================================================="

# ── 1. System Requirements Check ─────────────────────────────────
log "Checking system requirements..."
[[ "$(uname)" == "Darwin" ]] || err "This installer targets macOS only. Linux support: see README."
[[ "$(uname -m)" == "arm64" ]] && ok "Apple Silicon detected" || warn "Intel Mac: Metal acceleration disabled."

# ── 2. Homebrew ──────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  log "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
ok "Homebrew ready"

# ── 3. System Dependencies ───────────────────────────────────────
log "Installing system dependencies via Homebrew..."
brew install --quiet \
  python@3.12 \
  node@20 \
  rust \
  cmake \
  pkg-config \
  openssl \
  opa \
  neo4j \
  || true
ok "System deps installed"

# ── 4. uv (ultra-fast Python package manager) ───────────────────
if ! command -v uv &>/dev/null; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi
ok "uv ready"

# ── 5. Python Virtual Environment ───────────────────────────────
log "Creating Python 3.12 virtual environment..."
uv venv --python=3.12 .venv
source .venv/bin/activate

# Install llama-cpp-python with Apple Metal (MPS) support
log "Installing llama-cpp-python with Metal support (this may take ~5 min)..."
CMAKE_ARGS="-DLLAMA_METAL=on -DLLAMA_NATIVE=off" \
  FORCE_CMAKE=1 \
  uv pip install llama-cpp-python --no-cache-dir

# Install remaining Python dependencies
log "Installing Python dependencies..."
uv pip install -e ".[dev]"
ok "Python environment ready"

# ── 6. Node / Frontend ───────────────────────────────────────────
log "Installing Node.js frontend dependencies..."
npm install --legacy-peer-deps
ok "Node dependencies installed"

# ── 7. Rust / Tauri CLI ──────────────────────────────────────────
log "Installing Tauri CLI..."
cargo install tauri-cli --version "^2.1" --locked 2>/dev/null || true
ok "Tauri CLI ready"

# ── 8. OPA Binary ────────────────────────────────────────────────
if ! command -v opa &>/dev/null; then
  log "Installing OPA..."
  brew install opa
fi
ok "OPA ready at $(command -v opa)"

# ── 9. Download BitNet GGUF Model ────────────────────────────────
MODEL_DIR="$SCRIPT_DIR/models"
MODEL_FILE="$MODEL_DIR/bitnet-b1.58-2b-4t.gguf"
mkdir -p "$MODEL_DIR"

if [[ ! -f "$MODEL_FILE" ]]; then
  log "Downloading BitNet 1.58-bit model (~1.2 GB)..."
  log "Model: microsoft/bitnet-b1.58-2b-4t-gguf"
  # Using huggingface-cli if available, else curl
  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download microsoft/bitnet-b1.58-2b-4t-gguf \
      --include "*.gguf" \
      --local-dir "$MODEL_DIR"
  else
    warn "huggingface-cli not found. Attempting direct download..."
    curl -L --progress-bar \
      "https://huggingface.co/microsoft/bitnet-b1.58-2b-4t-gguf/resolve/main/ggml-model-i2_s.gguf" \
      -o "$MODEL_FILE" || warn "Model download failed — place .gguf file manually in models/"
  fi
else
  ok "BitNet model already present: $MODEL_FILE"
fi

# ── 10. Create .env from template ────────────────────────────────
if [[ ! -f ".env" ]]; then
  log "Creating .env from template..."
  cat > .env <<'ENVEOF'
# AetherForge v1.0 — Local Environment Config
# ─────────────────────────────────────────────
# SECURITY: Never commit this file. It's in .gitignore.

# ── Core ──────────────────────────────────────────────────────
AETHERFORGE_ENV=development
AETHERFORGE_HOST=127.0.0.1
AETHERFORGE_PORT=8765
AETHERFORGE_LOG_LEVEL=info

# ── Model ─────────────────────────────────────────────────────
BITNET_MODEL_PATH=./models/bitnet-b1.58-2b-4t.gguf
BITNET_N_CTX=4096
BITNET_N_GPU_LAYERS=-1
BITNET_N_THREADS=8

# ── OPA / Guardrails ──────────────────────────────────────────
OPA_MODE=embedded
OPA_SERVER_URL=http://localhost:8181
SILICON_COLOSSEUM_MAX_TOOL_CALLS=8
SILICON_COLOSSEUM_MIN_FAITHFULNESS=0.92

# ── Storage ───────────────────────────────────────────────────
DATA_DIR=./data
REPLAY_BUFFER_PATH=./data/replay_buffer.parquet
SQLCIPHER_KEY_FILE=./data/.db_key
CHROMA_PATH=./data/chroma

# ── Telemetry (local Langfuse — optional) ─────────────────────
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=af_local
LANGFUSE_SECRET_KEY=af_local_secret

# ── Neo4j (X-Ray causal graph — optional) ─────────────────────
NEO4J_ENABLED=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=aetherforge123

# ── Nightly Learning Scheduler ────────────────────────────────
OPLOРА_NIGHTLY_HOUR=3
OPLOРА_NIGHTLY_MINUTE=0
OPLOРА_MIN_BATTERY_PCT=30
ENVEOF
  ok ".env created"
else
  ok ".env already exists — skipping"
fi

# ── 11. Data Directories ─────────────────────────────────────────
log "Creating data directories..."
mkdir -p data/{chroma,replay,sessions,logs} models
touch data/.gitkeep models/.gitkeep
ok "Data directories ready"

# ── 12. Final Summary ────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════════"
ok "AetherForge v1.0 installation complete!"
log "═══════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Review .env and adjust settings if needed"
echo "  2. Start dev mode:  ./run_dev.sh"
echo "  3. Open Tauri app:  npm run tauri:dev"
echo ""
echo "  (Optional) Self-hosted telemetry stack:"
echo "  docker compose up -d"
echo ""
