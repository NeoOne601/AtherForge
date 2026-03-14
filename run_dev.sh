#!/usr/bin/env bash
# AetherForge v1.0 — run_dev.sh
# ─────────────────────────────────────────────────────────────────
# Starts the full AetherForge development stack:
#   1. FastAPI backend (port 8765)
#   2. Vite frontend dev server (port 1420)
#   3. Tauri desktop shell (wraps both)
#   4. Optionally: OPA server, Neo4j, Langfuse via Docker
#
# Usage:  ./run_dev.sh [--no-docker] [--backend-only] [--frontend-only]
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[AetherForge]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse Args ───────────────────────────────────────────────────
NO_DOCKER=false
BACKEND_ONLY=false
FRONTEND_ONLY=false
for arg in "$@"; do
  case $arg in
    --no-docker)    NO_DOCKER=true ;;
    --backend-only) BACKEND_ONLY=true ;;
    --frontend-only) FRONTEND_ONLY=true ;;
  esac
done

# ── Activate Python venv ──────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  ok "Python venv activated"
else
  warn "No .venv found. Run ./install.sh first."
  exit 1
fi

# ── Load .env ─────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  set -o allexport
  source .env
  set +o allexport
  ok ".env loaded"
fi

# ── Create data dirs ──────────────────────────────────────────────
mkdir -p data/{chroma,replay,sessions,logs} models

# ── Optional: Docker services ─────────────────────────────────────
if [[ "$NO_DOCKER" == "false" ]] && command -v docker &>/dev/null; then
  log "Starting optional self-hosted services (Langfuse, Neo4j, OPA)..."
  docker compose up -d --quiet-pull 2>/dev/null || warn "Docker services failed to start — continuing without them"
fi

# ── Cleanup on exit ───────────────────────────────────────────────
PIDS=()
cleanup() {
  log "Shutting down AetherForge dev stack..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
  log "Shutdown complete."
}
trap cleanup EXIT INT TERM

# ── Start FastAPI backend ──────────────────────────────────────────
if [[ "$FRONTEND_ONLY" == "false" ]]; then
  log "Starting FastAPI backend on http://127.0.0.1:${AETHERFORGE_PORT:-8765}..."
  uvicorn src.main:app \
    --host "${AETHERFORGE_HOST:-127.0.0.1}" \
    --port "${AETHERFORGE_PORT:-8765}" \
    --reload \
    --log-level "${AETHERFORGE_LOG_LEVEL:-info}" \
    --reload-dir src \
    &
  PIDS+=($!)
  ok "Backend started (PID $!)"
  # Give it a moment to bind
  sleep 2
fi

# ── Start Vite frontend dev server ────────────────────────────────
if [[ "$BACKEND_ONLY" == "false" ]]; then
  log "Starting Vite frontend dev server on http://localhost:1420..."
  npm run dev &
  PIDS+=($!)
  ok "Frontend started (PID $!)"
  sleep 2
fi

# ── Health check ──────────────────────────────────────────────────
if [[ "$FRONTEND_ONLY" == "false" ]]; then
  log "Waiting for backend to be healthy..."
  for i in {1..20}; do
    if curl -sf "http://127.0.0.1:${AETHERFORGE_PORT:-8765}/health" > /dev/null 2>&1; then
      ok "Backend is healthy!"
      break
    fi
    sleep 1
    [[ $i -eq 20 ]] && warn "Backend did not respond in time — check logs"
  done
fi

log "═════════════════════════════════════════════════════════"
ok "AetherForge dev stack is running!"
log "  Backend API:  http://127.0.0.1:${AETHERFORGE_PORT:-8765}"
log "  API Docs:     http://127.0.0.1:${AETHERFORGE_PORT:-8765}/docs"
log "  Frontend:     http://localhost:1420"
log "  Desktop:      run 'npm run tauri:dev' in a separate terminal"
log "═════════════════════════════════════════════════════════"
log "Press Ctrl+C to stop all services"

# ── Wait for all background processes ────────────────────────────
wait
