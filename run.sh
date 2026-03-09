#!/bin/bash
# AetherForge v1.0 Unified Startup Script

# 1. Surgical Ollama Reset
echo "🔄 Resetting Ollama..."
OLLAMA_PID=$(lsof -ti:11434)
if [ ! -z "$OLLAMA_PID" ]; then
    echo "Stopping existing Ollama process (PID: $OLLAMA_PID)..."
    kill -9 $OLLAMA_PID
fi
sleep 1
ollama serve > /dev/null 2>&1 &
echo "✅ Ollama restarted in background."
# 2. Surgical Port Cleanup
echo "🔄 Clearing AetherForge ports (8765, 1420-1425)..."
PIDS=$(lsof -ti:8765,1420,1421,1422,1423,1424,1425)
if [ ! -z "$PIDS" ]; then
    echo "Stopping existing processes: $PIDS"
    kill -9 $PIDS > /dev/null 2>&1
fi
sleep 1

# 3. Virtual Environment Check
if [ ! -d ".venv" ]; then
    echo "❌ Error: .venv not found. Please run install.sh first."
    exit 1
fi
source .venv/bin/activate

# 4. Backend Launch with correct PYTHONPATH
echo "⚙️ Starting AetherForge Backend..."
export PYTHONPATH=$(pwd)
# Run backend in background so we can show next steps
python3 src/main.py serve --port 8765 --host 127.0.0.1 > backend.log 2>&1 &
BACKEND_PID=$!

echo "✅ Backend started in background (PID: $BACKEND_PID). Logs are in backend.log."
echo ""
echo "🚀 To launch the Frontend, choose one:"
echo "   1. Desktop (Recommended): npm run tauri:dev"
echo "   2. Browser: cd frontend && npm run dev"
echo ""
echo "Press Ctrl+C to stop the backend session."
wait $BACKEND_PID
