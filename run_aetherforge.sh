#!/bin/bash
echo "Stopping existing AetherForge backend processes..."
lsof -ti:8765 | xargs kill -9 2>/dev/null || true

echo "Starting AetherForge..."
cd /Users/macuser/AtherForge
source .venv/bin/activate
python3 -m uvicorn src.main:app --host 127.0.0.1 --port 8765 --log-level debug
