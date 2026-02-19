#!/bin/bash
# InsightAI - Conversation Intelligence POC
# Starts both backend (FastAPI) and frontend (Vite dev server)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================="
echo "  InsightAI - Conversation Intelligence POC"
echo "=================================="
echo ""

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg is not installed. Video processing will fail."
    echo "Install with: brew install ffmpeg"
    echo ""
fi

# Check for demo data
if [ ! -f "$SCRIPT_DIR/data.db" ]; then
    echo "[0/3] Seeding demo data..."
    cd "$SCRIPT_DIR" && python3 seed_demo_data.py
    echo ""
fi

# Install backend dependencies
echo "[1/3] Installing Python dependencies..."
pip3 install -r "$SCRIPT_DIR/backend/requirements.txt" --quiet 2>/dev/null

# Install frontend dependencies
echo "[2/3] Installing frontend dependencies..."
cd "$SCRIPT_DIR/frontend" && npm install --silent 2>/dev/null

echo "[3/3] Starting servers..."
echo ""
echo "  Dashboard: http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""

# Start backend in background
cd "$SCRIPT_DIR"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
cd "$SCRIPT_DIR/frontend"
npx vite --port 5173 &
FRONTEND_PID=$!

# Handle cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
