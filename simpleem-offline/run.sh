#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== InsightAI Offline — Starting ==="
pip install -r backend/requirements.txt -q 2>/dev/null
python -m uvicorn backend.main:app --host 0.0.0.0 --port 9000 --reload &
BACKEND_PID=$!
cd frontend && npm install --silent 2>/dev/null && npm run dev &
FRONTEND_PID=$!
echo "Backend: http://localhost:9000"
echo "Frontend: http://localhost:4000"
wait $BACKEND_PID $FRONTEND_PID
