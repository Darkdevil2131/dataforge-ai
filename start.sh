#!/bin/bash
# DataForge AI — One-command startup
# Usage: ./start.sh

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         DataForge AI — Starting          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
  echo "❌ Python 3 not found. Install from https://python.org"
  exit 1
fi

# ── Check Node ────────────────────────────────────────────────────────────────
if ! command -v node &> /dev/null; then
  echo "❌ Node.js not found. Install from https://nodejs.org"
  exit 1
fi

# ── Backend setup ─────────────────────────────────────────────────────────────
echo "▶  Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
  echo "   Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "   Installing Python dependencies..."
pip install -r requirements.txt -q

echo "   Starting backend on http://localhost:8000 ..."
python3 main.py &
BACKEND_PID=$!

cd ..

# ── Frontend setup ────────────────────────────────────────────────────────────
echo ""
echo "▶  Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
  echo "   Installing Node dependencies..."
  npm install --silent
fi

# Point frontend at local backend
echo "VITE_API_URL=" > .env.local

echo "   Starting frontend on http://localhost:5173 ..."
npm run dev &
FRONTEND_PID=$!

cd ..

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅ DataForge AI is running!             ║"
echo "║                                          ║"
echo "║  Frontend  →  http://localhost:5173      ║"
echo "║  Backend   →  http://localhost:8000      ║"
echo "║  API Docs  →  http://localhost:8000/docs ║"
echo "║                                          ║"
echo "║  Press Ctrl+C to stop both servers       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Keep running until Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
