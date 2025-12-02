#!/bin/bash
# Start script for the Market Map backend

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use!"
    echo "Please stop the existing server or use a different port."
    exit 1
fi

echo "🚀 Starting Market Map Backend Server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

