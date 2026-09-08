#!/bin/bash
set -e

echo "================================================="
echo "🚀 Starting Local RAG System on Hugging Face Spaces"
echo "================================================="

# Set environment defaults if not provided
export HF_HOME=${HF_HOME:-"/app/models/huggingface"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/app/models/huggingface"}
export QDRANT_STORAGE_PATH=${QDRANT_STORAGE_PATH:-"/app/data/qdrant_db"}
export BM25_INDEX_PATH=${BM25_INDEX_PATH:-"/app/data/production_bm25.pkl"}

# Start FastAPI backend service on localhost:8000 in background
echo ">>> Launching FastAPI Backend on http://127.0.0.1:8000..."
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Poll until FastAPI backend health check responds
echo ">>> Waiting for FastAPI Backend to initialize..."
MAX_ATTEMPTS=60
ATTEMPT=0

until curl -s http://127.0.0.1:8000/api/v1/health | grep -q '"status":"online"'; do
    ATTEMPT=$((ATTEMPT+1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "❌ FastAPI Backend failed to start within timeout."
        exit 1
    fi
    echo "    Waiting for Backend... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

echo "✅ FastAPI Backend is online and healthy!"

# Start Streamlit Frontend service on 0.0.0.0:7860 (Hugging Face Spaces default port)
echo ">>> Launching Streamlit Web UI on http://0.0.0.0:7860..."
exec python -m streamlit run frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
