#!/bin/bash
# start_rag_demo.sh — Start all RAG demo services
#
# Services started:
#   - Milvus standalone (etcd + MinIO + Milvus)      via docker-compose.rag.yml
#   - vllm-embed-amx  (embedding AMX,    port 8002)  via docker-compose.rag.yml
#   - vllm-embed-no-amx (embedding no-AMX, port 8003) via docker-compose.rag.yml
#   - vllm-amx        (LLM AMX,          port 8000)  via docker-compose.rag.yml
#   - vllm-no-amx     (LLM no-AMX,       port 8001)  via docker-compose.rag.yml
#
# Prerequisites:
#   export HF_TOKEN=hf_...
#   export VLLM_LLM_MODEL=ibm-granite/granite-3.3-8b-instruct  (optional, this is the default)
#   export VLLM_EMBED_MODEL=BAAI/bge-m3                         (optional, this is the default)
#
# Usage:
#   bash start_rag_demo.sh            # start everything
#   bash start_rag_demo.sh --infra-only  # start Milvus only (skip vLLM containers)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.rag.yml"

INFRA_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--infra-only" ]] && INFRA_ONLY=true
done

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export your Hugging Face token first:"
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

if ! command -v docker compose &>/dev/null; then
    echo "ERROR: docker compose is not available."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║   AMX RAG Demo — Starting Services                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  LLM model:   ${VLLM_LLM_MODEL:-ibm-granite/granite-3.3-8b-instruct}"
echo "  Embed model: ${VLLM_EMBED_MODEL:-BAAI/bge-m3}"
echo ""

# ---------------------------------------------------------------------------
# Start Milvus infrastructure (etcd, minio, milvus-standalone)
# ---------------------------------------------------------------------------
echo ">>> Starting Milvus infrastructure..."
docker compose -f "$COMPOSE_FILE" up -d etcd minio milvus

echo "    Waiting for Milvus to be healthy..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:9091/healthz &>/dev/null; then
        echo "    ✅ Milvus is ready (${i}s)"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "    ❌ Milvus did not become healthy within 60 seconds"
        echo "    Check logs: docker compose -f docker-compose.rag.yml logs milvus"
        exit 1
    fi
    sleep 1
done

if $INFRA_ONLY; then
    echo ""
    echo "✅ Milvus started (--infra-only mode, vLLM containers skipped)."
    echo ""
    echo "Next: python3 rag_index_amx.py --skip-no-amx  (index the corpus)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Start vLLM embedding containers (smaller model, start faster)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Starting vLLM embedding containers (AMX :8002, no-AMX :8003)..."
docker compose -f "$COMPOSE_FILE" up -d vllm-embed-amx vllm-embed-no-amx

# ---------------------------------------------------------------------------
# Start vLLM LLM containers (larger model, start slower)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Starting vLLM LLM containers (AMX :8000, no-AMX :8001)..."
docker compose -f "$COMPOSE_FILE" up -d vllm-amx vllm-no-amx

# ---------------------------------------------------------------------------
# Wait for all vLLM containers to be healthy
# ---------------------------------------------------------------------------
wait_healthy() {
    local name="$1"
    local url="$2"
    local timeout="${3:-300}"
    echo -n "    Waiting for $name ($url)..."
    for i in $(seq 1 $timeout); do
        if curl -sf "$url/health" &>/dev/null; then
            echo " ✅ ready (${i}s)"
            return 0
        fi
        sleep 1
        [[ $((i % 30)) -eq 0 ]] && echo -n " ${i}s..."
    done
    echo " ❌ TIMEOUT after ${timeout}s"
    return 1
}

echo ""
echo ">>> Waiting for vLLM services to load models..."
echo "    (This takes 30-120 seconds per container depending on model size)"
echo ""

ALL_OK=true
wait_healthy "vllm-embed-amx"   "http://localhost:8002" 180 || ALL_OK=false
wait_healthy "vllm-embed-no-amx" "http://localhost:8003" 180 || ALL_OK=false
wait_healthy "vllm-amx"          "http://localhost:8000" 300 || ALL_OK=false
wait_healthy "vllm-no-amx"       "http://localhost:8001" 300 || ALL_OK=false

echo ""
if $ALL_OK; then
    echo "✅ All services are ready!"
    echo ""
    echo "  Milvus:          localhost:19530"
    echo "  vLLM AMX (LLM):  http://localhost:8000"
    echo "  vLLM no-AMX LLM: http://localhost:8001"
    echo "  vLLM AMX embed:  http://localhost:8002"
    echo "  vLLM no-AMX emb: http://localhost:8003"
    echo ""
    echo "Next steps:"
    echo "  1. Index the corpus:    python3 rag_index_amx.py"
    echo "  2. Run RAG query:       python3 rag_query_amx.py"
    echo "  3. Launch Flask demo:   python3 PWI-Flask-RAG.py"
    echo "     Open: http://localhost:5002"
else
    echo "⚠️  Some services did not start cleanly. Check logs:"
    echo "    docker compose -f docker-compose.rag.yml logs --tail=50 <service>"
fi
