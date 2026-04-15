#!/bin/bash
# stop_rag_demo.sh — Stop all RAG demo services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.rag.yml"

echo "Stopping all RAG demo services..."
docker compose -f "$COMPOSE_FILE" down

echo "✅ All services stopped."
