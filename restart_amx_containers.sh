#!/bin/bash

# AMX container — port 8000
docker restart vllm-amx

# No-AMX container — port 8001
docker restart vllm-no-amx
	
# Wait for both to be ready
until curl -sf http://localhost:8000/health; do echo "waiting for AMX..."; sleep 5; done
until curl -sf http://localhost:8001/health; do echo "waiting for No-AMX..."; sleep 5; done

echo "Both containers restarted"
