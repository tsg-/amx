#!/bin/bash

# Wait for both to be ready
until curl -sf http://localhost:8000/health; do echo "waiting for AMX..."; sleep 5; done
until curl -sf http://localhost:8001/health; do echo "waiting for No-AMX..."; sleep 5; done
echo "Both containers ready"
