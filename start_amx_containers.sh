#!/bin/bash

# AMX container — port 8000
docker run -d --rm \
  --name vllm-amx \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN} \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  -e VLLM_CPU_OMP_THREADS_BIND="0-19" \
  -e DNNL_MAX_CPU_ISA=AVX512_CORE_AMX \
  -e VLLM_CPU_SGL_KERNEL=1 \
  -e LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so" \
  --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  --shm-size=4g \
  -p 8000:8000 \
  vllm-cpu-amx:latest \
  --model ibm-granite/granite-3.3-8b-instruct \
  --dtype bfloat16

# No-AMX container — port 8001
docker run -d --rm \
  --name vllm-no-amx \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN} \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  -e VLLM_CPU_OMP_THREADS_BIND="20-39" \
  -e DNNL_MAX_CPU_ISA=AVX512_CORE_BF16 \
  -e ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 \
  -e VLLM_CPU_SGL_KERNEL=0 \
  -e LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so" \
  --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  --shm-size=4g \
  -p 8001:8000 \
  vllm-cpu-no-amx:latest \
  --model ibm-granite/granite-3.3-8b-instruct \
  --dtype bfloat16

# Wait for both to be ready
until curl -sf http://localhost:8000/health; do echo "waiting for AMX..."; sleep 5; done
until curl -sf http://localhost:8001/health; do echo "waiting for No-AMX..."; sleep 5; done
echo "Both containers ready"

# LD_PRELOAD is required — vLLM's CPU worker explicitly checks for libiomp at startup
# and raises RuntimeError if missing. tcmalloc reduces malloc contention under heavy threading.
# -e LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so"
