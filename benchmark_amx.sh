#!/bin/bash
# benchmark_amx.sh — AMX vs No-AMX vLLM benchmark runner
#
# Prerequisites:
#   pip install openai rich
#
# Both containers must be running:
#   ./run_amx.sh
#
# See perftests.md for full results and realism analysis.

pip install openai rich

# ---------------------------------------------------------------------------
# 1. Pure prefill benchmark (synthetic — isolates AMX GEMM advantage)
#    max-tokens=1 → Total Time ≈ TTFT, no decode noise
#    Expected: ~6x TTFT speedup
# ---------------------------------------------------------------------------
# echo "=== Test 1: Pure prefill (max-tokens=1) ==="
# python3 query_vllm_amx.py \
#   --prompt "You are evaluating Intel AMX performance for LLM inference on Xeon processors. AMX tile instructions perform 16x16 BF16 matrix multiply-accumulate in a single op. Explain how this accelerates the prefill phase of transformer inference compared to AVX-512 BF16 VNNI." \
#   --runs 5 \
#   --max-tokens 1 \
#   --cooldown 3

# ---------------------------------------------------------------------------
# 2. RAG / summarisation (realistic — long context, short answer)
#    Matches: enterprise search, document Q&A, copilot tools
#    Expected: ~6x TTFT, ~3x total time speedup
# ---------------------------------------------------------------------------
echo ""
echo "=== Test 2: RAG / summarisation — 2K prompt, 50-token answer (realistic) ==="
python3 query_vllm_amx.py \
  --prompt "$(head -c 8000 vllm/v1/worker/cpu_worker.py)

Briefly summarize what this file does and list its main classes." \
  --runs 5 \
  --max-tokens 50 \
  --cooldown 3

# ---------------------------------------------------------------------------
# 3. Detailed answer (realistic — long context, longer output)
#    Matches: report generation, structured output, longer explanations
#    Expected: ~6x TTFT, ~2x total time speedup
# ---------------------------------------------------------------------------
# echo ""
# echo "=== Test 3: Detailed answer — 2K prompt, 200-token answer (realistic) ==="
# python3 query_vllm_amx.py \
#   --prompt "$(head -c 8000 vllm/v1/worker/cpu_worker.py)
# 
# Briefly summarize what this file does and list its main classes." \
#   --runs 5 \
#   --max-tokens 200 \
#   --cooldown 3

# ---------------------------------------------------------------------------
# Custom endpoints (e.g. different nodes)
# ---------------------------------------------------------------------------
# python3 query_vllm_amx.py \
#   --amx-url http://node1:8000 \
#   --no-amx-url http://node2:8001 \
#   --model ibm-granite/granite-3.3-8b-instruct \
#   --runs 5 --max-tokens 50 --cooldown 3
