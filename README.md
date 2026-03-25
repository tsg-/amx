# Intel AMX Demo with vLLM

Benchmark and demo suite that quantifies the performance impact of **Intel Advanced Matrix Extensions (AMX)** on CPU-based LLM inference using [vLLM](https://github.com/vllm-project/vllm).

Two Docker images are built from the same vLLM source — one with AMX enabled, one without — and run side-by-side so every comparison happens on identical hardware.

---

## Hardware Requirements

- **Intel 4th-gen Xeon Scalable** (Sapphire Rapids) or later for AMX support
- Docker with BuildKit enabled
- Hugging Face account with access to the model (see [§ Environment Variables](#step-2--environment-variables))

Verify that the host CPU exposes the required AMX flags:

```bash
grep -o 'amx[^ ]*' /proc/cpuinfo | sort -u
# expected output:
#   amx_bf16
#   amx_int8
#   amx_tile
```

If any of these flags are missing the AMX image will still run, but AMX tile units will not be available and results will be identical to the no-AMX container.

---

## Repository Structure

```
.
├── Dockerfile.cpu.amx        # vLLM CPU image — AMX + AVX-512 BF16 enabled
├── Dockerfile.cpu.no-amx     # vLLM CPU image — AVX-512 BF16 only, AMX disabled
├── build_docker_amx.sh       # Build the AMX image
├── build_docker_no_amx.sh    # Build the no-AMX image
├── start_amx_containers.sh   # Launch both containers and wait for health
├── stop_amx_containers.sh    # Stop both containers
├── restart_amx_containers.sh # Restart both containers and wait for health
├── show_docker.sh            # List Docker images and running containers
├── check_vllm_services.sh    # Poll health endpoints until both are ready
├── test_vLLM.sh              # Quick smoke test — sends "Hello" to each container
├── benchmark_amx.sh          # Automated benchmark runner (wraps query_vllm_amx.py)
├── query_vllm_amx.py         # Python benchmark client — TTFT, prefill tok/s, decode tok/s
├── PWI-Flask-2vLLM.py        # Flask demo app v1 — select service & question in browser
└── PWI-Flask-2vLLM-v2.py     # Flask demo app v2 — multi-run, cache-busted, richer metrics
```

---

## Step 1 — Build the Docker Images

> **Note:** The Dockerfiles must be built from inside a cloned vLLM repository because they `COPY` the source tree.

```bash
# Clone vLLM and copy the Dockerfiles in
git clone https://github.com/vllm-project/vllm.git
cp Dockerfile.cpu.amx    vllm/
cp Dockerfile.cpu.no-amx vllm/
cd vllm
```

Build both images (each takes 20–40 min the first time; subsequent builds are fast thanks to ccache):

```bash
# AMX image — AMX tile units + Intel OpenMP enabled
bash ../build_docker_amx.sh

# No-AMX baseline image — AVX-512 BF16 only
bash ../build_docker_no_amx.sh
```

The scripts are thin wrappers around:

| Image tag           | Key build args                                          |
|---------------------|---------------------------------------------------------|
| `vllm-cpu-amx:latest`    | `VLLM_CPU_AMXBF16=1 VLLM_CPU_AVX512BF16=1 VLLM_CPU_AVX512VNNI=1` |
| `vllm-cpu-no-amx:latest` | `VLLM_CPU_AMXBF16=0 VLLM_CPU_AVX512BF16=1 VLLM_CPU_AVX512VNNI=1` |

Confirm both images exist:

```bash
bash show_docker.sh
```

---

## Step 2 — Environment Variables

Export your Hugging Face token before starting containers:

```bash
export HF_TOKEN=hf_...
```

The model used by default is `ibm-granite/granite-3.3-8b-instruct` (BF16, ~16 GB). Weights are cached in `~/.cache/huggingface` and mounted into each container.

---

## Step 3 — Start the Containers

```bash
bash start_amx_containers.sh
```

This launches two detached containers and blocks until both `/health` endpoints respond:

| Container      | Port | ISA                    |
|----------------|------|------------------------|
| `vllm-amx`     | 8000 | `AVX512_CORE_AMX`      |
| `vllm-no-amx`  | 8001 | `AVX512_CORE_BF16`     |

The AMX container binds OMP threads to cores 0–19; the no-AMX container to cores 20–39. Adjust `VLLM_CPU_OMP_THREADS_BIND` in the script to match your socket/NUMA topology.

---

## Step 4 — Verify the Containers

```bash
# Check both health endpoints
bash check_vllm_services.sh

# Quick functional smoke test (sends "Hello", expects ≤10 tokens back)
bash test_vLLM.sh

# Show running containers and images
bash show_docker.sh
```

### Verify oneDNN kernel dispatch with `DNNL_VERBOSE`

`DNNL_VERBOSE` controls oneDNN's kernel-selection logging. It is set to `0` (silent) by default in `start_amx_containers.sh`. To confirm that the AMX container is actually dispatching AMX kernels, restart it with `DNNL_VERBOSE=1`:

```bash
# Stop the running AMX container first
docker stop vllm-amx

# Relaunch with verbose oneDNN logging, capturing output
docker run --rm \
  --name vllm-amx-verbose \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN} \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  -e VLLM_CPU_OMP_THREADS_BIND="0-19" \
  -e DNNL_MAX_CPU_ISA=AVX512_CORE_AMX \
  -e VLLM_CPU_SGL_KERNEL=1 \
  -e DNNL_VERBOSE=1 \
  -e LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so" \
  --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  --shm-size=4g \
  -p 8000:8000 \
  vllm-cpu-amx:latest \
  --model ibm-granite/granite-3.3-8b-instruct \
  --dtype bfloat16 2>&1 | grep -i "avx512_core_amx"
```

While the container is handling a request, lines like the following confirm AMX kernels are being dispatched:

```
dnnl_verbose,exec,cpu,matmul,...,avx512_core_amx,...
```

For the no-AMX container the same lines will show `avx512_core_bf16` instead — confirming AMX tile units are correctly disabled:

```
dnnl_verbose,exec,cpu,matmul,...,avx512_core_bf16,...
```

Set `DNNL_VERBOSE=0` (the default) for normal operation — verbose logging adds overhead and produces large amounts of output under load.

---

## Step 5 — Run the Benchmark

### Automated runner

```bash
bash benchmark_amx.sh
```

Runs the **RAG / summarisation** scenario by default (2 K-token prompt, 50-token answer) — the most realistic test case for showing AMX value.

### Manual runs with `query_vllm_amx.py`

```bash
pip install openai rich

# Default prompt, 3 runs
python3 query_vllm_amx.py

# Custom prompt, 5 runs, 1-token output (pure prefill, max AMX signal)
python3 query_vllm_amx.py \
  --prompt "Explain how AMX tile instructions accelerate transformer prefill." \
  --runs 5 \
  --max-tokens 1 \
  --cooldown 3

# Custom endpoints (e.g. two separate nodes)
python3 query_vllm_amx.py \
  --amx-url http://node1:8000 \
  --no-amx-url http://node2:8001 \
  --model ibm-granite/granite-3.3-8b-instruct \
  --runs 5 --max-tokens 50 --cooldown 3

# List built-in sample prompts
python3 query_vllm_amx.py --list-prompts
```

**Key CLI options:**

| Option | Default | Description |
|---|---|---|
| `--runs` | 3 | Benchmark runs per container |
| `--max-tokens` | 256 | Output length (use `1` for pure-prefill) |
| `--cooldown` | 2 | Seconds between runs (reduces DRAM contention) |
| `--skip-health` | — | Skip `/health` check |

**Metrics reported:**

| Metric | What it measures |
|---|---|
| Avg / P50 / P95 TTFT | Time to first token — dominated by prefill GEMM |
| Prefill throughput (tok/s) | `prompt_tokens / TTFT` — the primary AMX signal |
| Avg Total Time | TTFT + decode; diluted by output length |
| Decode throughput (tok/s) | Memory-bandwidth bound — identical for both |

---

## Step 6 — Interactive Demo (Flask)

A browser-based demo app is included for live demonstrations.

```bash
python3 PWI-Flask-2vLLM-v2.py
# open http://localhost:5001
```

Select a vLLM service (AMX or no-AMX), pick a question, and watch the streamed response with live TTFT and tokens/sec metrics.

Notes:
- Long-context prompts so prefill dominates and the AMX advantage is clearly visible
- Cache busting per run (unique prefix defeats vLLM prefix caching)
- `stream_options: include_usage` for accurate prompt token counts
- Multiple runs with Avg TTFT, P95 TTFT, Prefill tok/s, Decode tok/s
- Sequential execution (AMX first, then no-AMX) to avoid DRAM contention noise
- Default `max_tokens=50` (RAG sweet spot: ~46% prefill, ~3× end-to-end speedup)

---

## Container Management

```bash
# Stop both containers
bash stop_amx_containers.sh

# Restart both containers (waits for health)
bash restart_amx_containers.sh
```

---

## Benchmark Results Sample Summary

Full results and analysis are in [`perftests.md`](perftests.md). Key highlights:

### Context length sweep (`max-tokens=1`, pure prefill, cache-busted)

| Prompt tokens | AMX TTFT | No-AMX TTFT | Speedup | AMX Prefill tok/s | No-AMX Prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 550 | 777 ms | 4,680 ms | **6.0×** | 708 | 118 |
| 1,032 | 1,317 ms | 8,347 ms | **6.3×** | 784 | 124 |
| 1,877 | 2,379 ms | 14,646 ms | **6.2×** | 789 | 128 |
| 4,393 | 5,843 ms | 34,857 ms | **6.0×** | 752 | 126 |
| 8,343 | 12,389 ms | 67,175 ms | **5.4×** | 673 | 124 |

### Realistic workload — 2,666-token prompt (RAG / summarisation)

| Scenario | Output tokens | TTFT speedup | Total time speedup |
|---|---:|---:|---:|
| Pure prefill benchmark | 1 | **6.1×** | **6.1×** |
| Summarisation (RAG sweet spot) | 50 | **6.1×** | **3.3×** |
| Detailed answer | 200 | **6.2×** | **1.9×** |

### Why AMX helps prefill but not decode

- **Prefill** — large matrix multiplications across the full prompt. AMX 16×16 BF16 tile-MACC instructions directly accelerate this compute-bound GEMM phase.
- **Decode** — one token at a time, loading full weight matrices each step. Memory-bandwidth bound; AMX provides no benefit.

The **50-150 token output / 2,600-token prompt** scenario is the most honest demo: it represents a genuine RAG or document Q&A workload and delivers a meaningful **3.3× end-to-end speedup** (7.5 s → 25 s) that users can feel.

---

## License

See [LICENSE](LICENSE).
