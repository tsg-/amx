# Intel AMX Demo — vLLM + RAG + VectorDB

Benchmark and demo suite that quantifies the performance impact of **Intel Advanced Matrix Extensions (AMX)** across the full CPU-based AI inference stack:

| Demo | What it measures | Files |
|---|---|---|
| **LLM Inference** | TTFT, prefill tok/s, decode tok/s | `query_vllm_amx.py`, `PWI-Flask-2vLLM-v2.py` |
| **Embedding & Indexing** | Document encoding throughput for vector DB | `rag_index_amx.py` |
| **End-to-end RAG** | Embed → [Milvus](https://milvus.io/) search → LLM answer latency | `rag_query_amx.py`, `PWI-Flask-RAG.py` |

The same two Docker images power every comparison — AMX enabled on one, disabled on the other — so results reflect the ISA difference only, not hardware variance.

---

## Why RAG Is the Best AMX Showcase

RAG workloads combine two operations where AMX shines and one where it doesn't:

```
Long retrieved context                        Short answer
(2,000–4,000 tokens)                         (50–200 tokens)
        │                                           │
        ▼                                           ▼
  PREFILL PHASE                            DECODE PHASE
  (process full prompt)                    (one token at a time)
  Compute-bound GEMM ← AMX 6× faster      Memory-bandwidth bound
  Large matrix multiply                    Matrix-vector multiply
  AMX tile instructions ✅                 DRAM throughput bottleneck ❌
```

**Why RAG specifically:**
- **Embedding generation** (encoding documents and queries) is pure transformer inference — the same compute-bound GEMM as LLM prefill. AMX accelerates it ~3–6×.
- **LLM prefill** over the retrieved context (~2,600 tokens) is the dominant latency. AMX delivers ~6× TTFT speedup here.
- **Short answer** (50 tokens) keeps decode time small, so the prefill AMX win carries through to a **~3× end-to-end speedup** the user can feel (7.5 s → 25 s).
- Long decode outputs dilute the AMX win. Pure keyword lookups have no embedding/prefill cost. RAG hits the sweet spot.

**Why Milvus:**  
Milvus is the leading open-source vector database for enterprise RAG stacks — recognizable to the Xeon target audience and realistic for production deployments. Its HNSW index uses Intel-optimized BLAS for distance computations, and Milvus Standalone deploys as a single Docker container alongside the existing vLLM containers. The `pymilvus` SDK mirrors the OpenAI SDK pattern already used by the LLM benchmark client, keeping the code cohesive.

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
├── Dockerfile.cpu.amx          # vLLM CPU image — AMX + AVX-512 BF16 enabled
├── Dockerfile.cpu.no-amx       # vLLM CPU image — AVX-512 BF16 only, AMX disabled
├── build_docker_amx.sh         # Build the AMX image
├── build_docker_no_amx.sh      # Build the no-AMX image
│
├── ── Track A: LLM Inference Demo ─────────────────────────────────────────────
├── start_amx_containers.sh     # Launch LLM containers (ports 8000/8001) and wait for health
├── stop_amx_containers.sh      # Stop LLM containers
├── restart_amx_containers.sh   # Restart LLM containers and wait for health
├── show_docker.sh              # List Docker images and running containers
├── check_vllm_services.sh      # Poll health endpoints until both are ready
├── test_vLLM.sh                # Quick smoke test — sends "Hello" to each container
├── benchmark_amx.sh            # Automated benchmark runner (wraps query_vllm_amx.py)
├── query_vllm_amx.py           # Python benchmark client — TTFT, prefill tok/s, decode tok/s
├── PWI-Flask-2vLLM.py          # Flask demo v1 — select service & question in browser
├── PWI-Flask-2vLLM-v2.py       # Flask demo v2 — multi-run, cache-busted, richer metrics
│
└── ── Track B: RAG + VectorDB Demo ────────────────────────────────────────────
    ├── docker-compose.rag.yml  # Milvus standalone + all 4 vLLM containers
    ├── start_rag_demo.sh       # One-command start for all RAG demo services
    ├── stop_rag_demo.sh        # Stop all RAG demo services
    ├── rag_corpus.py           # Generates 100-doc synthetic corpus (5 topic clusters)
    ├── rag_index_amx.py        # Embed corpus, benchmark AMX vs no-AMX throughput, index Milvus
    ├── rag_query_amx.py        # Full RAG query: embed → search → generate (AMX vs no-AMX)
    └── PWI-Flask-RAG.py        # Browser demo: Indexing + RAG Query + LLM Inference (port 5002)
```

> **Tip:** Both tracks share the same two Docker images. Track B adds Milvus and two additional vLLM embedding containers on top of Track A.

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

| Image tag | Key build args |
|---|---|
| `vllm-cpu-amx:latest` | `VLLM_CPU_AMXBF16=1 VLLM_CPU_AVX512BF16=1 VLLM_CPU_AVX512VNNI=1` |
| `vllm-cpu-no-amx:latest` | `VLLM_CPU_AMXBF16=0 VLLM_CPU_AVX512BF16=1 VLLM_CPU_AVX512VNNI=1` |

Confirm both images exist:

```bash
bash show_docker.sh
```

---

## Step 2 — Environment Variables

```bash
export HF_TOKEN=hf_...

# RAG track only — override model defaults if needed
export VLLM_LLM_MODEL=ibm-granite/granite-3.3-8b-instruct   # default
export VLLM_EMBED_MODEL=BAAI/bge-m3                          # default
```

The LLM model (`ibm-granite/granite-3.3-8b-instruct`, BF16, ~16 GB) and embedding model (`BAAI/bge-m3`, BF16, ~1.5 GB) weights are cached in `~/.cache/huggingface` and mounted into each container.

---

## Track A — LLM Inference Demo

### Step 3A — Start the LLM Containers

```bash
bash start_amx_containers.sh
```

Launches two detached containers and waits for both `/health` endpoints:

| Container | Port | ISA |
|---|---|---|
| `vllm-amx` | 8000 | `AVX512_CORE_AMX` |
| `vllm-no-amx` | 8001 | `AVX512_CORE_BF16` |

The AMX container binds OMP threads to cores 0–19; the no-AMX container to cores 20–39. Adjust `VLLM_CPU_OMP_THREADS_BIND` in the script to match your socket/NUMA topology.

### Step 4A — Verify the Containers

```bash
bash check_vllm_services.sh          # poll /health endpoints
bash test_vLLM.sh                    # smoke test — sends "Hello"
bash show_docker.sh                  # list running containers
```

#### Verify oneDNN kernel dispatch with `DNNL_VERBOSE`

To confirm the AMX container is dispatching AMX kernels, restart it with `DNNL_VERBOSE=1`:

```bash
docker stop vllm-amx

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

While the container handles a request, lines like the following confirm AMX kernels are active:

```
dnnl_verbose,exec,cpu,matmul,...,avx512_core_amx,...
```

The no-AMX container shows `avx512_core_bf16` instead — confirming AMX is correctly disabled:

```
dnnl_verbose,exec,cpu,matmul,...,avx512_core_bf16,...
```

Set `DNNL_VERBOSE=0` (the default) for normal operation — verbose logging adds overhead.

### Step 5A — Run the LLM Benchmark

```bash
bash benchmark_amx.sh
```

Runs a 2K-token prompt / 50-token answer scenario — long context so prefill dominates, short output so the AMX TTFT win carries through to a meaningful end-to-end speedup.

#### Manual runs with `query_vllm_amx.py`

```bash
pip install openai rich

# Default prompt, 3 runs
python3 query_vllm_amx.py

# Custom prompt, 5 runs, 1-token output (pure prefill, maximum AMX signal)
python3 query_vllm_amx.py \
  --prompt "Explain how AMX tile instructions accelerate transformer prefill." \
  --runs 5 --max-tokens 1 --cooldown 3

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

### Step 6A — LLM Interactive Demo (Flask)

```bash
python3 PWI-Flask-2vLLM-v2.py
# open http://localhost:5001
```

Pick a long-context question, watch both containers stream their response side by side, and compare live TTFT and prefill tok/s metrics.

---

## Track B — RAG + VectorDB Demo

### Pipeline Architecture

```
rag_corpus.py → 100 synthetic documents (5 topic clusters)
                        │
                        ▼
        ┌───────────────┴───────────────┐
        │  INDEXING (one-time setup)    │
        │                               │
        │  vllm-embed-amx  :8002        │   AMX embedding
        │  vllm-embed-noamx :8003       │   no-AMX embedding
        │  (model: BAAI/bge-m3, 570M)   │
        │  ← throughput benchmark       │
        │                               │
        │  Milvus :19530                │
        │  (HNSW index, cosine sim)     │
        └───────────────────────────────┘
                        │
                        ▼ at query time
User question → embed (AMX or no-AMX) → Milvus top-k search
                        │
                        ▼
        RAG prompt = retrieved chunks + question
                        │
                        ▼
        vllm-amx :8000       vllm-noamx :8001
        (Granite 8B + AMX)   (Granite 8B, no AMX)
        ← TTFT / total time comparison
```

### AMX Benefit at Each Stage

| Pipeline stage | Metric | Expected AMX speedup |
|---|---|---|
| Document embedding (indexing) | Docs/sec, avg ms/doc | ~3–6× |
| Query embedding (per request) | Embed latency (ms) | ~3–6× |
| Milvus HNSW search | Search latency (ms) | Marginal — already fast |
| LLM TTFT | Time to first token | ~6× |
| LLM total time (50-token output) | End-to-end request time | ~3× |
| **End-to-end RAG latency** | **User-visible total** | **~3×** |

The embedding speedup compounds with the LLM prefill speedup: every round-trip in a RAG pipeline — both the encode step and the generate step — benefits from AMX.

### Service Port Map

| Container | Port | Model | ISA |
|---|---|---|---|
| `vllm-amx` | 8000 | Granite 8B (LLM) | `AVX512_CORE_AMX` |
| `vllm-no-amx` | 8001 | Granite 8B (LLM) | `AVX512_CORE_BF16` |
| `vllm-embed-amx` | 8002 | BAAI/bge-m3 (embedding) | `AVX512_CORE_AMX` |
| `vllm-embed-no-amx` | 8003 | BAAI/bge-m3 (embedding) | `AVX512_CORE_BF16` |
| `milvus-standalone` | 19530 | — | — |

> **Memory note:** The two LLM containers require ~16 GB DRAM each (8B BF16); the two embedding containers require ~1.5 GB each. Recommended minimum: **64 GB per socket** on a dual-socket system with NUMA-bound thread pinning.

### Step 3B — Start All RAG Services

```bash
bash start_rag_demo.sh
```

Starts Milvus (etcd + MinIO + milvus-standalone) and all four vLLM containers, then waits for every health endpoint. First run downloads model weights (~130 s on 1 Gbps); subsequent starts load from local NVMe cache.

To start Milvus only (e.g., while LLM containers are still loading):

```bash
bash start_rag_demo.sh --infra-only
```

### Step 4B — Install Python Dependencies

```bash
pip install openai pymilvus rich flask
```

### Step 5B — Index the Corpus

```bash
python3 rag_index_amx.py
```

This script:
1. Loads the 100-document synthetic corpus (`rag_corpus.py`)
2. Encodes all documents through the **AMX embedding endpoint** — measuring docs/sec and latency
3. Encodes the same corpus through the **no-AMX embedding endpoint** — for comparison
4. Prints an AMX vs no-AMX throughput comparison table
5. Inserts the AMX-generated embeddings into **Milvus** (HNSW index, cosine similarity)

Sample output:
```
╔══════════════════════════════════════════════════════╗
║   AMX VectorDB Indexing Benchmark                    ║
╚══════════════════════════════════════════════════════╝

Generated 100 synthetic documents in-memory

--- AMX Embedding (port 8002) ---
  100 docs | avg 45.2ms/doc | 22.1 docs/sec | errors: 0

--- No-AMX Embedding (port 8003) ---
  100 docs | avg 245.8ms/doc | 4.1 docs/sec | errors: 0

  Embedding Performance
  ┌───────────────────────────┬────────────┬────────────────┬─────────────┐
  │ Metric                    │ AMX ✅     │ No AMX (AVX-512)│ AMX Speedup │
  ├───────────────────────────┼────────────┼────────────────┼─────────────┤
  │ Avg latency / doc (ms)    │ 45.2ms     │ 245.8ms        │ 5.4x faster │
  │ Throughput (docs/sec)     │ 22.1       │ 4.1            │ 5.4x higher │
  │ Total indexing time (s)   │ 4.5s       │ 24.6s          │             │
  └───────────────────────────┴────────────┴────────────────┴─────────────┘

--- Milvus Indexing (HNSW, dim=1024) ---
  Inserted 100 documents into Milvus
  ✅ Index ready: 100 documents in 'amx_rag_demo' collection
```

#### `rag_index_amx.py` CLI options

| Option | Default | Description |
|---|---|---|
| `--amx-embed-url` | `http://localhost:8002` | AMX embedding endpoint |
| `--no-amx-embed-url` | `http://localhost:8003` | no-AMX embedding endpoint |
| `--embed-model` | `BAAI/bge-m3` | Embedding model name |
| `--milvus-host` | `localhost` | Milvus hostname |
| `--milvus-port` | `19530` | Milvus port |
| `--corpus-file` | *(in-memory)* | Path to corpus JSON |
| `--skip-no-amx` | — | Skip no-AMX benchmark, only build index |
| `--skip-milvus` | — | Benchmark embedding only, do not insert |

### Step 6B — Run a RAG Query Benchmark

```bash
python3 rag_query_amx.py \
  --question "How does Intel AMX accelerate LLM inference?" \
  --top-k 5 --max-tokens 50 --runs 3
```

Each run:
1. Embeds the question with the AMX endpoint → records embed latency
2. Searches Milvus for top-5 most relevant document chunks
3. Builds a RAG prompt (system prompt + retrieved context + question)
4. Streams the answer from the AMX LLM → records TTFT, prefill tok/s, total time
5. Repeats steps 1 and 4 with the no-AMX endpoints

Reports per-stage and end-to-end latency for both paths.

```bash
# List sample questions
python3 rag_query_amx.py --list-questions
```

#### `rag_query_amx.py` CLI options

| Option | Default | Description |
|---|---|---|
| `--question` | *(first sample)* | Question to ask |
| `--top-k` | 5 | Retrieved chunks to include in the prompt |
| `--max-tokens` | 50 | LLM output length (50 = RAG sweet spot) |
| `--runs` | 3 | Benchmark runs per path |
| `--cooldown` | 2 | Seconds between runs |
| `--skip-no-amx` | — | AMX path only |
| `--list-questions` | — | Print sample questions and exit |

### Step 7B — RAG Interactive Demo (Flask)

```bash
python3 PWI-Flask-RAG.py
# open http://localhost:5002
```

Three tabs in one UI:

| Tab | What it demonstrates |
|---|---|
| **① Embedding & Indexing** | Select a sample size → watch AMX encode the corpus faster, see docs/sec and total indexing time comparison |
| **② RAG Query** | Ask a question → see the retrieved Milvus chunks → watch AMX and no-AMX stream answers side by side with per-stage metrics |
| **③ LLM Inference** | Direct LLM comparison (no retrieval) — equivalent to the Track A Flask demo |

Key implementation notes:
- Cache busting per run (unique suffix per request defeats vLLM prefix caching)
- `stream_options: include_usage` for accurate prompt token counts
- Sequential execution (AMX first, then no-AMX) to avoid DRAM contention noise
- `max_tokens=50` default (RAG sweet spot: ~46% prefill, ~3× end-to-end speedup)

### Corpus Details

`rag_corpus.py` generates 100 self-contained technical documents — no internet access or external data required:

| Topic cluster | Docs | Content |
|---|---|---|
| `cpu-architecture` | 20 | ISA extensions, AMX tile arch, NUMA, cache hierarchy, prefetch |
| `ml-inference` | 20 | Transformers, embeddings, RAG, quantization, batching, KV cache |
| `llm-serving` | 20 | vLLM, OpenAI API, Granite/Llama, serving patterns, latency SLOs |
| `data-center` | 20 | Docker, Kubernetes, Milvus, TCO, networking, observability |
| `intel-amx` | 20 | oneDNN, TDPBF16PS, benchmarking, ISA dispatch, Sapphire Rapids |

Documents are 200–400 words each — typical enterprise RAG retrieval chunk size. Topics are chosen so nearly every question in the demo retrieves highly relevant, grounded context from multiple clusters.

### Stop the RAG Stack

```bash
bash stop_rag_demo.sh
```

---

## Benchmark Results

Full results and analysis are in [`perftests.md`](perftests.md). Key highlights:

### LLM Prefill — Context length sweep (`--max-tokens 1`, pure prefill, cache-busted)

| Prompt tokens | AMX TTFT | No-AMX TTFT | Speedup | AMX Prefill tok/s | No-AMX Prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 550 | 777 ms | 4,680 ms | **6.0×** | 708 | 118 |
| 1,032 | 1,317 ms | 8,347 ms | **6.3×** | 784 | 124 |
| 1,877 | 2,379 ms | 14,646 ms | **6.2×** | 789 | 128 |
| 4,393 | 5,843 ms | 34,857 ms | **6.0×** | 752 | 126 |
| 8,343 | 12,389 ms | 67,175 ms | **5.4×** | 673 | 124 |

### LLM End-to-end — 2,666-token prompt, varying output length

| Scenario | Output tokens | TTFT speedup | Total time speedup |
|---|---:|---:|---:|
| Pure prefill benchmark | 1 | **6.1×** | **6.1×** |
| Summarization / RAG (sweet spot) | 50 | **6.1×** | **3.3×** |
| Detailed answer | 200 | **6.2×** | **1.9×** |

### Why AMX helps prefill but not decode

- **Prefill** — the entire input prompt is processed in parallel as large matrix multiplications. AMX 16×16 BF16 tile-MACC instructions directly accelerate this compute-bound GEMM phase (~6× speedup).
- **Decode** — one token at a time, loading full weight matrices each step. Memory-bandwidth bound; AMX provides no arithmetic benefit. Both containers are equally constrained by DRAM throughput.

The **50-token output / 2,600-token prompt** scenario is the most honest demo: it represents a genuine RAG or document Q&A workload and delivers a **3.3× end-to-end speedup** (7.5 s → 25 s) that users can feel — while the 6.1× TTFT difference (3.5 s → 21 s) directly translates to perceived responsiveness.

---

## Container Management

```bash
# ── Track A (LLM only) ──────────────────────────────
bash stop_amx_containers.sh
bash restart_amx_containers.sh

# ── Track B (RAG stack: Milvus + 4 vLLM containers) ─
bash stop_rag_demo.sh
bash start_rag_demo.sh

# ── Inspect running containers ───────────────────────
bash show_docker.sh
docker compose -f docker-compose.rag.yml ps
```

---

## License

See [LICENSE](LICENSE).
