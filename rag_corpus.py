#!/usr/bin/env python3
"""
rag_corpus.py
-------------
Generates a self-contained synthetic corpus of 100 technical documents
for the AMX RAG demo. Covers five topic clusters that are relevant to
the Intel Xeon / AMX target audience:

  1. CPU architecture & ISA extensions (20 docs)
  2. Machine learning inference & workloads (20 docs)
  3. LLM serving & inference engines (20 docs)
  4. Data center infrastructure (20 docs)
  5. Intel AMX & oneDNN technology (20 docs)

Each document is 200-400 words — typical RAG retrieval chunk size.

Usage:
    python3 rag_corpus.py                   # write corpus.json
    python3 rag_corpus.py --out my.json
    python3 rag_corpus.py --count           # print doc count and exit
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus definition
# Each entry: {"id": str, "title": str, "topic": str, "text": str}
# ---------------------------------------------------------------------------

_DOCS = [

    # -----------------------------------------------------------------------
    # Cluster 1 — CPU Architecture & ISA Extensions
    # -----------------------------------------------------------------------
    {
        "id": "cpu-001",
        "title": "SIMD Evolution: MMX to AMX",
        "topic": "cpu-architecture",
        "text": (
            "Single Instruction Multiple Data (SIMD) extensions have been a cornerstone of x86 performance "
            "for decades. Intel introduced MMX in 1997 with 64-bit packed integer operations. SSE expanded "
            "this to 128-bit floating-point vectors in 1999. AVX (Advanced Vector Extensions) doubled "
            "the register width to 256 bits in 2011, followed by AVX-512 in 2017 which provided 512-bit "
            "vectors and new arithmetic capabilities including BF16 dot products via VNNI. "
            "Intel Advanced Matrix Extensions (AMX) represents the next architectural leap: instead of "
            "operating on 1D vectors, AMX introduces two-dimensional tile registers (up to 16 rows × 64 "
            "bytes each) and TMUL instructions that perform a full 16×16 matrix multiply-accumulate in a "
            "single operation. This delivers up to 2048 BF16 multiply-adds per cycle per core — "
            "approximately 8× the throughput of AVX-512 BF16 VNNI for matrix-heavy workloads. AMX first "
            "appeared in Sapphire Rapids (4th Gen Intel Xeon Scalable) and is available on all subsequent "
            "Xeon Scalable processor generations."
        ),
    },
    {
        "id": "cpu-002",
        "title": "BFloat16 Numeric Format in AI Workloads",
        "topic": "cpu-architecture",
        "text": (
            "BFloat16 (BF16) is a 16-bit floating-point format that trades mantissa precision for dynamic "
            "range compared to IEEE 754 float16. It uses 1 sign bit, 8 exponent bits (same as float32), "
            "and 7 mantissa bits. Because BF16 shares the same exponent range as float32, converting "
            "between them requires only truncation — no rescaling. This property makes BF16 attractive "
            "for deep learning: gradients and activations rarely require the full 23-bit mantissa of "
            "float32, but they do need the wide dynamic range. Hardware support for BF16 appeared first "
            "in Google TPUs, then Intel Cooper Lake (3rd Gen Xeon) with AVX512_BF16, and later with "
            "full TMUL acceleration in AMX on Sapphire Rapids. Training and inference in BF16 typically "
            "achieve accuracy within 0.5% of float32 while doubling arithmetic throughput on compatible "
            "hardware. PyTorch, TensorFlow, and JAX all support automatic mixed precision with BF16."
        ),
    },
    {
        "id": "cpu-003",
        "title": "Cache Hierarchy and Memory Bandwidth in Modern Xeon",
        "topic": "cpu-architecture",
        "text": (
            "Modern Intel Xeon Scalable processors (Sapphire Rapids and later) feature a multi-level "
            "cache hierarchy optimized for throughput workloads. Each core has a 48 KB L1 data cache "
            "and 2 MB L2 cache. The last-level cache (LLC) is a unified, distributed structure — on "
            "Sapphire Rapids, up to 112.5 MB of LLC is shared across four tiles connected by a mesh "
            "interconnect. Memory bandwidth is provided by 8-channel DDR5-4800 per socket, delivering "
            "approximately 307 GB/s peak. For AI inference, DRAM bandwidth is the dominant bottleneck "
            "during the decode phase of LLM serving: each token requires loading the full weight "
            "matrices from memory. In contrast, the prefill phase is compute-bound — large GEMMs over "
            "the prompt sequence length can be sustained in cache or streamed efficiently. AMX tile "
            "operations maximize arithmetic intensity during prefill, hiding memory latency by "
            "overlapping computation with prefetch pipelines."
        ),
    },
    {
        "id": "cpu-004",
        "title": "NUMA Architecture in Multi-Socket Xeon Systems",
        "topic": "cpu-architecture",
        "text": (
            "Non-Uniform Memory Access (NUMA) is a memory architecture used in multi-socket server "
            "systems. Each CPU socket has local DRAM with low-latency access (~80 ns), while accessing "
            "remote memory through the inter-socket interconnect incurs additional latency (~120-160 ns "
            "total). Intel's Ultra Path Interconnect (UPI) links sockets at up to 16 GT/s per lane in "
            "Sapphire Rapids. For AI inference workloads, NUMA-aware process binding is critical: "
            "vLLM and other inference engines expose VLLM_CPU_OMP_THREADS_BIND to pin OpenMP threads "
            "to specific NUMA nodes. This prevents the OS scheduler from migrating threads across "
            "sockets, which would incur remote DRAM penalties and reduce effective bandwidth. "
            "numactl --membind and --cpunodebind flags, or libnuma APIs, provide fine-grained control. "
            "Optimal configuration places model weights and KV cache in local NUMA memory."
        ),
    },
    {
        "id": "cpu-005",
        "title": "OpenMP and Intel Threading for AI Workloads",
        "topic": "cpu-architecture",
        "text": (
            "OpenMP is a portable shared-memory parallelism API widely used in scientific computing "
            "and AI libraries. Intel's implementation, libiomp5 (Intel OpenMP Runtime), provides "
            "performance advantages over GNU libgomp for workloads with fine-grained parallelism: "
            "it uses a spin-wait strategy with adaptive back-off that reduces thread synchronization "
            "overhead. For vLLM CPU inference, libiomp5 is loaded via LD_PRELOAD before GNU OpenMP to "
            "prevent OMP library conflicts that would raise a runtime error. Thread affinity is "
            "controlled through OMP_NUM_THREADS and GOMP_CPU_AFFINITY or Intel's KMP_AFFINITY "
            "environment variable. For AMX workloads, pinning threads to physical cores (not "
            "hyperthreads) generally yields better performance since AMX tile registers are per-core "
            "resources. TCMalloc (libtcmalloc_minimal) complements this by reducing heap contention "
            "across the parallel allocator."
        ),
    },
    {
        "id": "cpu-006",
        "title": "Instruction-Level Parallelism in Modern Out-of-Order CPUs",
        "topic": "cpu-architecture",
        "text": (
            "Modern CPUs exploit instruction-level parallelism (ILP) through out-of-order execution, "
            "superscalar dispatch, and speculative execution. Sapphire Rapids cores can dispatch up to "
            "6 micro-operations per cycle across multiple execution ports. The TMUL unit for AMX "
            "operations occupies dedicated execution ports (port 11 on Sapphire Rapids) and can "
            "sustain one 16×16 BF16 tile multiply-accumulate per cycle when the pipeline is full. "
            "Achieving peak AMX throughput requires software to issue TMUL instructions without "
            "data hazards — libraries like oneDNN and Intel MKL are hand-tuned to schedule AMX "
            "micro-kernels with optimal prefetch distances, register blocking, and loop nest "
            "ordering. The key insight is that AMX's large tile size amortizes instruction fetch "
            "overhead across 256 multiply-add operations per instruction, making it far more "
            "efficient than scalar or vector code for matrix workloads."
        ),
    },
    {
        "id": "cpu-007",
        "title": "AVX-512 vs AMX: When to Use Each",
        "topic": "cpu-architecture",
        "text": (
            "AVX-512 and AMX are complementary ISA extensions targeting different computational "
            "patterns. AVX-512 excels at data-parallel element-wise operations: vector normalization, "
            "activation functions (ReLU, GELU), elementwise add/multiply, and gather/scatter memory "
            "operations. Its 512-bit registers process 16 floats or 32 BF16 values simultaneously. "
            "AMX targets matrix multiply specifically: its tile instructions operate on 2D blocks "
            "of up to 16×64 bytes, performing an entire 16×16 BF16 GEMM tile in a single instruction. "
            "In transformer inference, the dominant operations are linear projections (attention Q/K/V, "
            "feed-forward layers) — all dense GEMMs — which map directly to AMX. Softmax, layer norm, "
            "and activation functions use AVX-512. oneDNN automatically selects the optimal kernel "
            "for each operation based on the DNNL_MAX_CPU_ISA setting, dispatching TMUL kernels when "
            "AMX is available and AVX-512 kernels otherwise."
        ),
    },
    {
        "id": "cpu-008",
        "title": "Power and Thermal Management in AI Server CPUs",
        "topic": "cpu-architecture",
        "text": (
            "AI inference workloads place sustained, high-utilization demands on CPU power delivery "
            "systems. Intel Xeon Scalable processors implement multiple power management features: "
            "Turbo Boost Max Technology 3.0 identifies the two highest-performing cores per die for "
            "priority scheduling; Speed Select Technology (SST) allows operators to configure base "
            "frequency, core count, and power limits independently. For AMX workloads, which sustain "
            "peak FLOP utilization, thermal design must account for the TDP envelope. Sapphire Rapids "
            "processors have TDPs from 250 W to 350 W depending on SKU. Air cooling with high-static-"
            "pressure fans or direct liquid cooling (DLC) is typically required for sustained "
            "inference loads. Power capping via RAPL (Running Average Power Limit) registers or "
            "BMC-level policies can be used to cap power at the cost of reduced throughput, which "
            "is sometimes preferable in shared data center environments."
        ),
    },
    {
        "id": "cpu-009",
        "title": "Prefetch Strategies for High-Performance GEMM",
        "topic": "cpu-architecture",
        "text": (
            "Matrix multiplication performance is highly sensitive to memory access patterns. "
            "L2 and LLC hardware prefetchers detect sequential and strided access patterns and "
            "issue early DRAM reads, hiding memory latency. For AMX kernels, software-initiated "
            "prefetch instructions (PREFETCHT1/PREFETCHT2) are inserted by the compiler or "
            "library to bring the next tile of matrix data into L2 cache while the current TMUL "
            "instruction is executing. The critical loop in GEMM — the K-dimension reduction over "
            "the inner product — should be structured so that matrix A tiles are reused across "
            "multiple B tiles (register blocking) to minimize memory bandwidth consumption. "
            "oneDNN's AMX kernels implement this with a brgemm (batch-reduce GEMM) micro-kernel "
            "that explicitly prefetches the next batch of A/B tiles while accumulating the current "
            "batch, keeping the AMX tile multiply units saturated."
        ),
    },
    {
        "id": "cpu-010",
        "title": "Hardware Counters and CPU Performance Monitoring",
        "topic": "cpu-architecture",
        "text": (
            "Performance Monitoring Units (PMUs) in Intel CPUs expose hardware counters that allow "
            "precise measurement of microarchitectural events: cache misses, branch mispredictions, "
            "memory bandwidth, and instruction throughput. For AMX workloads, relevant counters "
            "include AMX_OPS_RETIRED (TMUL instructions retired), FP_ARITH_INST_RETIRED for "
            "floating-point throughput, and OFFCORE_REQUESTS for memory traffic. The perf tool "
            "(Linux), Intel VTune Profiler, and Intel Advisor provide high-level analysis on top "
            "of these PMU counters. For demonstrating AMX vs no-AMX, TMUL instruction throughput "
            "and LLC miss rate are the most diagnostic: AMX should show high TMUL op counts and "
            "low miss rates (good cache reuse), while AVX-512 code shows no TMUL activity and "
            "higher cache pressure for the same matrix sizes."
        ),
    },
    {
        "id": "cpu-011",
        "title": "Xeon Scalable Processor Generations Overview",
        "topic": "cpu-architecture",
        "text": (
            "Intel Xeon Scalable processors have evolved through multiple generations. 1st Gen "
            "(Skylake-SP, 2017): up to 28 cores, AVX-512 launch, UPI interconnect. 2nd Gen "
            "(Cascade Lake, 2019): DL Boost with VNNI for INT8 inference acceleration. 3rd Gen "
            "(Ice Lake, 2021): up to 40 cores, PCIe 4.0, AVX-512 BF16. 4th Gen (Sapphire Rapids, "
            "2023): tile-based die layout, AMX launch, HBM2e option (Max Series), DDR5 and PCIe 5.0. "
            "5th Gen (Emerald Rapids, 2024): same socket as Sapphire Rapids (LGA4677), improved "
            "LLC, up to 64 cores, refined AMX scheduler. 6th Gen (Granite Rapids, 2024): new tile "
            "architecture, wider AMX, up to 128 cores on P-core tiles. AMX is available from "
            "Sapphire Rapids onward and remains backward compatible — software compiled for AMX "
            "will run (and benefit) on all subsequent generations."
        ),
    },
    {
        "id": "cpu-012",
        "title": "Hyperthreading and Its Effect on AI Inference",
        "topic": "cpu-architecture",
        "text": (
            "Simultaneous Multithreading (SMT), marketed by Intel as Hyperthreading (HT), allows "
            "two logical threads to share one physical core's execution resources. For AI inference, "
            "HT impact is workload-dependent. During prefill (GEMM-heavy, compute-bound), the AMX "
            "tile units are dedicated per-core resources — the two SMT threads compete for TMUL "
            "dispatch slots, which can reduce throughput compared to running one thread per core. "
            "During decode (memory-bandwidth bound), a second HT thread can improve CPU utilization "
            "by issuing additional memory requests while the first thread waits for data. Best "
            "practice for single-tenant inference: disable HT or bind to physical cores only. "
            "VLLM_CPU_OMP_THREADS_BIND should specify physical core IDs, not logical CPU IDs. "
            "lscpu -e shows physical vs logical CPU mapping; cpupower or BIOS settings control HT."
        ),
    },
    {
        "id": "cpu-013",
        "title": "Intel AMX Tile Register Architecture",
        "topic": "cpu-architecture",
        "text": (
            "AMX introduces eight 2D tile registers, named tmm0 through tmm7. Each tile is up to "
            "16 rows × 64 bytes (1024 bytes total). The tile configuration — including the actual "
            "number of rows and columns used — is specified by a TILECONFIG instruction that loads "
            "a 64-byte configuration structure. TILESTOREDX and TILELOADDT1 move data between tile "
            "registers and memory. The TDPBF16PS instruction computes a BF16 tile matrix multiply-"
            "accumulate, accumulating results into a float32 tile. The operation is: "
            "C[i,j] += sum_k(A[i,k] * B[k,j]) for all i,j in the tile. This single instruction "
            "computes 16×16×32 = 8192 scalar multiply-adds (or 2048 BF16 MAC pairs in one pass). "
            "The tile registers are preserved across context switches by the OS kernel's XSAVE/XRSTOR "
            "state management, which has been updated to handle the large tile state."
        ),
    },
    {
        "id": "cpu-014",
        "title": "ECC Memory and RAS Features in Xeon",
        "topic": "cpu-architecture",
        "text": (
            "Error Correcting Code (ECC) memory is a standard feature of Xeon platforms, detecting "
            "and correcting single-bit memory errors in real time. Multi-bit errors trigger a "
            "machine check exception (MCE). Intel's Reliability, Availability, and Serviceability "
            "(RAS) features extend this with Memory Patrol Scrubbing (periodically reads and "
            "corrects memory in the background), Demand Scrubbing (on first access), and Machine "
            "Check Architecture (MCA) for structured error reporting. For AI inference deployments "
            "running 24×7, ECC is essential: DRAM bit error rates at elevated temperatures can "
            "corrupt model weights silently in non-ECC systems. The large model weight buffers in "
            "LLM serving (~16 GB for an 8B BF16 model) increase the probability of encountering "
            "a random bit flip over long operational periods."
        ),
    },
    {
        "id": "cpu-015",
        "title": "PCIe 5.0 and CXL in AI Infrastructure",
        "topic": "cpu-architecture",
        "text": (
            "PCIe 5.0, introduced with 4th Gen Xeon Scalable, doubles bandwidth to 64 GT/s per "
            "lane compared to PCIe 4.0's 32 GT/s. A ×16 slot delivers 128 GB/s bidirectional, "
            "enabling high-bandwidth NVMe SSDs and accelerators. Compute Express Link (CXL) is "
            "a cache-coherent interconnect built on PCIe physical layers that enables memory "
            "pooling and disaggregation. CXL 1.1 (on Sapphire Rapids) supports CXL.mem for "
            "memory expansion: CXL-attached DRAM appears as additional NUMA nodes accessible at "
            "near-DRAM latency. This is relevant for LLM inference where KV cache memory can "
            "exceed local DRAM capacity. CXL memory expanders allow KV cache offload without "
            "the high latency of NVMe swap. CXL 2.0 adds switch-level pooling for multi-host "
            "memory sharing."
        ),
    },
    {
        "id": "cpu-016",
        "title": "Compiler Optimizations for AMX: Auto-Vectorization and Loop Transforms",
        "topic": "cpu-architecture",
        "text": (
            "Modern compilers (GCC 13+, LLVM/Clang 16+, Intel ICX) can auto-vectorize loops to "
            "use AVX-512 and, in limited cases, generate AMX code for matrix patterns. However, "
            "practical AMX performance typically requires hand-written or library-provided kernels. "
            "The key compiler flags for AMX are -mamx-bf16 -mamx-tile -mamx-int8 (GCC/Clang) or "
            "/arch:CORE-AMX (MSVC). At runtime, dispatch is gated by CPUID leaf 7, sub-leaf 0, "
            "bit 24 (AMX-BF16). Runtime dispatch libraries like x86-simd-sort use CPU feature "
            "detection to select AMX or AVX-512 paths. Intel's oneAPI DPC++/C++ Compiler and "
            "MKL provide the most mature AMX code generation, with hand-tuned brgemm (batch-"
            "reduce GEMM) kernels that saturate the TMUL units."
        ),
    },
    {
        "id": "cpu-017",
        "title": "Sparse Computation and Structured Pruning on CPU",
        "topic": "cpu-architecture",
        "text": (
            "Model compression via pruning reduces parameter count and can improve inference "
            "speed. Unstructured pruning (setting individual weights to zero) creates irregular "
            "sparsity that is difficult to exploit with SIMD hardware. Structured pruning removes "
            "entire rows, columns, or blocks, producing dense sub-matrices that AMX can accelerate "
            "directly. Intel's Neural Compressor supports structured pruning and quantization for "
            "PyTorch models targeting Xeon deployment. For 2:4 structured sparsity (NVIDIA's "
            "format), hardware support exists on newer NVIDIA GPUs but not natively in AMX. "
            "Block-wise sparsity (8×8 or 16×16 zero blocks) maps well to AMX tile boundaries "
            "and can skip TDPBF16PS execution for zero tiles. Realistic sparsity ratios of 50-80% "
            "with block structure can achieve 1.5-2× additional speedup over dense AMX inference."
        ),
    },
    {
        "id": "cpu-018",
        "title": "Memory Interleaving and Channel Configuration",
        "topic": "cpu-architecture",
        "text": (
            "DDR5 memory in 4th Gen Xeon supports up to 8 memory channels per socket. Populating "
            "all channels and enabling memory interleaving in the BIOS distributes memory accesses "
            "across all channels simultaneously, maximizing aggregate bandwidth. Sub-NUMA Clustering "
            "(SNC) modes (SNC2, SNC4) partition the CPU die into NUMA sub-domains, each with a "
            "subset of channels and LLC, reducing average NUMA latency. For LLM inference, SNC4 "
            "with process pinning can improve decode throughput by ensuring weight matrix fetches "
            "hit only local channels. However, SNC increases NUMA complexity and requires careful "
            "application configuration. The trade-off: lower latency per NUMA node vs higher "
            "programming complexity. Most inference frameworks default to flat (non-SNC) mode "
            "with explicit NUMA binding via numactl or environment variables."
        ),
    },
    {
        "id": "cpu-019",
        "title": "AMX INT8 for Quantized Inference",
        "topic": "cpu-architecture",
        "text": (
            "In addition to BF16 tile multiply (TDPBF16PS), AMX supports INT8 tile operations "
            "via TDPBSSD (signed 8-bit × signed 8-bit → int32 accumulate) and TDPBUSD (unsigned "
            "× signed). INT8 quantization with AMX can achieve 2× higher arithmetic throughput "
            "than BF16 AMX (since two INT8 values pack into the same space as one BF16). Post-"
            "training quantization (PTQ) converts BF16 model weights and activations to INT8 with "
            "minimal accuracy loss using calibration data. Tools: Intel Neural Compressor, "
            "llm.int8() (bitsandbytes), smooth quantization (SmoothQuant). For embedding models "
            "like bge-m3, INT8 AMX inference can deliver retrieval quality within 0.2% of BF16 "
            "while halving latency. vLLM supports INT8 quantization via --quantization bitsandbytes "
            "or --quantization fp8."
        ),
    },
    {
        "id": "cpu-020",
        "title": "ISA Feature Detection and Runtime CPU Dispatch",
        "topic": "cpu-architecture",
        "text": (
            "Runtime CPU feature detection allows a single binary to use optimal instructions on "
            "any CPU without recompilation. The CPUID instruction reports supported features: "
            "leaf 7.0 ECX bit 5 = AVX-512 foundation; leaf 7.1 EDX bit 22 = AMX-BF16. Libraries "
            "query CPUID at initialization and select code paths accordingly. oneDNN uses "
            "DNNL_MAX_CPU_ISA to cap the ISA level, which allows controlled comparison benchmarks: "
            "setting DNNL_MAX_CPU_ISA=AVX512_CORE_BF16 disables AMX dispatch even on AMX-capable "
            "hardware, running the AVX-512 fallback instead. This is exactly how the AMX demo "
            "works: both containers run on the same physical CPU, but one constrains oneDNN to "
            "BF16 kernels and the other allows full AMX dispatch, measuring the speedup from "
            "AMX tile units on identical hardware."
        ),
    },

    # -----------------------------------------------------------------------
    # Cluster 2 — Machine Learning Inference & Workloads
    # -----------------------------------------------------------------------
    {
        "id": "ml-001",
        "title": "Transformer Architecture and Attention Mechanism",
        "topic": "ml-inference",
        "text": (
            "The transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., "
            "2017), is the foundation of modern LLMs. Each transformer layer consists of multi-head "
            "self-attention followed by a position-wise feed-forward network. In self-attention, "
            "the input sequence X is projected into queries Q, keys K, and values V via learned "
            "weight matrices WQ, WK, WV. Attention scores are computed as softmax(QK^T / sqrt(d_k)) "
            "then multiplied by V. Multi-head attention runs H parallel attention heads with "
            "smaller dimension d_k = d_model / H. The dominant compute operations are the linear "
            "projections — GEMMs of shape [batch × seq_len × d_model] × [d_model × d_k] — making "
            "transformers ideal for AMX acceleration during the prefill phase when the full sequence "
            "is processed at once."
        ),
    },
    {
        "id": "ml-002",
        "title": "Embedding Models and Semantic Vector Search",
        "topic": "ml-inference",
        "text": (
            "Embedding models convert text into dense vector representations that capture semantic "
            "meaning. Architecturally, they are transformer encoders (BERT-style or decoder-only "
            "with mean pooling) fine-tuned on contrastive tasks such as natural language inference "
            "or sentence pair similarity. Popular models: sentence-transformers/all-MiniLM-L6-v2 "
            "(22M params, 384-dim), BAAI/bge-m3 (570M, 1024-dim, multilingual), "
            "intfloat/e5-mistral-7b-instruct (7B, 4096-dim). Embedding quality is measured by "
            "MTEB benchmark scores. For RAG applications, embedding model choice involves a "
            "trade-off: smaller models are faster to index and query but may miss semantic nuances; "
            "larger models capture richer representations but at higher latency. BAAI/bge-m3 offers "
            "an excellent quality/performance balance and supports dense, sparse, and ColBERT-style "
            "multi-vector retrieval in a single model."
        ),
    },
    {
        "id": "ml-003",
        "title": "Retrieval-Augmented Generation (RAG) Architecture",
        "topic": "ml-inference",
        "text": (
            "Retrieval-Augmented Generation (RAG) augments LLM responses with relevant external "
            "knowledge retrieved at inference time. The pipeline has three stages: (1) Indexing — "
            "documents are split into chunks, encoded by an embedding model, and stored in a vector "
            "database; (2) Retrieval — the user query is encoded by the same embedding model, and "
            "the vector DB returns the top-k most similar document chunks via approximate nearest "
            "neighbor (ANN) search; (3) Generation — the retrieved chunks are concatenated with the "
            "query into a context-rich prompt sent to the LLM. RAG addresses key LLM limitations: "
            "knowledge cutoff (retrieved documents can be updated without retraining), hallucination "
            "(grounded answers cite specific sources), and context window limits (only relevant chunks "
            "are included). Enterprise RAG stacks typically combine Milvus or pgvector for retrieval, "
            "a hosted or on-premise LLM, and an orchestration layer (LangChain, LlamaIndex)."
        ),
    },
    {
        "id": "ml-004",
        "title": "KV Cache and Memory Management in LLM Inference",
        "topic": "ml-inference",
        "text": (
            "During LLM inference, the key (K) and value (V) projections computed for all previous "
            "tokens must be stored and reused in each autoregressive decode step — this is the KV "
            "cache. KV cache size grows linearly with sequence length: for a 8B-parameter model "
            "with 32 layers, hidden size 4096, and BF16 precision, a single token requires "
            "32 × 2 × 4096 × 2 bytes = 512 KB. A 2,000-token context window requires ~1 GB KV "
            "cache per request. vLLM manages KV cache through PagedAttention — a virtual memory "
            "system that allocates cache in fixed-size blocks (pages), allowing fragmentation-free "
            "sharing of KV pages across requests with the same prefix (prefix caching). "
            "VLLM_CPU_KVCACHE_SPACE controls the maximum KV cache size in GB. Large KV caches "
            "enable longer contexts but reduce available memory for model weights and other state."
        ),
    },
    {
        "id": "ml-005",
        "title": "Batch Processing and Continuous Batching in LLM Serving",
        "topic": "ml-inference",
        "text": (
            "Batching multiple inference requests together increases arithmetic intensity, allowing "
            "hardware to amortize the cost of loading weight matrices across multiple requests. "
            "Static batching groups requests at scheduler boundaries and processes them together — "
            "simple but wastes capacity when requests have different lengths. Continuous batching "
            "(also called dynamic batching or iteration-level scheduling) inserts new requests into "
            "a running batch as soon as a slot becomes available, improving GPU/CPU utilization by "
            "2-4× in production. vLLM implements continuous batching via its scheduler. For CPU "
            "inference with AMX, batching is particularly valuable: a batch of B requests with "
            "prompt length L performs GEMMs of shape [B×L × d_model], increasing the GEMM "
            "dimension that AMX processes — larger tiles = better AMX utilization."
        ),
    },
    {
        "id": "ml-006",
        "title": "Quantization Methods for LLM Inference",
        "topic": "ml-inference",
        "text": (
            "Quantization reduces model precision to lower bit widths, trading a small accuracy "
            "reduction for large improvements in memory footprint, bandwidth, and compute speed. "
            "Post-training quantization (PTQ) requires no fine-tuning: GPTQ applies per-channel "
            "weight quantization with layer-wise reconstruction; AWQ (Activation-aware Weight "
            "Quantization) uses activation statistics to identify salient weights; SmoothQuant "
            "migrates quantization difficulty from activations to weights. Quantization-aware "
            "training (QAT) fine-tunes with fake quantization operations for better accuracy. "
            "On Intel Xeon with AMX: BF16 (16-bit) is natively accelerated by TMUL and provides "
            "the best accuracy. INT8 via AMX-INT8 doubles throughput vs BF16 with ~0.5-1% "
            "accuracy loss. INT4 requires software dequantization before TMUL and shows 1.5-2× "
            "improvement over INT8 AMX in practice."
        ),
    },
    {
        "id": "ml-007",
        "title": "Speculative Decoding to Accelerate LLM Generation",
        "topic": "ml-inference",
        "text": (
            "Speculative decoding accelerates autoregressive generation by using a small draft "
            "model to propose K candidate tokens in parallel, then verifying them with the large "
            "target model in a single forward pass. Accepted tokens advance the sequence; the "
            "first rejected token is replaced and generation continues. Wall-clock speedup is "
            "typically 2-3× when draft acceptance rate is high (common for predictable text). "
            "The verification step is a prefill over K+1 tokens — compute-bound and directly "
            "accelerated by AMX. Draft generation is memory-bandwidth bound (single-token decode). "
            "On CPU, speculative decoding is particularly attractive: the verification prefill "
            "shows the full AMX speedup, and the draft model's small size fits in LLC for fast "
            "access. Eagle, Medusa, and EAGLE-2 are variants that use feature-level or tree-based "
            "draft prediction to improve acceptance rates."
        ),
    },
    {
        "id": "ml-008",
        "title": "Attention Variants: Multi-Query, Grouped-Query, Flash Attention",
        "topic": "ml-inference",
        "text": (
            "Standard multi-head attention (MHA) uses separate K and V heads per Q head, leading "
            "to KV cache growth proportional to the number of heads. Multi-Query Attention (MQA) "
            "shares a single K and V head across all Q heads, reducing KV cache size by H× at "
            "the cost of some quality degradation. Grouped-Query Attention (GQA) is a compromise: "
            "G groups share K/V heads (G = 1 is MQA; G = H is MHA). Granite-3.3-8B uses GQA with "
            "32 Q heads and 8 KV heads, reducing KV cache 4× vs MHA. Flash Attention is an "
            "IO-aware algorithm that fuses the attention softmax computation to avoid writing "
            "the full attention matrix to HBM, reducing memory traffic. On CPU, vLLM implements "
            "a Flash Attention variant for the prefill phase that processes attention blocks "
            "without materializing the full N×N attention matrix."
        ),
    },
    {
        "id": "ml-009",
        "title": "Vector Embeddings: Distance Metrics and Index Types",
        "topic": "ml-inference",
        "text": (
            "Vector similarity search supports multiple distance metrics. Cosine similarity "
            "measures the angle between vectors — preferred for embedding models normalized to "
            "unit length. L2 (Euclidean) distance measures geometric separation — equivalent to "
            "cosine for unit-norm vectors. Inner product (IP) computes dot product — used when "
            "embeddings are not normalized. Vector index types balance recall vs latency: "
            "FLAT (exhaustive brute force) gives 100% recall but O(N) query time — only practical "
            "for small collections. IVF_FLAT partitions vectors into Voronoi cells (nlist), "
            "searching only the nearest nprobe cells — sub-linear query time with 95-99% recall "
            "for moderate nprobe. HNSW builds a hierarchical navigable small world graph — "
            "logarithmic query time, high recall, high memory overhead. Milvus supports all "
            "three plus SCANN, DISKANN, and GPU-accelerated RAFT indices."
        ),
    },
    {
        "id": "ml-010",
        "title": "Reranking and Hybrid Search in RAG Pipelines",
        "topic": "ml-inference",
        "text": (
            "A two-stage retrieval pipeline improves RAG accuracy: first-stage ANN search retrieves "
            "top-100 candidate chunks efficiently; second-stage reranking scores them with a "
            "more expensive cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-12-v2) "
            "and returns the top-5. Cross-encoders process (query, document) pairs jointly, "
            "capturing interaction features impossible in bi-encoder embedding models. They are "
            "slower (O(k) inference calls vs 1) but provide substantially better ranking quality. "
            "Hybrid search combines dense vector search with sparse BM25 keyword search using "
            "Reciprocal Rank Fusion (RRF) or Convex Combination to merge rankings. BAAI/bge-m3 "
            "uniquely provides all three: dense embeddings, sparse (learned SPLADE-style), and "
            "multi-vector (ColBERT-style) representations in a single model forward pass."
        ),
    },
    {
        "id": "ml-011",
        "title": "Model Context Length and Long-Document RAG",
        "topic": "ml-inference",
        "text": (
            "Transformer context length determines how much text the LLM can attend to in a single "
            "forward pass. Older models (GPT-2, BERT) used 512-1024 tokens. Modern LLMs support "
            "much longer contexts: Granite-3.3-8B supports 131,072 tokens; Llama 3.1 supports "
            "128,000; Claude 3.5 supports 200,000. Long context enables direct document Q&A but "
            "increases prefill time quadratically in attention (O(n²) for standard attention) or "
            "linearly with Flash Attention. AMX accelerates the linear projections in prefill "
            "regardless of sequence length. For RAG, long context allows including more retrieved "
            "chunks, improving coverage. However, 'lost in the middle' research shows LLM recall "
            "degrades for information in the middle of long contexts — optimal RAG inserts the most "
            "relevant chunk at the beginning or end of the context window."
        ),
    },
    {
        "id": "ml-012",
        "title": "Fine-tuning LLMs for Domain Adaptation",
        "topic": "ml-inference",
        "text": (
            "Fine-tuning adapts a pre-trained LLM to a specific domain or task. Full fine-tuning "
            "updates all model parameters — expensive (8B model requires 80+ GB GPU memory with "
            "Adam optimizer states). Parameter-Efficient Fine-Tuning (PEFT) methods reduce this: "
            "LoRA (Low-Rank Adaptation) adds trainable low-rank matrix pairs to each layer, "
            "updating only ~0.1% of parameters. QLoRA combines LoRA with 4-bit base model "
            "quantization. Intel's Gaudi accelerators and Xeon with AMX support LoRA fine-tuning "
            "through Hugging Face PEFT and Intel Extension for PyTorch (IPEX). Instruction tuning "
            "on domain-specific Q&A pairs improves RAG accuracy by teaching the model to cite "
            "sources, acknowledge uncertainty, and format answers for the target use case."
        ),
    },
    {
        "id": "ml-013",
        "title": "Benchmarks for LLM Inference Performance",
        "topic": "ml-inference",
        "text": (
            "Standard metrics for LLM inference performance evaluation include: Time To First "
            "Token (TTFT) — latency from request submission to first output token, measures "
            "prefill speed; Time Per Output Token (TPOT) — average time between output tokens, "
            "measures decode speed; Tokens Per Second (TPS) — overall generation throughput; "
            "Prefill Throughput (tok/s) — input tokens processed per second (= prompt_tokens/TTFT). "
            "For comparing AMX vs no-AMX, TTFT and Prefill Throughput are the primary metrics "
            "since both are dominated by prefill GEMM performance. Total time and TPS are decode-"
            "dominated for long outputs and show smaller differences. Benchmark tools: vLLM's "
            "built-in benchmark_serving.py, llmperf, and custom clients like query_vllm_amx.py. "
            "Always cache-bust between runs to prevent vLLM prefix caching from masking compute."
        ),
    },
    {
        "id": "ml-014",
        "title": "Tokenization and Its Impact on Prompt Length",
        "topic": "ml-inference",
        "text": (
            "Tokenizers convert raw text into integer token IDs before LLM inference. The "
            "tokenization strategy affects prompt length, and therefore prefill time. Byte-Pair "
            "Encoding (BPE) and SentencePiece (used by Llama, Granite) encode common subwords "
            "as single tokens — 'inference' might be a single token while 'antidisestablishment' "
            "is split into 5-7 tokens. English text averages ~1.3 characters per token for modern "
            "BPE tokenizers. Technical content with identifiers, numbers, and code averages ~1.5-2 "
            "characters per token. For AMX demos, prompt token count directly scales TTFT: a 2×-"
            "longer prompt means 2× more prefill compute and 2× longer TTFT for both AMX and "
            "no-AMX containers — but the 6× AMX speedup remains constant, so the absolute time "
            "saved grows proportionally."
        ),
    },
    {
        "id": "ml-015",
        "title": "ONNX Runtime and OpenVINO for CPU Inference",
        "topic": "ml-inference",
        "text": (
            "ONNX Runtime (ORT) and OpenVINO are inference engines that compete with vLLM for "
            "CPU LLM deployment. ORT supports CPU execution providers including the DirectML EP "
            "and CPU EP with AMX acceleration via the oneDNN execution provider. OpenVINO is "
            "Intel's dedicated inference engine, supporting INT8 and BF16 quantization, model "
            "optimization (constant folding, op fusion), and hardware-aware graph compilation. "
            "OpenVINO 2024 includes LLM-specific optimizations: KV cache quantization, speculative "
            "decoding, and GenAI C++ samples. For AMX utilization, OpenVINO automatically leverages "
            "AMX kernels when running on Sapphire Rapids and later without manual ISA flags — "
            "the runtime detects the CPU and dispatches optimal kernels. For the AMX demo, vLLM "
            "is preferred because it provides OpenAI-compatible APIs and the clean AMX on/off "
            "comparison via DNNL_MAX_CPU_ISA."
        ),
    },
    {
        "id": "ml-016",
        "title": "Mixture of Experts Models and Sparse Activation",
        "topic": "ml-inference",
        "text": (
            "Mixture of Experts (MoE) models replace the dense feed-forward network in each "
            "transformer layer with multiple 'expert' FFN sub-networks and a learned router that "
            "activates only k of N experts per token (typically k=2 of N=8 or N=64). This "
            "allows total parameter count to scale independently of per-token compute: Mixtral "
            "8×7B has 47B total parameters but activates only 13B per token. For CPU inference, "
            "MoE models present a different compute profile than dense models: expert routing "
            "creates irregular access patterns (different experts per token), reducing the "
            "effectiveness of AMX tile reuse. However, the expert FFN GEMMs themselves are "
            "still AMX-accelerated when batch sizes are large enough. Expert parallelism "
            "distributes different experts to different nodes, reducing per-node memory requirements."
        ),
    },
    {
        "id": "ml-017",
        "title": "Chunked Prefill and Prefill/Decode Disaggregation",
        "topic": "ml-inference",
        "text": (
            "Chunked prefill is a serving optimization that processes long prompts in fixed-size "
            "chunks (e.g., 512 tokens) interleaved with decode steps. This prevents long prompts "
            "from blocking the decode queue (head-of-line blocking) at the cost of slightly higher "
            "TTFT for the chunked request. For AMX demonstration purposes, chunked prefill can "
            "reduce observed TTFT speedup if chunks are too small (AMX is most efficient on large "
            "GEMMs). Prefill/Decode (P/D) disaggregation separates prefill and decode into "
            "dedicated server pools: prefill servers (benefit from AMX) and decode servers (memory-"
            "bandwidth bound, no AMX benefit). This architecture allows independent scaling: add "
            "AMX-capable prefill nodes as prompt lengths grow; add decode nodes for higher "
            "concurrent user throughput."
        ),
    },
    {
        "id": "ml-018",
        "title": "Text Chunking Strategies for RAG",
        "topic": "ml-inference",
        "text": (
            "Document chunking divides source text into retrieval units before embedding. Chunk "
            "size is a critical RAG hyperparameter: too small (< 128 tokens) loses context and "
            "causes embeddings to lack semantic richness; too large (> 1024 tokens) reduces "
            "retrieval precision and wastes LLM context window. Common strategies: fixed-size "
            "chunking with overlap (e.g., 512 tokens, 50-token overlap) is simple and predictable; "
            "recursive character text splitter respects paragraph and sentence boundaries; "
            "semantic chunking uses embedding similarity to detect topic shifts and split there. "
            "For the AMX demo, synthetic documents are sized at 200-400 words (~150-300 tokens), "
            "representing typical enterprise knowledge base chunks — small enough for high-precision "
            "retrieval, large enough for embeddings to be semantically meaningful."
        ),
    },
    {
        "id": "ml-019",
        "title": "Evaluation Metrics for RAG Systems",
        "topic": "ml-inference",
        "text": (
            "RAG evaluation combines retrieval quality and generation quality metrics. Retrieval "
            "metrics: Recall@k (fraction of relevant docs in top-k results), MRR (Mean Reciprocal "
            "Rank), NDCG (Normalized Discounted Cumulative Gain). Generation metrics: RAGAS "
            "framework measures faithfulness (answer supported by retrieved context), answer "
            "relevance (answer addresses the question), context precision (retrieved docs contain "
            "the answer), context recall (retrieved docs cover the answer). Automated evaluation "
            "uses an LLM as judge: prompting a capable model (GPT-4, Claude) to score answer "
            "quality on a 1-5 scale. For the AMX demo, the primary evaluation is performance "
            "(latency, throughput), not quality — since identical models are compared with AMX "
            "on/off, answer quality is identical by construction."
        ),
    },
    {
        "id": "ml-020",
        "title": "Model Parallelism Strategies for Large Models",
        "topic": "ml-inference",
        "text": (
            "Models too large for a single node require distributed inference. Tensor parallelism "
            "(TP) splits each weight matrix across devices along a dimension; all-reduce "
            "synchronization is required after each linear layer — communication-intensive. "
            "Pipeline parallelism (PP) assigns consecutive layers to different devices; "
            "micro-batching hides pipeline bubbles. For CPU inference on multi-socket Xeon, "
            "tensor parallelism across sockets is supported by vLLM via distributed_executor_backend "
            "and tensor_parallel_size. Each socket runs a model shard; UPI provides the all-reduce "
            "communication. AMX accelerates each shard's GEMM independently. In practice, a "
            "4-socket Xeon system with 1 TB DRAM can serve 70B parameter models (BF16) with "
            "tensor parallelism, with AMX providing ~6× prefill speedup on each socket."
        ),
    },

    # -----------------------------------------------------------------------
    # Cluster 3 — LLM Serving & Inference Engines
    # -----------------------------------------------------------------------
    {
        "id": "llm-001",
        "title": "vLLM Architecture Overview",
        "topic": "llm-serving",
        "text": (
            "vLLM is an open-source LLM inference and serving engine developed at UC Berkeley. "
            "Its key innovation is PagedAttention — a virtual-memory-inspired KV cache management "
            "system that eliminates memory fragmentation and enables zero-copy KV cache sharing "
            "across requests with common prefixes. The vLLM serving stack includes: a FastAPI-based "
            "OpenAI-compatible HTTP server; an async LLM engine with continuous batching scheduler; "
            "a model executor (GPU or CPU worker); and platform-specific attention backends "
            "(FlashAttention, FlashInfer, or CPU attention). For CPU deployment, vLLM uses a "
            "dedicated CPU worker (vllm/v1/worker/cpu_worker.py) that leverages Intel oneDNN via "
            "PyTorch's CPU backend, with optional AMX acceleration via DNNL_MAX_CPU_ISA."
        ),
    },
    {
        "id": "llm-002",
        "title": "OpenAI API Compatibility in Self-Hosted LLMs",
        "topic": "llm-serving",
        "text": (
            "The OpenAI API has become the de-facto standard interface for LLM services. "
            "Self-hosted inference engines (vLLM, Ollama, LM Studio, llama.cpp server, TGI) "
            "implement the same REST endpoints: POST /v1/chat/completions, POST /v1/completions, "
            "POST /v1/embeddings, GET /v1/models. This compatibility allows client code written "
            "for the OpenAI API to work unchanged with local deployments by changing the base_url "
            "parameter. The openai Python SDK supports this via OpenAI(base_url='http://localhost:8000/v1', "
            "api_key='dummy'). Streaming responses use Server-Sent Events (SSE) with data: prefixed "
            "chunks in JSON format and a final data: [DONE] marker. stream_options: {include_usage: true} "
            "returns token counts alongside the stream."
        ),
    },
    {
        "id": "llm-003",
        "title": "Continuous Batching Implementation in vLLM",
        "topic": "llm-serving",
        "text": (
            "vLLM's scheduler implements continuous batching (also called iteration-level "
            "scheduling). At each forward pass step, the scheduler: (1) checks for completed "
            "sequences and removes them from the batch; (2) promotes waiting requests to the "
            "running state if KV cache space is available; (3) may preempt running sequences "
            "(swapping their KV cache to CPU or disk) if memory is exhausted. The scheduler "
            "prioritizes filling the GPU/CPU with useful work at every step. For CPU inference, "
            "batch sizes are typically smaller than GPU due to lower memory bandwidth, but the "
            "continuous batching logic is identical. Each forward pass processes a mixed batch "
            "of prefill tokens (from newly admitted requests) and decode tokens (from requests "
            "already in the generate phase), enabling high utilization."
        ),
    },
    {
        "id": "llm-004",
        "title": "Prefix Caching and Prompt Sharing in vLLM",
        "topic": "llm-serving",
        "text": (
            "vLLM's prefix caching (enable_prefix_caching=True, the default) stores computed "
            "KV cache blocks for previously seen prompt prefixes and reuses them for new "
            "requests with the same prefix. This dramatically reduces TTFT for repeated prompts "
            "— a request with a cached 2,000-token system prompt pays near-zero prefill cost. "
            "For benchmarking, prefix caching must be defeated to measure true prefill throughput: "
            "appending a unique per-run suffix (e.g., [run 3] or a UUID) ensures each request "
            "is cache-cold. The vLLM benchmark client query_vllm_amx.py uses this technique. "
            "Prefix caching is particularly valuable for RAG: if many queries share the same "
            "retrieved document chunks, the KV cache for those chunks is computed only once. "
            "This is a production optimization that can halve TTFT for common knowledge base queries."
        ),
    },
    {
        "id": "llm-005",
        "title": "IBM Granite Model Family",
        "topic": "llm-serving",
        "text": (
            "IBM Granite is a family of open-source enterprise-focused language models. "
            "Granite-3.3-8B-Instruct is an 8-billion-parameter model fine-tuned for instruction "
            "following, question answering, and code generation. Architecture: 32 transformer "
            "layers, hidden size 4096, 32 attention heads with GQA (8 KV heads), 4×4096 "
            "feed-forward size, RoPE positional embeddings, 131,072-token context window. "
            "Training: trained on 12 trillion tokens of curated English and code data with "
            "safety filtering. Released under Apache 2.0 license, available on Hugging Face. "
            "The model is optimized for enterprise deployment: it follows instructions reliably, "
            "acknowledges uncertainty, and avoids hallucination better than many comparably-sized "
            "models. BF16 weights require ~16 GB memory; INT8 quantization reduces this to ~8 GB."
        ),
    },
    {
        "id": "llm-006",
        "title": "Llama Model Architecture and Inference",
        "topic": "llm-serving",
        "text": (
            "Meta's Llama model family has driven open-source LLM adoption. Llama 3.1-8B-Instruct "
            "uses 32 transformer layers, hidden size 4096, 32 Q heads with GQA (8 KV heads), "
            "SwiGLU activation in the FFN, RoPE-θ=500000 positional embedding supporting "
            "128,000-token context. Training: 15 trillion tokens. The model was fine-tuned with "
            "supervised fine-tuning (SFT) and RLHF for instruction following. Llama 3.1 405B "
            "(released simultaneously) is a frontier-scale model with 128K context. For CPU "
            "inference, Llama 3.1-8B BF16 requires 16 GB DRAM and delivers ~12 tok/s decode "
            "throughput on a 40-core Xeon with AMX at 50 output tokens. The model is available "
            "under the Llama 3.1 Community License for commercial use up to 700M MAU."
        ),
    },
    {
        "id": "llm-007",
        "title": "LLM Serving Latency SLOs in Production",
        "topic": "llm-serving",
        "text": (
            "Production LLM deployments operate against Service Level Objectives (SLOs) that "
            "define acceptable latency bounds. Interactive applications (chatbots, copilots) "
            "typically require P95 TTFT < 2 seconds and TPOT < 50 ms/token to maintain the "
            "perception of real-time streaming. Batch processing (document analysis, classification) "
            "tolerates higher latency in exchange for higher throughput. For RAG use cases, "
            "TTFT is the user-visible latency metric — users wait for the first word before "
            "reading begins. AMX reduces TTFT 6× on Xeon, bringing a 21-second no-AMX response "
            "to 3.5 seconds — crossing the threshold from 'unusably slow' to 'acceptable' for "
            "interactive use. SLO attainment rates are measured at the P99 level; outlier TTFT "
            "spikes are often caused by cache misses or OS scheduler interference."
        ),
    },
    {
        "id": "llm-008",
        "title": "Inference Engine Comparison: vLLM, TGI, Ollama",
        "topic": "llm-serving",
        "text": (
            "Several inference engines provide OpenAI-compatible LLM serving. vLLM (UC Berkeley) "
            "leads in GPU throughput via PagedAttention and continuous batching; its CPU backend "
            "supports AMX via oneDNN. Text Generation Inference (TGI, Hugging Face) focuses on "
            "production robustness and Docker deployment; CPU support is more limited. Ollama "
            "wraps llama.cpp for consumer-grade CPU/GPU deployment with one-command model download "
            "and serving — no AMX optimization. llama.cpp uses hand-written SIMD kernels (AVX2, "
            "AVX-512) but does not leverage oneDNN AMX paths. For demonstrating Intel AMX, vLLM "
            "is the preferred choice: it uses PyTorch CPU backend with oneDNN dispatch, making "
            "DNNL_MAX_CPU_ISA an exact on/off switch for AMX kernels with no other code changes."
        ),
    },
    {
        "id": "llm-009",
        "title": "Structured Output and JSON Mode in LLM APIs",
        "topic": "llm-serving",
        "text": (
            "Structured output constrains LLM generation to follow a specific schema, enabling "
            "reliable parsing for downstream applications. vLLM supports guided decoding via "
            "outlines (guided_json parameter) and lm-format-enforcer: at each decode step, a "
            "grammar constraint masks out logits for tokens that would violate the schema, "
            "ensuring the output is always valid JSON/YAML/regex. JSON mode (response_format: "
            "{'type': 'json_object'}) is a softer constraint that prompts the model to produce "
            "JSON without formal grammar enforcement. For RAG applications, structured output "
            "enables reliable extraction of cited sources, confidence scores, or structured "
            "answers (e.g., {answer: '...', sources: [1, 3], confidence: 0.92}). The grammar "
            "constraint overhead is small — typically < 5% additional latency per token."
        ),
    },
    {
        "id": "llm-010",
        "title": "Prompt Engineering for RAG Quality",
        "topic": "llm-serving",
        "text": (
            "Well-designed prompts are critical for RAG answer quality. A standard RAG system "
            "prompt structure: (1) role definition ('You are a helpful assistant that answers "
            "questions based only on the provided context'); (2) retrieved context (formatted "
            "clearly with document boundaries); (3) instructions ('If the context does not "
            "contain the answer, say so clearly. Do not fabricate information.'); (4) user "
            "question. For AMX demos, long system prompts + retrieved context is deliberately "
            "used to maximize prefill length (2,000-3,000 tokens), which is where AMX provides "
            "maximum benefit. Cache busting is achieved by adding a unique timestamp or run ID "
            "to prevent vLLM from reusing KV cache across benchmark runs. Temperature=0 (or 0.1) "
            "ensures reproducible outputs for quality evaluation."
        ),
    },
    {
        "id": "llm-011",
        "title": "LLM Safety and Alignment Methods",
        "topic": "llm-serving",
        "text": (
            "Modern LLMs are aligned with human preferences through post-training techniques. "
            "Reinforcement Learning from Human Feedback (RLHF) trains a reward model on human "
            "preference pairs, then uses PPO to fine-tune the LLM to maximize reward. Direct "
            "Preference Optimization (DPO) simplifies this by directly optimizing the policy "
            "on preference data without a separate reward model. Constitutional AI (Anthropic) "
            "uses AI-generated critiques and revisions based on a constitution of principles. "
            "Granite-3.3 applies all these techniques plus safety-specific fine-tuning to reduce "
            "harmful outputs. For enterprise RAG deployment, alignment reduces hallucination and "
            "ensures the model defers appropriately to retrieved context rather than confabulating "
            "from parametric memory — critical for legal, medical, and financial applications."
        ),
    },
    {
        "id": "llm-012",
        "title": "Tokenizer Parallelism and vLLM Performance",
        "topic": "llm-serving",
        "text": (
            "Tokenization is a preprocessing step that can become a bottleneck at high request "
            "rates. A Rust-based tokenizer (HuggingFace tokenizers library) processes ~1M tokens/"
            "second per core — fast enough that it rarely bottlenecks single-request inference "
            "but can limit throughput in high-concurrency serving. vLLM runs tokenization in the "
            "FastAPI request handler (async, on the main thread) before submitting to the LLM "
            "engine. Setting TOKENIZERS_PARALLELISM=false prevents tokenizer internal threading "
            "from conflicting with the inference process's OpenMP threads. For CPU inference "
            "benchmarks, tokenization overhead is < 1% of total TTFT for 2,000-token prompts — "
            "negligible. It is more relevant for very short prompts where prefill is fast."
        ),
    },
    {
        "id": "llm-013",
        "title": "Model Serving Frameworks: Ray Serve and Triton",
        "topic": "llm-serving",
        "text": (
            "Production LLM serving typically uses a deployment framework on top of the inference "
            "engine. Ray Serve provides a Python-native distributed serving layer with autoscaling, "
            "A/B testing, and model composition. vLLM integrates natively with Ray for multi-node "
            "deployment. NVIDIA Triton Inference Server is a C++ serving system with model "
            "repository management, dynamic batching, and ensemble pipelines (preprocessing → "
            "inference → postprocessing as a DAG). Both support CPU backends. For the AMX demo, "
            "a simple Flask frontend (PWI-Flask-2vLLM-v2.py) is sufficient — production deployments "
            "would add an API gateway, authentication, rate limiting, and observability (metrics, "
            "traces, logs via OpenTelemetry). The key metric for production sizing is P95 TTFT "
            "under concurrent load, not single-request benchmarks."
        ),
    },
    {
        "id": "llm-014",
        "title": "Tensor Parallelism in vLLM for Multi-Socket CPU",
        "topic": "llm-serving",
        "text": (
            "vLLM supports tensor parallelism for CPU via the VLLM_CPU_OMP_THREADS_BIND mechanism "
            "and --tensor-parallel-size flag. With TP=2 on a dual-socket system, each socket "
            "holds half the weight matrices, and an all-reduce communication occurs after each "
            "attention and FFN layer. The inter-socket bandwidth (via UPI at ~200 GB/s on "
            "Sapphire Rapids) limits scaling efficiency — for an 8B model, the all-reduce volume "
            "is ~32 KB per layer per step (small relative to compute). TP effectively doubles "
            "available DRAM bandwidth for decode (each socket loads only half the weights) but "
            "adds communication overhead. For prefill (compute-bound), TP halves TTFT. AMX "
            "accelerates each socket's half-model GEMMs independently, and the combined AMX + TP "
            "speedup can approach 10-12× over single-socket no-AMX."
        ),
    },
    {
        "id": "llm-015",
        "title": "LLM Cost Models: CPU vs GPU Inference Economics",
        "topic": "llm-serving",
        "text": (
            "GPU inference (A100, H100) offers highest throughput for large batches but at high "
            "capital cost ($30,000-$40,000 per H100) and power draw (400-700W). CPU inference "
            "on Xeon with AMX offers a compelling alternative for lower-concurrency deployments: "
            "a dual-socket Xeon system serves one user request at ~12 tok/s decode at ~500W "
            "total system power, vs a fraction of an A100 for similar throughput. For RAG "
            "deployments with RAG-typical workloads (long prompt, short answer), the AMX Xeon's "
            "TTFT approaches GPU performance (3.5s AMX vs 0.5s A100 for 2,600 tokens), "
            "making it viable for use cases where GPU availability is limited, data privacy "
            "requires on-premise deployment, or infrastructure already includes high-core-count "
            "Xeon servers that can be repurposed for inference."
        ),
    },
    {
        "id": "llm-016",
        "title": "Monitoring LLM Inference with Prometheus and Grafana",
        "topic": "llm-serving",
        "text": (
            "vLLM exposes Prometheus metrics at /metrics endpoint including: vllm:e2e_request_latency_seconds "
            "(histogram), vllm:prompt_tokens_total, vllm:generation_tokens_total, "
            "vllm:num_requests_running (active batch size), vllm:gpu_cache_usage_perc (KV cache "
            "utilization). For CPU deployments, gpu_cache_usage_perc reports CPU KV cache usage. "
            "These metrics enable Grafana dashboards showing TTFT percentiles, throughput, and "
            "cache efficiency over time. Alerting on P95 TTFT > SLO threshold triggers capacity "
            "planning actions. AMX vs no-AMX comparison is visible in the TTFT histograms: the "
            "AMX container shows a sharply lower TTFT distribution. In production, both containers "
            "behind a load balancer would appear as a single service; the demo side-by-sides them "
            "explicitly for educational comparison."
        ),
    },
    {
        "id": "llm-017",
        "title": "Chat Templates and System Prompts",
        "topic": "llm-serving",
        "text": (
            "Different LLMs expect different chat message formats. OpenAI's chat format (system, "
            "user, assistant turns) is widely adopted. Models apply a Jinja2-based chat template "
            "during tokenization to format messages into the expected token sequence. Granite-3.3's "
            "chat template uses <|system|>, <|user|>, <|assistant|> special tokens. Llama 3.1 "
            "uses <|begin_of_text|>, <|start_header_id|>system<|end_header_id|> etc. When using "
            "the OpenAI-compatible API, vLLM automatically applies the correct chat template for "
            "the loaded model. For RAG, the system prompt contains the retrieved context and "
            "grounding instructions; the user turn contains only the question. This ensures the "
            "full context (1,000-3,000 tokens) is in the prefill, maximizing the AMX advantage."
        ),
    },
    {
        "id": "llm-018",
        "title": "Logits Processing: Temperature, Top-p, and Greedy Decoding",
        "topic": "llm-serving",
        "text": (
            "After each LLM forward pass, the raw output logits (unnormalized log-probabilities "
            "over the vocabulary) are post-processed before sampling. Temperature scaling divides "
            "logits by T: T=1 preserves the distribution, T<1 sharpens it (more deterministic), "
            "T→0 approaches greedy decoding (argmax). Top-p (nucleus) sampling truncates the "
            "vocabulary to the smallest set of tokens whose cumulative probability exceeds p (e.g., "
            "0.9), renormalizes, and samples. Top-k limits sampling to the k highest-probability "
            "tokens. For benchmarking AMX performance, temperature=0 (greedy) ensures reproducible "
            "outputs across runs — identical output token sequences allow direct comparison of "
            "generation time. Temperature=0.1 (near-greedy) is recommended for interactive demos "
            "to add slight variation while remaining mostly deterministic."
        ),
    },
    {
        "id": "llm-019",
        "title": "Flash Attention on CPU: vLLM's CPU Attention Backend",
        "topic": "llm-serving",
        "text": (
            "Flash Attention is an IO-aware attention algorithm that fuses the attention softmax "
            "computation into a single pass over Q/K/V matrices without materializing the full "
            "N×N attention score matrix. On GPUs, this reduces HBM memory traffic by up to 10×. "
            "On CPU, vLLM implements an analogous tiled attention computation in C++ (cpu_worker.py "
            "calls into the torch CPU attention C++ extension). The attention computation is "
            "split into blocks that fit in L2 cache, with online softmax normalization (Milakov-"
            "Norouzi algorithm). This is less critical on CPU than GPU (CPU caches are large) "
            "but still improves cache efficiency for long contexts. The prefill attention "
            "computation scales O(n²) with sequence length for standard attention; Flash "
            "Attention maintains the same asymptotic complexity but with better constants."
        ),
    },
    {
        "id": "llm-020",
        "title": "Intel Gaudi Accelerators for LLM Training and Inference",
        "topic": "llm-serving",
        "text": (
            "Intel Gaudi accelerators (formerly Habana Labs) are custom AI processors optimized "
            "for transformer training and inference. Gaudi 3 (2024) features 128 GB HBM2e, "
            "3.7 TB/s HBM bandwidth, and 64 Matrix Multiplication Engines (MMEs) with native "
            "BF16/FP8 support, delivering up to 1835 TFLOPS BF16. Gaudi connects to the host via "
            "PCIe 5.0 and supports scale-out via 24× 200 Gbps OSFP ports (direct server-to-server). "
            "Intel Gaudi software stack includes SynapseAI SDK, optimized Transformers library, "
            "and vLLM integration via the habana_frameworks backend. For the AMX demo narrative, "
            "Gaudi represents Intel's GPU-class AI accelerator while AMX demonstrates that "
            "Intel Xeon CPUs alone can deliver meaningful LLM inference performance for enterprise "
            "workloads without a discrete accelerator."
        ),
    },

    # -----------------------------------------------------------------------
    # Cluster 4 — Data Center Infrastructure
    # -----------------------------------------------------------------------
    {
        "id": "dc-001",
        "title": "Containerization and Docker for AI Workloads",
        "topic": "data-center",
        "text": (
            "Docker containers provide reproducible, isolated environments for AI inference "
            "deployment. Key patterns for LLM serving: multi-stage builds separate the "
            "compilation environment (with full SDK, compilers) from the runtime image, "
            "reducing final image size. BuildKit cache mounts (--mount=type=cache) preserve "
            "ccache and pip/uv download caches between builds, dramatically reducing rebuild "
            "time from 40 minutes to 2-3 minutes after initial build. For AMX workloads, "
            "containers must be started with --security-opt seccomp=unconfined to allow "
            "XSAVE/XRSTOR with AMX state (the default seccomp profile blocks XSAVE extensions). "
            "--cap-add SYS_NICE enables thread priority adjustment for optimal scheduler behavior. "
            "Volume mounts (-v ~/.cache/huggingface:/root/.cache/huggingface) share the model "
            "weight cache between host and container."
        ),
    },
    {
        "id": "dc-002",
        "title": "Kubernetes Deployment of LLM Inference Services",
        "topic": "data-center",
        "text": (
            "Production LLM serving on Kubernetes uses Deployments with resource requests and "
            "limits for CPU and memory. For Xeon AMX inference: cpu.requests should match the "
            "OMP thread count (e.g., 20 cores); memory.requests should cover model weights + "
            "KV cache (e.g., 64 Gi for an 8B BF16 model with 40 GB KV cache). "
            "securityContext.capabilities.add: [SYS_NICE] and seccompProfile: Unconfined are "
            "required for AMX. Horizontal Pod Autoscaler (HPA) scales replica count based on "
            "custom metrics (TTFT, queue depth) via Prometheus Adapter. For AMX-specific "
            "scheduling, Node Affinity rules target nodes with intel.com/amx label (set by "
            "Node Feature Discovery operator). Liveness probes poll /health; readiness probes "
            "poll /health after model load completes."
        ),
    },
    {
        "id": "dc-003",
        "title": "Milvus Vector Database Architecture",
        "topic": "data-center",
        "text": (
            "Milvus is an open-source vector database designed for billion-scale similarity "
            "search. Architecture: a stateless query layer (proxy nodes, query nodes) separates "
            "from a stateful storage layer (data nodes, index nodes, object storage via MinIO "
            "or S3). Metadata is stored in etcd. The separation enables independent scaling: "
            "add query nodes for higher search throughput without re-indexing. Milvus Standalone "
            "packages all components in a single Docker container for development and small-scale "
            "production. Index types: FLAT (brute force), IVF_FLAT, IVF_SQ8, HNSW, DISKANN, "
            "GPU_IVF_FLAT. The knowhere index library (based on Faiss) provides CPU and GPU "
            "implementations of all index types. BFloat16 vector storage is supported in "
            "Milvus 2.4+ for memory reduction."
        ),
    },
    {
        "id": "dc-004",
        "title": "Object Storage and Model Serving Infrastructure",
        "topic": "data-center",
        "text": (
            "Large-scale LLM deployment requires efficient model storage and loading. Object "
            "storage (S3, MinIO, Azure Blob) provides durable, high-throughput model weight "
            "storage. Safetensors format enables memory-mapped loading — model weights are loaded "
            "directly into process address space without intermediate copies, reducing load time "
            "from minutes to seconds for NVMe-backed storage. Hugging Face Hub is the dominant "
            "model registry; weights are cached locally in ~/.cache/huggingface. For Docker "
            "deployments, volume-mounting the HF cache ensures weights are downloaded once and "
            "shared across container restarts. Snapshots versioning in the Hub cache uses "
            "symlinks to deduplicate weights across model revisions. Model download bandwidth "
            "matters: a 16 GB 8B BF16 model on 1 Gbps takes ~130 seconds; on 10 Gbps, ~13 seconds."
        ),
    },
    {
        "id": "dc-005",
        "title": "Network Infrastructure for AI Clusters",
        "topic": "data-center",
        "text": (
            "Multi-node AI inference and training deployments require high-bandwidth, low-latency "
            "networking. InfiniBand (200 Gbps HDR, 400 Gbps NDR) with RDMA (Remote Direct Memory "
            "Access) enables GPU-to-GPU and CPU-to-CPU data transfer bypassing the OS kernel, "
            "reducing all-reduce latency for distributed inference. For vLLM multi-node CPU "
            "deployment, Ray uses TCP-based communication by default; RDMA backends (NCCL with "
            "RDMA transport or UCX) can improve scaling efficiency. 100 GbE (RoCEv2 RDMA over "
            "Converged Ethernet) provides a more cost-effective alternative to InfiniBand for "
            "smaller clusters. For single-node AMX demos, network is not a bottleneck — all "
            "inference happens within one server."
        ),
    },
    {
        "id": "dc-006",
        "title": "TCO Analysis: On-Premise vs Cloud LLM Inference",
        "topic": "data-center",
        "text": (
            "Total Cost of Ownership (TCO) for LLM inference depends on usage patterns, latency "
            "requirements, and data governance constraints. Cloud (API): OpenAI GPT-4o costs "
            "$5-15 per million tokens; no capital expense, pay-per-use. At 100M tokens/month, "
            "cloud cost is $500-1,500/month. On-premise (Xeon AMX): a 2-socket Xeon 8592+ "
            "system (~$25,000) serves ~12 tok/s decode; at 8 hours/day utilization, generates "
            "~350M tokens/month. Amortized over 3 years + power ($150/month), cost is ~$0.25/M "
            "tokens — 20-60× cheaper than cloud API at sustained utilization. For data-sensitive "
            "enterprises (healthcare, finance, government), on-premise also eliminates data "
            "egress and compliance risks. AMX doubles the Xeon ROI by delivering 6× prefill "
            "speedup without hardware upgrade."
        ),
    },
    {
        "id": "dc-007",
        "title": "Load Balancing Strategies for LLM Inference",
        "topic": "data-center",
        "text": (
            "Load balancing for LLM inference differs from traditional web services due to "
            "variable request cost (prompt length × model size) and stateful KV cache. "
            "Round-robin load balancing ignores request cost — a single long-prompt request "
            "can block shorter requests. Least-connections routing directs to the server with "
            "the fewest active requests — better but still ignores prompt length. Queue-depth-"
            "aware routing uses a custom metric (vllm:num_requests_running) to direct to the "
            "least-loaded instance. Prefix-aware routing (supported in SGLang and experimental "
            "in vLLM) routes requests with common prefixes to the same server to maximize "
            "prefix cache hits — critical for RAG where many queries share the same system "
            "prompt and retrieved documents. For the AMX demo, two separate containers serve "
            "as independent instances rather than a load-balanced pool."
        ),
    },
    {
        "id": "dc-008",
        "title": "Data Center Cooling and Power for AI Inference",
        "topic": "data-center",
        "text": (
            "AI inference workloads differ from traditional data center workloads in power "
            "density and thermal profile. A server with two Xeon Scalable 8592+ processors "
            "(350W TDP each) plus memory and storage runs at 900-1100W under full AI load. "
            "Rack power density increases from traditional 5-10 kW/rack to 30-40 kW/rack for "
            "AI-dense configurations. Air cooling requires hot aisle/cold aisle containment "
            "and high-airflow fans at 52 dBA+. Direct Liquid Cooling (DLC) with water blocks "
            "on CPUs and DIMMs reduces rack airflow requirements and allows higher rack density. "
            "Intel's Rear-Door Heat Exchanger (RDHx) captures exhaust air heat for facility "
            "heating recapture. Power Usage Effectiveness (PUE) is the ratio of total facility "
            "power to IT load; AI-optimized DLC data centers achieve PUE of 1.1-1.2 vs 1.4-1.6 "
            "for traditional air-cooled facilities."
        ),
    },
    {
        "id": "dc-009",
        "title": "Security Considerations for On-Premise LLM Deployment",
        "topic": "data-center",
        "text": (
            "Deploying LLMs on-premise addresses cloud data privacy concerns but introduces "
            "on-premise security requirements. Model weight access control: weights should be "
            "stored with filesystem permissions (chmod 600) and optionally encrypted at rest. "
            "API authentication: even internal LLM APIs should require bearer token authentication "
            "to prevent unauthorized usage and enable per-user rate limiting. Network isolation: "
            "LLM inference containers should be on a separate network segment, accessible only "
            "through an API gateway or proxy. Prompt injection: sanitize user input to prevent "
            "adversarial prompts from extracting system prompt contents or jailbreaking the model. "
            "Audit logging: log all inference requests (with redaction of sensitive content) for "
            "compliance. Hardware security: Secure Boot and measured boot chains prevent "
            "tampering with the inference environment."
        ),
    },
    {
        "id": "dc-010",
        "title": "Observability Stack for AI Inference: OpenTelemetry and Jaeger",
        "topic": "data-center",
        "text": (
            "Full observability for LLM serving requires metrics, traces, and logs. OpenTelemetry "
            "(OTel) provides a vendor-neutral SDK for instrumenting services. vLLM exposes "
            "Prometheus metrics natively; trace instrumentation can be added via OTel's Python "
            "SDK to capture per-request spans including tokenization, queue wait, prefill, and "
            "decode phases. Jaeger or Zipkin visualize distributed traces for debugging latency "
            "outliers. Structured logging (JSON format) with request IDs enables correlation "
            "between metrics, traces, and logs in tools like Elasticsearch/Kibana (ELK) or "
            "Grafana Loki. For AMX comparison dashboards, a Grafana panel showing TTFT CDF "
            "(cumulative distribution function) for AMX vs no-AMX containers provides a "
            "compelling visual demonstration of the performance difference."
        ),
    },
    {
        "id": "dc-011",
        "title": "Helm Charts and GitOps for LLM Deployment",
        "topic": "data-center",
        "text": (
            "Infrastructure-as-code patterns improve LLM deployment reliability. Helm charts "
            "package Kubernetes manifests with configurable values (model name, replica count, "
            "CPU/memory requests, AMX environment variables). ArgoCD or Flux implement GitOps: "
            "the cluster state is driven by Git commits, providing audit trails, rollback, and "
            "multi-environment promotion (dev → staging → prod). For LLM deployments, blue-green "
            "deployment enables zero-downtime model version upgrades: deploy new model version "
            "alongside old, validate with canary traffic, then switch load balancer. Rolling "
            "updates are trickier for LLMs due to model load time (30-120 seconds for 8B models "
            "on cold start) — pre-warming replicas before traffic cutover is essential."
        ),
    },
    {
        "id": "dc-012",
        "title": "NVMe Storage and Model Loading Performance",
        "topic": "data-center",
        "text": (
            "Model loading time — from filesystem read to inference-ready — is dominated by "
            "storage I/O for large models. PCIe 5.0 NVMe SSDs deliver 12-14 GB/s sequential read, "
            "loading a 16 GB 8B BF16 model in ~1.2 seconds. PCIe 4.0 NVMe at 7 GB/s takes ~2.3s. "
            "SATA SSD at 550 MB/s takes ~30 seconds. For Docker deployments with HuggingFace cache, "
            "the first container start downloads weights (~130s on 1 Gbps); subsequent starts load "
            "from local NVMe. Memory-mapped loading (safetensors format, mmap=True in vLLM) "
            "loads only accessed pages from disk — for inference, all pages are accessed during "
            "model initialization so full load time applies. NUMA-local NVMe (PCIe slots "
            "connected to the same socket as the inference process) avoids UPI latency for "
            "initial weight loading."
        ),
    },
    {
        "id": "dc-013",
        "title": "Milvus Standalone vs Distributed Deployment",
        "topic": "data-center",
        "text": (
            "Milvus offers two deployment modes. Standalone (single node, Docker or binary): "
            "all components co-located (coordinator, proxy, query node, data node, index node); "
            "MinIO and etcd embedded or external. Suitable for up to ~100M vectors. Distributed "
            "(Kubernetes, helm chart): each component scales independently; supports billions of "
            "vectors and thousands of QPS. For the AMX RAG demo, Milvus Standalone is appropriate: "
            "the synthetic corpus is 100 documents (~100 vectors), far below Standalone limits. "
            "Milvus 2.5 introduces MixCoord (merged coordinator) for Standalone, reducing "
            "memory footprint. The Milvus Lite variant (pip install pymilvus) embeds the vector "
            "store directly in Python without any server — simpler but limited to ~1M vectors "
            "and not production-grade. For demo credibility, Standalone is preferred."
        ),
    },
    {
        "id": "dc-014",
        "title": "Backup and Disaster Recovery for Vector Databases",
        "topic": "data-center",
        "text": (
            "Vector database backup requires capturing both index structures and raw vector data. "
            "Milvus's milvus-backup tool (open source) exports collections to object storage "
            "(MinIO/S3) in a portable format. Incremental backup captures only newly inserted "
            "vectors since the last full backup. For point-in-time recovery, WAL (write-ahead log) "
            "replay reconstructs the collection state to any historical point. Cross-datacenter "
            "replication for Milvus uses etcd replication for metadata and MinIO bucket "
            "replication for object data. Recovery Time Objective (RTO) for a 100M-vector "
            "collection restore from object storage is ~10-30 minutes depending on index "
            "rebuild time. For RAG applications, corpus data is typically more durable than "
            "the index (can be rebuilt from source documents if needed)."
        ),
    },
    {
        "id": "dc-015",
        "title": "eBPF and Linux Performance Tools for Server Profiling",
        "topic": "data-center",
        "text": (
            "eBPF (extended Berkeley Packet Filter) enables safe kernel-space programs for "
            "observability without kernel module compilation. The bcc toolkit and bpftrace "
            "provide eBPF-based tools: biolatency (block I/O latency distribution), funclatency "
            "(function call latency), offcputime (off-CPU time analysis for blocking operations). "
            "For AMX inference profiling, relevant tools include: perf stat (hardware counter "
            "summary), perf record/report (sample-based profiling to find hot functions), "
            "Intel VTune Profiler (GUI-based, shows pipeline stalls and TMUL utilization), "
            "emon (Event Monitor, bulk PMU counter collection). The DNNL_VERBOSE=1 environment "
            "variable enables oneDNN's per-kernel logging showing ISA dispatch, tensor shapes, "
            "and execution time — directly confirming AMX vs AVX-512 kernel selection."
        ),
    },
    {
        "id": "dc-016",
        "title": "Infrastructure for Multi-Tenant LLM Services",
        "topic": "data-center",
        "text": (
            "Multi-tenant LLM infrastructure must enforce resource isolation between users. "
            "Kubernetes resource quotas limit per-namespace CPU, memory, and GPU allocation. "
            "vLLM's multi-lora support serves multiple LoRA-adapted models from a single base "
            "model instance, with dynamic adapter loading/unloading for efficient multi-tenant "
            "serving. Per-tenant rate limiting (tokens/minute, requests/minute) prevents noisy "
            "neighbor effects. Model-level access control (API key → model allowlist) prevents "
            "unauthorized access to premium models. For audit compliance, each inference request "
            "logs the API key, model, prompt hash (not content), token counts, and latency. "
            "Prompt content logging requires explicit opt-in due to privacy implications. "
            "Tenant-specific prompt prefixes (system prompts) are cached via prefix caching "
            "for latency benefits."
        ),
    },
    {
        "id": "dc-017",
        "title": "CI/CD Pipelines for AI Model Deployment",
        "topic": "data-center",
        "text": (
            "Continuous integration for AI models requires additional stages beyond traditional "
            "software CI. Model CI pipeline: (1) lint and test inference code; (2) build Docker "
            "image with new model or code; (3) run accuracy evaluation on a held-out test set "
            "(MMLU, HellaSwag, GSM8K); (4) run latency regression tests (TTFT, TPS) — flag if "
            "P95 TTFT degrades > 10%; (5) safety evaluation (run adversarial prompts, check "
            "refusal rates); (6) gate on all checks before deploying to staging. GitHub Actions "
            "or Jenkins orchestrate these pipelines. For the AMX demo, the Docker build CI "
            "(build_docker_amx.sh) compiles vLLM with AMX flags — a 20-40 minute build. "
            "Incremental builds via ccache reduce this to 2-5 minutes for code-only changes."
        ),
    },
    {
        "id": "dc-018",
        "title": "Edge AI Inference and Intel Core Ultra",
        "topic": "data-center",
        "text": (
            "Edge AI inference runs models on client devices or near-edge servers with power "
            "and thermal constraints. Intel Core Ultra (Meteor Lake and later) includes a "
            "Neural Processing Unit (NPU) alongside CPU and GPU tiles. The NPU is optimized "
            "for sustained low-power AI inference (10-15 TOPS, < 1W), suitable for always-on "
            "keyword detection or real-time video analytics. For LLM inference on client: "
            "Core Ultra 7 165H with AMX can run 3B-7B parameter models locally (with INT4 "
            "quantization). OpenVINO GenAI and LM Studio support NPU and CPU-AMX execution. "
            "Edge deployment avoids cloud latency and data privacy concerns. The AMX "
            "architecture is consistent from edge Core Ultra to data center Xeon — software "
            "written for AMX optimization on Xeon runs with the same code path on Core Ultra."
        ),
    },
    {
        "id": "dc-019",
        "title": "Service Mesh for AI Microservices",
        "topic": "data-center",
        "text": (
            "A service mesh (Istio, Linkerd) manages service-to-service communication in "
            "microservice deployments with mutual TLS, traffic policies, and observability. "
            "For LLM serving, a mesh adds: mTLS for encrypted in-cluster communication between "
            "the API gateway, embedding service, vector DB, and LLM backend; circuit breaking "
            "(open circuit if LLM latency exceeds threshold, return cached or degraded response); "
            "retry policies (retry failed requests with backoff); and traffic mirroring (shadow "
            "traffic to a new model version for validation without serving users). The sidecar "
            "proxy (Envoy) adds ~0.5 ms overhead per request — negligible compared to 3.5s "
            "AMX TTFT. Istio's Wasm extension support enables custom request transforms, "
            "useful for prompt preprocessing and response filtering at the network layer."
        ),
    },
    {
        "id": "dc-020",
        "title": "AMX Deployment Checklist for Production",
        "topic": "data-center",
        "text": (
            "Deploying Intel AMX for LLM inference in production requires verifying several "
            "layers of the stack. CPU: verify AMX flags via grep -o 'amx[^ ]*' /proc/cpuinfo; "
            "confirm amx_bf16, amx_int8, amx_tile present. OS: Linux kernel 5.17+ for AMX "
            "context save/restore support (XSAVE with AMX state). Docker: --security-opt "
            "seccomp=unconfined to allow XSAVE/XRSTOR; --cap-add SYS_NICE for thread priority. "
            "Runtime: LD_PRELOAD=libiomp5.so:libtcmalloc_minimal.so.4 for Intel OpenMP and "
            "TCMalloc. ISA dispatch: DNNL_MAX_CPU_ISA=AVX512_CORE_AMX to enable AMX kernels. "
            "Validation: run DNNL_VERBOSE=1 during a request and grep for avx512_core_amx in "
            "kernel dispatch output. Benchmarking: use query_vllm_amx.py with --max-tokens 1 "
            "for pure prefill measurement; expect ~6× TTFT speedup over AVX-512 BF16 baseline."
        ),
    },

    # -----------------------------------------------------------------------
    # Cluster 5 — Intel AMX & oneDNN Technology
    # -----------------------------------------------------------------------
    {
        "id": "amx-001",
        "title": "oneDNN: Intel's Deep Learning Math Library",
        "topic": "intel-amx",
        "text": (
            "oneAPI Deep Neural Network Library (oneDNN), formerly MKL-DNN, is Intel's open-source "
            "performance library for deep learning primitives. It provides highly optimized "
            "implementations of convolution, batch normalization, matrix multiplication, and "
            "attention operations targeting Intel CPUs and GPUs. Key AMX-related features: "
            "BRGEMM (batch-reduce GEMM) primitive uses AMX TMUL instructions for compute-bound "
            "GEMMs; matmul primitive dispatches to AMX when input dimensions exceed a threshold "
            "that makes tiling efficient. DNNL_MAX_CPU_ISA environment variable controls ISA "
            "dispatch ceiling without recompilation. DNNL_VERBOSE=1 logs each primitive execution "
            "with the selected kernel (avx512_core_amx or avx512_core_bf16), tensor shapes, and "
            "timing. PyTorch's CPU backend uses oneDNN via the ATen operator dispatch when "
            "USE_MKLDNN=1 (the default for PyTorch built with Intel optimizations)."
        ),
    },
    {
        "id": "amx-002",
        "title": "AMX Performance Characteristics by Tensor Shape",
        "topic": "intel-amx",
        "text": (
            "AMX delivers maximum throughput for large, 'square-ish' matrix multiply shapes. "
            "The TMUL instruction's 16×16 tile granularity means that matrices must be padded "
            "to 16-element boundaries — small dimensions (M, N, K < 16) have poor utilization. "
            "For transformer inference: prefill with a 2,000-token prompt performs GEMMs of shape "
            "[2000 × 4096] × [4096 × 4096] — both M and K are large, ideal for AMX (>95% tile "
            "utilization). Decode performs matrix-vector multiply [1 × 4096] × [4096 × 4096] — "
            "M=1 cannot fill a tile, AMX offers no advantage. Batch decode with B=16 requests "
            "gives M=16 — exactly one tile in height, moderate AMX benefit. Batch decode B=256 "
            "gives M=256 — 16 tiles, good AMX utilization, approaching 2-3× speedup over AVX-512."
        ),
    },
    {
        "id": "amx-003",
        "title": "TDPBF16PS Instruction Deep Dive",
        "topic": "intel-amx",
        "text": (
            "TDPBF16PS (Tile Dot Product of BF16 Pairs into Float32 Single precision) is the "
            "core AMX matrix multiply instruction. Syntax: TDPBF16PS tmm_dst, tmm_src1, tmm_src2. "
            "Operation: for each element (i,j) in the destination tile, compute "
            "tmm_dst[i,j] += sum_k(tmm_src1[i,k] * tmm_src2[k,j]) where src1 and src2 are BF16 "
            "and the accumulation is in float32. With 16 rows × 32 BF16 columns in each input "
            "tile and accumulating into 16×16 float32 output, a single TDPBF16PS instruction "
            "performs 16×16×32 = 8192 multiply-add operations. At 1 instruction/cycle on the "
            "TMUL execution unit, this delivers 8192 MADs/cycle. With AVX-512 VDPBF16PS "
            "(dot product of BF16 pairs), a 512-bit instruction processes 16 BF16 dot products "
            "= 32 MADs/cycle — 256× fewer per instruction, requiring 256× more instructions "
            "for the same compute."
        ),
    },
    {
        "id": "amx-004",
        "title": "AMX State Management and OS Context Switch",
        "topic": "intel-amx",
        "text": (
            "AMX introduces a new processor state component — the XTILEDATA state — that must be "
            "saved and restored during context switches. The XSAVE instruction extended (XSAVES/XRSTORS) "
            "handles AMX tile register state save/restore as part of the XSAVE framework. Linux "
            "kernel 5.17 added AMX support: arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) "
            "must be called by the process to request AMX permission (or the kernel enables it "
            "automatically for newer kernels). Docker containers with recent Linux kernels and "
            "--security-opt seccomp=unconfined have access to XSAVE instructions. The AMX tile "
            "state is 8 tiles × 1 KB = 8 KB per thread — larger than AVX-512's 512-byte ZMM state. "
            "Context switch overhead is proportional to XSAVE state size, typically negligible "
            "compared to OS scheduling latency."
        ),
    },
    {
        "id": "amx-005",
        "title": "Intel oneAPI and Software Ecosystem for AMX",
        "topic": "intel-amx",
        "text": (
            "Intel's oneAPI toolkit provides a comprehensive software development environment for "
            "AMX and other Intel hardware features. Key components: oneAPI Base Toolkit (oneMKL, "
            "oneTBB, oneDNN, oneDAL); Intel C++ Compiler Classic and Intel oneAPI DPC++/C++ "
            "Compiler; Intel Distribution for Python (optimized NumPy/SciPy/scikit-learn). "
            "For LLM inference: Intel Extension for PyTorch (IPEX) adds AMX-optimized LLM "
            "inference paths, BF16 auto-mixed precision, and a torch.compile backend for Xeon. "
            "IPEX's LLM module provides optimized forward passes for popular architectures "
            "(Llama, Granite, Mistral) with AMX acceleration. Alternative to vLLM for CPU "
            "inference; IPEX-LLM (formerly BigDL-LLM) targets both Xeon server and Core "
            "Ultra client platforms."
        ),
    },
    {
        "id": "amx-006",
        "title": "AMX Benchmark Methodology and Measurement Best Practices",
        "topic": "intel-amx",
        "text": (
            "Accurate AMX vs no-AMX comparison requires careful experimental design. Key "
            "controls: (1) Same hardware — use DNNL_MAX_CPU_ISA to disable AMX on the same "
            "CPU, not two different machines; (2) Cache busting — append unique run IDs to "
            "defeat vLLM prefix caching; (3) Warm-up runs — discard first 1-2 runs to ensure "
            "CPU frequency is boosted and code paths are JIT-compiled; (4) Cooldown between "
            "runs — 2-3 second sleep reduces DRAM thermal and bandwidth contention; (5) CPU "
            "frequency locking — use cpupower frequency-set -g performance to prevent frequency "
            "scaling noise; (6) Isolated cores — use taskset or numactl to pin to dedicated "
            "cores, avoiding OS background tasks; (7) Multiple runs + statistics — report P50 "
            "and P95, not just average, to capture tail behavior."
        ),
    },
    {
        "id": "amx-007",
        "title": "Sapphire Rapids Die Topology and Tile Architecture",
        "topic": "intel-amx",
        "text": (
            "Sapphire Rapids (4th Gen Xeon) uses a tile-based die architecture — four chiplets "
            "(tiles) connected by EMIB (Embedded Multi-die Interconnect Bridge) at ~896 GB/s "
            "bidirectional bandwidth. Each tile contains up to 15 Golden Cove cores, a "
            "portion of the distributed LLC (up to 30 MB per tile), 2 DDR5 memory controllers, "
            "and 2 UPI links for socket-to-socket connectivity. The four-tile design enables "
            "processor configurations from 28 to 60 cores (Sapphire Rapids standard) by varying "
            "the per-tile core count. High Bandwidth Memory (HBM) variants (Xeon Max / "
            "Sapphire Rapids HBM) add on-package HBM2e to each tile for ~1 TB/s memory bandwidth, "
            "dramatically improving decode throughput for LLM inference. AMX is present on all "
            "tiles and all core counts."
        ),
    },
    {
        "id": "amx-008",
        "title": "oneDNN Verbose Logging and AMX Kernel Verification",
        "topic": "intel-amx",
        "text": (
            "DNNL_VERBOSE=1 enables per-kernel logging in oneDNN, printing a comma-separated "
            "record for each primitive execution. Format: dnnl_verbose,exec,cpu,<primitive_kind>,"
            "<implementation_tag>,<message_format>,<shape_info>,<time_ms>. The implementation "
            "tag identifies the selected kernel: avx512_core_amx indicates AMX tile instructions; "
            "avx512_core_bf16 indicates AVX-512 BF16 VNNI (no AMX). For a running vLLM container "
            "with DNNL_VERBOSE=1 during inference: grep 'avx512_core_amx' should show lines for "
            "matmul and brgemm primitives in the AMX container; grep 'avx512_core_bf16' in the "
            "no-AMX container. DNNL_VERBOSE=2 adds timestamps and enables profiling mode. "
            "DNNL_VERBOSE_TIMESTAMP=1 adds wall-clock time. These are the ground-truth "
            "confirmations that AMX tile units are actually being used."
        ),
    },
    {
        "id": "amx-009",
        "title": "Emerald Rapids AMX Improvements",
        "topic": "intel-amx",
        "text": (
            "Emerald Rapids (5th Gen Intel Xeon Scalable, 2024) improves on Sapphire Rapids "
            "while maintaining socket compatibility (LGA4677). Key changes affecting AMX "
            "performance: increased L3 LLC per die (up to 320 MB vs 112.5 MB on Sapphire Rapids) "
            "significantly improves model weight cache residency — less DRAM traffic during "
            "prefill means higher sustained TMUL throughput. CPU count increases to 64 cores per "
            "socket. AMX scheduler improvements reduce pipeline stalls during TILELOADDT1 "
            "prefetch. DDR5 speed increases to 5600 MT/s improve decode bandwidth. Inference "
            "benchmarks on Emerald Rapids show 15-25% TTFT improvement over same-SKU Sapphire "
            "Rapids due to the LLC size advantage — long-prompt prefill GEMMs increasingly "
            "hit cache rather than DRAM."
        ),
    },
    {
        "id": "amx-010",
        "title": "AMX for Training vs Inference",
        "topic": "intel-amx",
        "text": (
            "AMX accelerates both training and inference, but the relative benefit differs. "
            "Training involves both forward pass (similar to inference prefill) and backward "
            "pass (gradient computation — also GEMM-heavy). Optimizer steps (Adam, SGD) are "
            "element-wise operations (use AVX-512, not AMX). Mixed-precision training (BF16 "
            "forward, float32 optimizer state) is well-supported by AMX-BF16 for forward/backward "
            "GEMMs. However, LLM training at scale requires gradient synchronization across "
            "many nodes (all-reduce) and high memory bandwidth for large batch sizes — training "
            "favors GPU clusters due to HBM bandwidth (2-4 TB/s on H100 vs 300 GB/s on Xeon) "
            "and NVLink/InfiniBand interconnects. For fine-tuning (LoRA, QLoRA) on smaller "
            "datasets, Xeon with AMX is competitive, especially for INT8 QLoRA."
        ),
    },
    {
        "id": "amx-011",
        "title": "Neural Network Layer Shapes and AMX Efficiency",
        "topic": "intel-amx",
        "text": (
            "AMX efficiency varies by neural network layer type and shape. Fully connected "
            "layers (linear projections) with large hidden dimensions (1024+) achieve near-peak "
            "AMX utilization — ideal case. Convolutional layers in vision models are implemented "
            "as GEMMs via im2col transformation, also AMX-accelerated. Embedding lookup is "
            "a memory-bound gather operation — no AMX benefit. Layer normalization (RMSNorm, "
            "LayerNorm) is element-wise with reduction — uses AVX-512. Activation functions "
            "(SiLU, GELU, ReLU) are element-wise — AVX-512. Softmax in attention is element-wise "
            "with exp and sum reduction — AVX-512. The attention score matrix multiply "
            "Q×K^T and scores×V are GEMMs — AMX-accelerated during prefill (batch QK^T over "
            "full sequence length), but matrix-vector during decode."
        ),
    },
    {
        "id": "amx-012",
        "title": "Intel AMX vs NVIDIA TensorRT for CPU Inference",
        "topic": "intel-amx",
        "text": (
            "TensorRT is NVIDIA's inference optimization toolkit for CUDA GPUs — not applicable "
            "to CPU inference. The CPU equivalents are Intel's AMX + oneDNN stack and OpenVINO. "
            "For comparative context: TensorRT on A100 achieves ~1,000 tok/s prefill throughput "
            "for Llama-7B with BF16; AMX on Sapphire Rapids 8592+ achieves ~784 tok/s prefill "
            "(from benchmark data at 1,032-token prompt). This ~1.3× difference, combined with "
            "the 3-8× higher cost per A100 vs Xeon, makes Xeon AMX competitive for workloads "
            "where GPU availability is constrained. For batched inference at high concurrency, "
            "GPU HBM bandwidth provides a larger advantage in the decode-dominated regime. "
            "The AMX vs GPU comparison is most favorable for prefill-heavy, long-input workloads."
        ),
    },
    {
        "id": "amx-013",
        "title": "XSAVE Feature Flags and Software AMX Enablement",
        "topic": "intel-amx",
        "text": (
            "Before using AMX instructions, software must verify CPU support and request "
            "permission from the OS. CPUID verification: leaf 7, sub-leaf 1, EDX bit 22 "
            "(AMX-BF16), bit 24 (AMX-TILE). Feature flag also visible in /proc/cpuinfo as "
            "amx_bf16, amx_tile, amx_int8. Permission request: on Linux 5.17+, "
            "arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) enables AMX for the calling "
            "thread. Failure returns EINVAL on non-AMX CPUs or ENOSYS on older kernels. "
            "Libraries (oneDNN, libblis) handle this automatically at initialization. If the "
            "permission is not granted, executing TILECONFIG raises a #UD (undefined instruction) "
            "exception. The TILERELEASE instruction frees tile state between AMX-using regions, "
            "reducing XSAVE overhead during context switches."
        ),
    },
    {
        "id": "amx-014",
        "title": "AMX in Cloud Instances: AWS, Azure, GCP",
        "topic": "intel-amx",
        "text": (
            "Major cloud providers offer instances with AMX-capable Xeon processors. AWS: "
            "M7i, C7i, R7i instance families (Sapphire Rapids); M8i (Emerald Rapids). "
            "Azure: Dv5, Ev5, Fsv3 use Sapphire Rapids; DCsv3 instances for confidential "
            "computing also feature AMX. GCP: C3 machine series (Sapphire Rapids). Verifying "
            "AMX in a cloud VM: grep 'amx' /proc/cpuinfo — most providers expose the AMX "
            "flags to guest VMs. DNNL_MAX_CPU_ISA can be used in cloud as on bare metal. "
            "TCO comparison shifts for cloud: AMX-capable instances don't cost more than "
            "non-AMX (same generation), so AMX provides free 6× prefill speedup in cloud "
            "vs running the same code on previous-generation non-AMX instances. This makes "
            "the upgrade from Ice Lake (3rd Gen, no AMX) to Sapphire Rapids (4th Gen, AMX) "
            "compelling even in cloud deployments."
        ),
    },
    {
        "id": "amx-015",
        "title": "Granite Rapids and Future AMX Architecture",
        "topic": "intel-amx",
        "text": (
            "Intel Granite Rapids (6th Gen Xeon, 2024) introduces a fundamentally new tile "
            "architecture with a dedicated tile fabric on die. P-core tiles (up to 128 cores "
            "per socket) each contain AMX units with improved throughput vs Sapphire/Emerald "
            "Rapids. E-core tiles (Granite Rapids D variant) target power-efficient workloads "
            "without AMX. Improvements: wider TMUL units (enhanced BF16 and FP16 support), "
            "larger tile register files, improved TILELOAD bandwidth from the extended LLC. "
            "INT8 AMX throughput doubles vs BF16 (narrower data type fits more values per tile). "
            "New in Granite Rapids: AMX-FP16 (TDPFP16PS instruction) for FP16 accumulation, "
            "enabling mixed FP16/BF16 inference pipelines. System memory expands with 12-channel "
            "DDR5 support on AP (multi-chip package) variants."
        ),
    },
    {
        "id": "amx-016",
        "title": "Intel Neural Compressor for AMX Optimization",
        "topic": "intel-amx",
        "text": (
            "Intel Neural Compressor (INC) is an open-source toolkit for model compression and "
            "optimization targeting Intel hardware. Key capabilities: automatic mixed precision "
            "(AMP) converts float32 ops to BF16/INT8 with accuracy validation; static and "
            "dynamic post-training quantization (PTQ) calibrates INT8 scales; smooth quantization "
            "(SmoothQuant) improves INT8 activation quantization; knowledge distillation and "
            "pruning for structured sparsity. For AMX deployment, INC's BF16 AMP generates "
            "models that use TDPBF16PS for all linear layers. The optimization workflow: load "
            "model → apply INC → evaluate accuracy on validation set → export quantized model "
            "→ serve with vLLM or OpenVINO. INC is model-framework agnostic (PyTorch, TensorFlow, "
            "ONNX) and supports all major LLM architectures."
        ),
    },
    {
        "id": "amx-017",
        "title": "AMX for Computer Vision: ResNet and ViT Inference",
        "topic": "intel-amx",
        "text": (
            "Beyond LLM inference, AMX accelerates classical and modern computer vision models. "
            "ResNet-50 inference: the dominant operations are 3×3 and 1×1 convolutions, "
            "implemented via GEMM (im2col). AMX accelerates the large 1×1 convolutions in "
            "residual blocks (56×56×64→256 spatial × channel GEMMs), delivering 4-5× throughput "
            "vs AVX-512 BF16. Vision Transformer (ViT) inference mirrors LLM: self-attention "
            "and FFN GEMMs dominate, showing 5-6× AMX speedup for large batch sizes. Real-time "
            "video analytics at the edge (Intel Core Ultra) and in the data center (Xeon) both "
            "benefit from AMX. ONNX Runtime with the oneDNN EP and OpenVINO both automatically "
            "leverage AMX for vision model inference on supported hardware."
        ),
    },
    {
        "id": "amx-018",
        "title": "Tile Configuration and Runtime Reconfiguration Overhead",
        "topic": "intel-amx",
        "text": (
            "Before using AMX instructions, the tile configuration must be loaded with TILECONFIG. "
            "The configuration specifies how many rows and columns each of the 8 tile registers "
            "contains — allowing tiles to be sized for the specific GEMM shape being computed. "
            "Tile reconfiguration (changing the configuration mid-computation) requires a new "
            "TILECONFIG and implicit TILERELEASE of current tiles. Overhead of TILECONFIG is "
            "~50-100 cycles — negligible for large GEMMs but potentially significant for many "
            "small reconfigured GEMMs. Library implementations minimize reconfigurations by "
            "processing all GEMMs of the same shape consecutively. For transformer inference, "
            "the Q/K/V projection GEMMs share the same shape within a layer, so a single "
            "TILECONFIG covers all three. TILERELEASE is called between different-shaped "
            "primitives (e.g., between attention and FFN layers)."
        ),
    },
    {
        "id": "amx-019",
        "title": "Power Efficiency of AMX: Performance Per Watt",
        "topic": "intel-amx",
        "text": (
            "AMX improves not just raw throughput but also performance-per-watt for matrix "
            "workloads. AVX-512 operations at peak load trigger dynamic frequency reduction "
            "(AVX-512 license throttle) due to high vector unit power draw. AMX TMUL instructions "
            "are more compute-dense per instruction issue — the processor does more useful work "
            "per clock cycle without proportionally higher power draw. Benchmarks comparing "
            "AMX vs AVX-512 at the same CPU power cap show AMX achieving 4-5× higher prefill "
            "throughput per watt. This is particularly relevant for data centers operating "
            "under power budget constraints: AMX allows the same prefill throughput with lower "
            "power cap, freeing power headroom for memory and I/O subsystems. Intel's "
            "Performance and Power Efficiency Processor (Efficient-Core variants) combine AMX "
            "with aggressive power gating for idle efficiency."
        ),
    },
    {
        "id": "amx-020",
        "title": "Building AMX Demo Infrastructure: Lessons Learned",
        "topic": "intel-amx",
        "text": (
            "Key lessons from building the AMX vs no-AMX vLLM benchmark infrastructure: "
            "(1) vLLM requires libiomp5.so in LD_PRELOAD — it checks at startup and raises a hard "
            "RuntimeError if missing; (2) prefix caching is enabled by default and will make "
            "TTFT appear near-zero on run 2+ if you don't add a unique per-run cache bust; "
            "(3) DNNL_MAX_CPU_ISA is the cleanest way to disable AMX — same container image, "
            "same binary, just different ISA dispatch ceiling; (4) max_tokens=1 is the cleanest "
            "prefill benchmark (Total Time ≈ TTFT); max_tokens=50 is the most honest demo "
            "workload (RAG sweet spot, 3.3× end-to-end speedup, looks realistic); (5) the no-AMX "
            "container will also show slow first responses during model load — wait for health "
            "endpoint before benchmarking; (6) stream_options:{include_usage:true} is needed to "
            "get actual prompt token counts from the vLLM API."
        ),
    },
]


def build_corpus() -> list[dict]:
    return _DOCS


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG corpus for AMX demo")
    parser.add_argument("--out", default="corpus.json", help="Output JSON file (default: corpus.json)")
    parser.add_argument("--count", action="store_true", help="Print document count and topic breakdown, then exit")
    args = parser.parse_args()

    corpus = build_corpus()

    if args.count:
        from collections import Counter
        topics = Counter(d["topic"] for d in corpus)
        print(f"Total documents: {len(corpus)}")
        for topic, count in sorted(topics.items()):
            print(f"  {topic}: {count}")
        return

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(corpus)} documents to {out_path}")


if __name__ == "__main__":
    main()
