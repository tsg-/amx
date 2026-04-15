#!/usr/bin/env python3
"""
rag_query_amx.py
----------------
Full RAG query benchmark: embed question → Milvus similarity search → LLM answer.
Compares AMX vs no-AMX at every stage.

Stages:
  1. Embed the user query using AMX (port 8002) and no-AMX (port 8003) endpoints
  2. Search Milvus for top-k similar document chunks (single search, shared)
  3. Build a RAG prompt with retrieved context
  4. Send to AMX LLM (port 8000) and no-AMX LLM (port 8001) for answer generation
  5. Report per-stage and end-to-end latency comparison

Usage:
    python3 rag_query_amx.py --question "What is Intel AMX and how does it accelerate LLM inference?"
    python3 rag_query_amx.py --runs 3 --top-k 5 --max-tokens 50
    python3 rag_query_amx.py --list-questions

Prerequisites:
    pip install openai pymilvus rich
    python3 rag_index_amx.py   # build the Milvus index first
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Run: pip install openai rich pymilvus")
    sys.exit(1)

try:
    from pymilvus import connections, Collection, utility
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False
    print("WARNING: pymilvus not installed. Retrieval will be skipped.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_AMX_EMBED_URL    = "http://localhost:8002"
DEFAULT_NO_AMX_EMBED_URL = "http://localhost:8003"
DEFAULT_AMX_LLM_URL      = "http://localhost:8000"
DEFAULT_NO_AMX_LLM_URL   = "http://localhost:8001"
DEFAULT_EMBED_MODEL      = "BAAI/bge-m3"
DEFAULT_LLM_MODEL        = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_MILVUS_HOST      = "localhost"
DEFAULT_MILVUS_PORT      = 19530
COLLECTION_NAME          = "amx_rag_demo"
DEFAULT_TOP_K            = 5
DEFAULT_MAX_TOKENS       = 50
DEFAULT_RUNS             = 3

SAMPLE_QUESTIONS = [
    "What is Intel AMX and how does it accelerate LLM inference?",
    "Why does AMX improve time-to-first-token but not decode throughput?",
    "What LLM serving workloads benefit most from AMX?",
    "How does Milvus store and search vector embeddings?",
    "Explain the difference between prefill and decode phases in transformer inference.",
    "What is retrieval-augmented generation and how does it work?",
    "Compare BF16 and INT8 quantization for CPU inference.",
    "How do NUMA and thread binding affect LLM inference performance?",
    "What is the difference between HNSW and IVF_FLAT vector indexes?",
    "Describe the AMX TDPBF16PS instruction and what it computes.",
]

RAG_SYSTEM_PROMPT = (
    "You are a helpful technical assistant answering questions based strictly on the "
    "provided context documents. If the context does not contain sufficient information "
    "to answer the question, say so clearly. Do not fabricate information."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    topic: str
    text: str
    score: float


@dataclass
class RAGRunResult:
    # Stage 1: query embedding
    embed_latency_ms: float
    # Stage 2: Milvus search
    search_latency_ms: float
    # Stage 3: LLM generation
    ttft_ms: float
    total_llm_ms: float
    prompt_tokens: int
    tokens_generated: int
    tokens_per_sec: float
    prefill_tps: float
    response_text: str
    error: Optional[str] = None

    @property
    def end_to_end_ms(self):
        return self.embed_latency_ms + self.search_latency_ms + self.total_llm_ms


@dataclass
class RAGBenchResult:
    label: str
    runs: list[RAGRunResult] = field(default_factory=list)

    @property
    def successful(self):
        return [r for r in self.runs if r.error is None]

    def _mean(self, attr):
        vals = [getattr(r, attr) for r in self.successful]
        return statistics.mean(vals) if vals else float("nan")

    @property
    def avg_embed_ms(self):      return self._mean("embed_latency_ms")
    @property
    def avg_search_ms(self):     return self._mean("search_latency_ms")
    @property
    def avg_ttft_ms(self):       return self._mean("ttft_ms")
    @property
    def avg_total_llm_ms(self):  return self._mean("total_llm_ms")
    @property
    def avg_e2e_ms(self):        return self._mean("end_to_end_ms")
    @property
    def avg_prefill_tps(self):   return self._mean("prefill_tps")
    @property
    def avg_tps(self):           return self._mean("tokens_per_sec")


# ---------------------------------------------------------------------------
# Stage 1: embed the query
# ---------------------------------------------------------------------------
def embed_query(client: OpenAI, model: str, text: str) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    response = client.embeddings.create(model=model, input=text)
    t1 = time.perf_counter()
    return response.data[0].embedding, (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# Stage 2: Milvus similarity search
# ---------------------------------------------------------------------------
def milvus_search(
    collection: "Collection",
    query_vector: list[float],
    top_k: int,
) -> tuple[list[RetrievedChunk], float]:
    t0 = time.perf_counter()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["id", "title", "topic", "text"],
    )
    t1 = time.perf_counter()
    search_ms = (t1 - t0) * 1000.0

    chunks = []
    for hit in results[0]:
        chunks.append(RetrievedChunk(
            doc_id=hit.entity.get("id", ""),
            title=hit.entity.get("title", ""),
            topic=hit.entity.get("topic", ""),
            text=hit.entity.get("text", ""),
            score=hit.score,
        ))
    return chunks, search_ms


# ---------------------------------------------------------------------------
# Stage 3: LLM generation with streaming TTFT
# ---------------------------------------------------------------------------
def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    context_chunks: list[RetrievedChunk],
    max_tokens: int,
    run_idx: int,
) -> RAGRunResult:
    # Build RAG prompt with retrieved context
    context_text = "\n\n".join(
        f"[Document {i+1}: {chunk.title}]\n{chunk.text}"
        for i, chunk in enumerate(context_chunks)
    )
    user_content = (
        f"Context documents:\n\n{context_text}\n\n"
        f"Question: {question} [run {run_idx}]"
    )

    prompt_tokens = len((RAG_SYSTEM_PROMPT + user_content)) // 4  # heuristic

    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = []

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                if chunk.usage.prompt_tokens:
                    prompt_tokens = chunk.usage.prompt_tokens
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text.append(delta.content)
                token_count += 1

        t1 = time.perf_counter()
        if first_token_time is None:
            first_token_time = t1

        ttft_ms = (first_token_time - t0) * 1000.0
        total_ms = (t1 - t0) * 1000.0
        tps = token_count / (t1 - t0) if (t1 - t0) > 0 else 0
        prefill_tps = prompt_tokens / (ttft_ms / 1000.0) if ttft_ms > 0 else 0

        return RAGRunResult(
            embed_latency_ms=0,   # filled in by caller
            search_latency_ms=0,  # filled in by caller
            ttft_ms=ttft_ms,
            total_llm_ms=total_ms,
            prompt_tokens=prompt_tokens,
            tokens_generated=token_count,
            tokens_per_sec=tps,
            prefill_tps=prefill_tps,
            response_text="".join(full_text),
        )

    except Exception as e:
        t1 = time.perf_counter()
        return RAGRunResult(
            embed_latency_ms=0,
            search_latency_ms=0,
            ttft_ms=0,
            total_llm_ms=(t1 - t0) * 1000.0,
            prompt_tokens=prompt_tokens,
            tokens_generated=0,
            tokens_per_sec=0,
            prefill_tps=0,
            response_text="",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Full RAG pipeline run
# ---------------------------------------------------------------------------
def run_rag_pipeline(
    amx_embed_url: str,
    no_amx_embed_url: str,
    amx_llm_url: str,
    no_amx_llm_url: str,
    embed_model: str,
    llm_model: str,
    question: str,
    collection,
    top_k: int,
    max_tokens: int,
    num_runs: int,
    cooldown: int = 2,
    skip_no_amx: bool = False,
) -> tuple[RAGBenchResult, RAGBenchResult, list[RetrievedChunk]]:

    amx_embed_client    = OpenAI(base_url=f"{amx_embed_url}/v1",    api_key="dummy")
    no_amx_embed_client = OpenAI(base_url=f"{no_amx_embed_url}/v1", api_key="dummy")
    amx_llm_client      = OpenAI(base_url=f"{amx_llm_url}/v1",      api_key="dummy")
    no_amx_llm_client   = OpenAI(base_url=f"{no_amx_llm_url}/v1",   api_key="dummy")

    amx_result    = RAGBenchResult(label="AMX")
    no_amx_result = RAGBenchResult(label="No-AMX")
    retrieved_chunks: list[RetrievedChunk] = []

    for run_idx in range(1, num_runs + 1):
        if run_idx > 1 and cooldown > 0:
            time.sleep(cooldown)

        print(f"\n  [Run {run_idx}/{num_runs}]")

        # --- AMX path ---
        print(f"  AMX:    embedding query...", end=" ", flush=True)
        amx_query_vec, amx_embed_ms = embed_query(amx_embed_client, embed_model, question)
        print(f"{amx_embed_ms:.0f}ms")

        # Milvus search (done once with AMX embeddings — both paths use same retrieved docs)
        if collection is not None:
            print(f"  Milvus: searching top-{top_k}...", end=" ", flush=True)
            chunks, search_ms = milvus_search(collection, amx_query_vec, top_k)
            retrieved_chunks = chunks
            print(f"{search_ms:.1f}ms  (scores: {[f'{c.score:.3f}' for c in chunks[:3]]})")
        else:
            chunks, search_ms = [], 0.0
            print("  Milvus: skipped (no collection)")

        print(f"  AMX:    generating answer...", end=" ", flush=True)
        amx_run = generate_answer(amx_llm_client, llm_model, question, chunks, max_tokens, run_idx)
        amx_run.embed_latency_ms = amx_embed_ms
        amx_run.search_latency_ms = search_ms
        amx_result.runs.append(amx_run)
        if amx_run.error:
            print(f"ERROR: {amx_run.error}")
        else:
            print(f"TTFT={amx_run.ttft_ms:.0f}ms  Total={amx_run.total_llm_ms:.0f}ms  "
                  f"E2E={amx_run.end_to_end_ms:.0f}ms")

        if not skip_no_amx:
            # --- No-AMX path ---
            print(f"  No-AMX: embedding query...", end=" ", flush=True)
            no_amx_query_vec, no_amx_embed_ms = embed_query(no_amx_embed_client, embed_model, question)
            print(f"{no_amx_embed_ms:.0f}ms")

            print(f"  No-AMX: generating answer...", end=" ", flush=True)
            no_amx_run = generate_answer(no_amx_llm_client, llm_model, question, chunks, max_tokens, run_idx)
            no_amx_run.embed_latency_ms = no_amx_embed_ms
            no_amx_run.search_latency_ms = search_ms  # same search, attributed to AMX path
            no_amx_result.runs.append(no_amx_run)
            if no_amx_run.error:
                print(f"ERROR: {no_amx_run.error}")
            else:
                print(f"TTFT={no_amx_run.ttft_ms:.0f}ms  Total={no_amx_run.total_llm_ms:.0f}ms  "
                      f"E2E={no_amx_run.end_to_end_ms:.0f}ms")

    return amx_result, no_amx_result, retrieved_chunks


# ---------------------------------------------------------------------------
# Print final comparison
# ---------------------------------------------------------------------------
def print_rag_comparison(
    amx: RAGBenchResult,
    no_amx: RAGBenchResult,
    question: str,
    chunks: list[RetrievedChunk],
):
    print("\n" + "=" * 72)
    print("  AMX vs NO-AMX END-TO-END RAG BENCHMARK")
    print("=" * 72)
    print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    print()

    def fmt_speedup(amx_val, no_amx_val, lower_is_better=True):
        if amx_val == 0 or no_amx_val == 0:
            return "N/A"
        ratio = no_amx_val / amx_val if lower_is_better else amx_val / no_amx_val
        return f"{ratio:.1f}x faster" if lower_is_better else f"{ratio:.1f}x higher"

    if HAS_RICH:
        # Retrieved docs panel
        if chunks:
            doc_list = "\n".join(
                f"  [{c.score:.3f}] {c.title} ({c.topic})"
                for c in chunks
            )
            console.print(Panel(doc_list, title="[dim]Retrieved Documents[/dim]", border_style="dim"))

        table = Table(title="RAG Pipeline Performance", show_header=True,
                      header_style="bold magenta")
        table.add_column("Stage / Metric",      style="dim",    width=28)
        table.add_column("AMX ✅",              style="cyan",   justify="right")
        table.add_column("No AMX (AVX-512)",    style="yellow", justify="right")
        table.add_column("AMX Speedup",         style="green",  justify="right")

        table.add_row(
            "Query embed latency (ms)",
            f"{amx.avg_embed_ms:.1f}ms",
            f"{no_amx.avg_embed_ms:.1f}ms" if no_amx.runs else "—",
            fmt_speedup(amx.avg_embed_ms, no_amx.avg_embed_ms) if no_amx.runs else "—",
        )
        table.add_row(
            "Milvus search latency (ms)",
            f"{amx.avg_search_ms:.1f}ms",
            "— (shared)",
            "",
        )
        table.add_row(
            "LLM TTFT (ms)",
            f"{amx.avg_ttft_ms:.1f}ms",
            f"{no_amx.avg_ttft_ms:.1f}ms" if no_amx.runs else "—",
            fmt_speedup(amx.avg_ttft_ms, no_amx.avg_ttft_ms) if no_amx.runs else "—",
        )
        table.add_row(
            "LLM prefill throughput (tok/s)",
            f"{amx.avg_prefill_tps:.0f}",
            f"{no_amx.avg_prefill_tps:.0f}" if no_amx.runs else "—",
            fmt_speedup(no_amx.avg_prefill_tps, amx.avg_prefill_tps, lower_is_better=False) if no_amx.runs else "—",
        )
        table.add_row(
            "LLM total time (ms)",
            f"{amx.avg_total_llm_ms:.1f}ms",
            f"{no_amx.avg_total_llm_ms:.1f}ms" if no_amx.runs else "—",
            fmt_speedup(amx.avg_total_llm_ms, no_amx.avg_total_llm_ms) if no_amx.runs else "—",
        )
        table.add_row(
            "End-to-end RAG latency (ms)",
            f"{amx.avg_e2e_ms:.1f}ms",
            f"{no_amx.avg_e2e_ms:.1f}ms" if no_amx.runs else "—",
            fmt_speedup(amx.avg_e2e_ms, no_amx.avg_e2e_ms) if no_amx.runs else "—",
        )
        console.print(table)

        if amx.successful:
            console.print(Panel(
                amx.successful[-1].response_text[:600] or "(empty)",
                title="[cyan]AMX Answer (last run)[/cyan]",
                border_style="cyan",
            ))
        if no_amx.successful:
            console.print(Panel(
                no_amx.successful[-1].response_text[:600] or "(empty)",
                title="[yellow]No-AMX Answer (last run)[/yellow]",
                border_style="yellow",
            ))
    else:
        if chunks:
            print("Retrieved documents:")
            for c in chunks:
                print(f"  [{c.score:.3f}] {c.title}")
            print()

        rows = [
            ("Query embed (ms)",         f"{amx.avg_embed_ms:.1f}",     f"{no_amx.avg_embed_ms:.1f}" if no_amx.runs else "—"),
            ("Milvus search (ms)",        f"{amx.avg_search_ms:.1f}",    "— (shared)"),
            ("LLM TTFT (ms)",             f"{amx.avg_ttft_ms:.1f}",      f"{no_amx.avg_ttft_ms:.1f}" if no_amx.runs else "—"),
            ("LLM total (ms)",            f"{amx.avg_total_llm_ms:.1f}", f"{no_amx.avg_total_llm_ms:.1f}" if no_amx.runs else "—"),
            ("End-to-end (ms)",           f"{amx.avg_e2e_ms:.1f}",       f"{no_amx.avg_e2e_ms:.1f}" if no_amx.runs else "—"),
        ]
        print(f"{'Stage/Metric':<28} {'AMX':>12} {'No-AMX':>14}")
        print("-" * 56)
        for label, a, b in rows:
            print(f"{label:<28} {a:>12} {b:>14}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Full RAG pipeline benchmark: embed → search → generate (AMX vs no-AMX)",
    )
    parser.add_argument("--question",          default=SAMPLE_QUESTIONS[0])
    parser.add_argument("--amx-embed-url",     default=DEFAULT_AMX_EMBED_URL)
    parser.add_argument("--no-amx-embed-url",  default=DEFAULT_NO_AMX_EMBED_URL)
    parser.add_argument("--amx-llm-url",       default=DEFAULT_AMX_LLM_URL)
    parser.add_argument("--no-amx-llm-url",    default=DEFAULT_NO_AMX_LLM_URL)
    parser.add_argument("--embed-model",       default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model",         default=DEFAULT_LLM_MODEL)
    parser.add_argument("--milvus-host",       default=DEFAULT_MILVUS_HOST)
    parser.add_argument("--milvus-port",       default=DEFAULT_MILVUS_PORT, type=int)
    parser.add_argument("--top-k",             default=DEFAULT_TOP_K, type=int)
    parser.add_argument("--max-tokens",        default=DEFAULT_MAX_TOKENS, type=int)
    parser.add_argument("--runs",              default=DEFAULT_RUNS, type=int)
    parser.add_argument("--cooldown",          default=2, type=int)
    parser.add_argument("--skip-no-amx",       action="store_true")
    parser.add_argument("--list-questions",    action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_questions:
        print("Sample questions:")
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"  {i:2d}. {q}")
        return

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   AMX RAG Query Benchmark                            ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"  Question:   {args.question[:60]}...")
    print(f"  Top-k:      {args.top_k}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Runs:       {args.runs}")
    print()

    # Connect to Milvus
    collection = None
    if HAS_MILVUS:
        try:
            connections.connect("default", host=args.milvus_host, port=args.milvus_port)
            if utility.has_collection(COLLECTION_NAME):
                collection = Collection(COLLECTION_NAME)
                collection.load()
                print(f"✅ Connected to Milvus — collection '{COLLECTION_NAME}' loaded")
            else:
                print(f"⚠️  Milvus collection '{COLLECTION_NAME}' not found.")
                print("   Run python3 rag_index_amx.py first to build the index.")
        except Exception as e:
            print(f"⚠️  Milvus connection failed: {e}")
    else:
        print("⚠️  pymilvus not installed — retrieval stage will be skipped.")

    print()

    amx_result, no_amx_result, chunks = run_rag_pipeline(
        amx_embed_url=args.amx_embed_url,
        no_amx_embed_url=args.no_amx_embed_url,
        amx_llm_url=args.amx_llm_url,
        no_amx_llm_url=args.no_amx_llm_url,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        question=args.question,
        collection=collection,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        num_runs=args.runs,
        cooldown=args.cooldown,
        skip_no_amx=args.skip_no_amx,
    )

    print_rag_comparison(amx_result, no_amx_result, args.question, chunks)


if __name__ == "__main__":
    main()
