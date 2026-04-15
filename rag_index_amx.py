#!/usr/bin/env python3
"""
rag_index_amx.py
----------------
Benchmarks AMX vs no-AMX embedding throughput and builds the Milvus vector index.

Steps:
  1. Load the synthetic corpus (corpus.json or generated in-memory)
  2. Encode all documents using the AMX embedding endpoint (port 8002)
     and the no-AMX embedding endpoint (port 8003) -- measuring throughput
  3. Compare embedding performance (docs/sec, avg latency per doc)
  4. Insert the AMX-generated embeddings into Milvus

Usage:
    # Full benchmark (both endpoints) + index into Milvus
    python3 rag_index_amx.py

    # Skip no-AMX benchmark, just index with AMX embeddings
    python3 rag_index_amx.py --skip-no-amx

    # Use pre-generated corpus file
    python3 rag_index_amx.py --corpus-file corpus.json

    # Custom endpoints
    python3 rag_index_amx.py --amx-embed-url http://localhost:8002 \
                              --no-amx-embed-url http://localhost:8003 \
                              --milvus-host localhost --milvus-port 19530

Prerequisites:
    pip install pymilvus openai rich
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Run: pip install openai rich pymilvus")
    sys.exit(1)

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_AMX_EMBED_URL    = "http://localhost:8002"
DEFAULT_NO_AMX_EMBED_URL = "http://localhost:8003"
DEFAULT_EMBED_MODEL      = "BAAI/bge-m3"
DEFAULT_MILVUS_HOST      = "localhost"
DEFAULT_MILVUS_PORT      = 19530
COLLECTION_NAME          = "amx_rag_demo"

# ---------------------------------------------------------------------------
# Embedding result
# ---------------------------------------------------------------------------
@dataclass
class EmbedResult:
    doc_id: str
    embedding: list[float]
    latency_ms: float


@dataclass
class EmbedBenchResult:
    label: str
    url: str
    results: list[EmbedResult] = field(default_factory=list)
    errors: int = 0

    @property
    def total_docs(self):
        return len(self.results)

    @property
    def avg_latency_ms(self):
        if not self.results:
            return float("nan")
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def total_time_s(self):
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / 1000.0

    @property
    def docs_per_sec(self):
        if not self.results:
            return 0.0
        # Wall-clock throughput (sequential)
        total_s = self.total_time_s
        return len(self.results) / total_s if total_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Core: embed a single document
# ---------------------------------------------------------------------------
def embed_document(
    client: OpenAI,
    model: str,
    text: str,
) -> tuple[list[float], float]:
    """Return (embedding_vector, latency_ms)."""
    t0 = time.perf_counter()
    response = client.embeddings.create(model=model, input=text)
    t1 = time.perf_counter()
    embedding = response.data[0].embedding
    return embedding, (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# Benchmark: embed all docs, measure throughput
# ---------------------------------------------------------------------------
def benchmark_embeddings(
    url: str,
    label: str,
    model: str,
    docs: list[dict],
    batch_size: int = 1,
) -> EmbedBenchResult:
    """
    Embed each document individually (batch_size=1) to measure per-doc latency.
    Returns EmbedBenchResult with embeddings and timing.
    """
    client = OpenAI(base_url=f"{url}/v1", api_key="dummy")
    result = EmbedBenchResult(label=label, url=url)

    iterator = track(docs, description=f"[cyan]Embedding ({label})[/cyan]") \
               if HAS_RICH else docs

    for doc in iterator:
        try:
            embedding, latency_ms = embed_document(client, model, doc["text"])
            result.results.append(EmbedResult(
                doc_id=doc["id"],
                embedding=embedding,
                latency_ms=latency_ms,
            ))
        except Exception as e:
            result.errors += 1
            if not HAS_RICH:
                print(f"  ERROR embedding {doc['id']}: {e}")

    return result


# ---------------------------------------------------------------------------
# Milvus: create collection and insert embeddings
# ---------------------------------------------------------------------------
def setup_milvus_collection(
    host: str,
    port: int,
    dim: int,
    drop_existing: bool = True,
) -> Collection:
    """Connect to Milvus, create (or recreate) the demo collection."""
    connections.connect("default", host=host, port=port)

    if drop_existing and utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"  Dropped existing collection '{COLLECTION_NAME}'")

    fields = [
        FieldSchema(name="id",        dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="title",     dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="topic",     dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text",      dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="AMX RAG demo corpus")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # HNSW index — high recall, logarithmic query time
    index_params = {
        "metric_type": "COSINE",
        "index_type":  "HNSW",
        "params":      {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"  Created collection '{COLLECTION_NAME}' with HNSW index (dim={dim})")
    return collection


def insert_into_milvus(
    collection: Collection,
    docs: list[dict],
    embed_results: list[EmbedResult],
) -> int:
    """Insert documents and their embeddings into the Milvus collection."""
    # Build a lookup from doc_id -> embedding
    embed_map = {r.doc_id: r.embedding for r in embed_results}

    ids, titles, topics, texts, embeddings = [], [], [], [], []
    for doc in docs:
        if doc["id"] not in embed_map:
            continue
        ids.append(doc["id"])
        titles.append(doc["title"])
        topics.append(doc["topic"])
        texts.append(doc["text"][:8000])  # truncate to field limit
        embeddings.append(embed_map[doc["id"]])

    data = [ids, titles, topics, texts, embeddings]
    collection.insert(data)
    collection.flush()
    print(f"  Inserted {len(ids)} documents into Milvus")
    return len(ids)


# ---------------------------------------------------------------------------
# Health check for embedding endpoints
# ---------------------------------------------------------------------------
def check_embed_health(url: str, label: str) -> tuple[bool, str]:
    import urllib.request
    try:
        req = urllib.request.urlopen(f"{url}/health", timeout=5)
        if req.getcode() != 200:
            return False, "unhealthy"
        req2 = urllib.request.urlopen(f"{url}/v1/models", timeout=5)
        data = json.loads(req2.read())
        model = data["data"][0]["id"] if data.get("data") else "unknown"
        return True, model
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
def print_embed_comparison(amx: EmbedBenchResult, no_amx: EmbedBenchResult):
    print("\n" + "=" * 64)
    print("  AMX vs NO-AMX EMBEDDING THROUGHPUT COMPARISON")
    print("=" * 64)
    print(f"  Documents embedded: {amx.total_docs}")
    print()

    if amx.avg_latency_ms > 0 and no_amx.avg_latency_ms > 0:
        speedup_latency = no_amx.avg_latency_ms / amx.avg_latency_ms
        speedup_throughput = amx.docs_per_sec / no_amx.docs_per_sec
    else:
        speedup_latency = float("nan")
        speedup_throughput = float("nan")

    if HAS_RICH:
        table = Table(title="Embedding Performance", show_header=True,
                      header_style="bold magenta")
        table.add_column("Metric",           style="dim",    width=26)
        table.add_column("AMX ✅",           style="cyan",   justify="right")
        table.add_column("No AMX (AVX-512)", style="yellow", justify="right")
        table.add_column("AMX Speedup",      style="green",  justify="right")

        table.add_row(
            "Avg latency / doc (ms)",
            f"{amx.avg_latency_ms:.1f}ms",
            f"{no_amx.avg_latency_ms:.1f}ms",
            f"{speedup_latency:.1f}x faster",
        )
        table.add_row(
            "Throughput (docs/sec)",
            f"{amx.docs_per_sec:.2f}",
            f"{no_amx.docs_per_sec:.2f}",
            f"{speedup_throughput:.1f}x higher",
        )
        table.add_row(
            "Total indexing time (s)",
            f"{amx.total_time_s:.1f}s",
            f"{no_amx.total_time_s:.1f}s",
            "",
        )
        table.add_row(
            "Errors",
            str(amx.errors),
            str(no_amx.errors),
            "",
        )
        console.print(table)
    else:
        rows = [
            ("Avg latency/doc (ms)",   f"{amx.avg_latency_ms:.1f}",  f"{no_amx.avg_latency_ms:.1f}",  f"{speedup_latency:.1f}x"),
            ("Throughput (docs/sec)",  f"{amx.docs_per_sec:.2f}",     f"{no_amx.docs_per_sec:.2f}",     f"{speedup_throughput:.1f}x"),
            ("Total indexing time (s)",f"{amx.total_time_s:.1f}s",    f"{no_amx.total_time_s:.1f}s",    ""),
            ("Errors",                 str(amx.errors),                str(no_amx.errors),                ""),
        ]
        print(f"{'Metric':<26} {'AMX':>12} {'No-AMX':>14} {'Speedup':>10}")
        print("-" * 64)
        for label, a, b, s in rows:
            print(f"{label:<26} {a:>12} {b:>14} {s:>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark AMX vs no-AMX embedding throughput and build Milvus index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--amx-embed-url",    default=DEFAULT_AMX_EMBED_URL)
    parser.add_argument("--no-amx-embed-url", default=DEFAULT_NO_AMX_EMBED_URL)
    parser.add_argument("--embed-model",      default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--milvus-host",      default=DEFAULT_MILVUS_HOST)
    parser.add_argument("--milvus-port",      default=DEFAULT_MILVUS_PORT, type=int)
    parser.add_argument("--corpus-file",      default=None,
                        help="Path to corpus JSON (default: generate in-memory)")
    parser.add_argument("--skip-no-amx",      action="store_true",
                        help="Skip no-AMX benchmark, only index with AMX embeddings")
    parser.add_argument("--skip-health",      action="store_true")
    parser.add_argument("--skip-milvus",      action="store_true",
                        help="Skip Milvus insertion (benchmark only)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   AMX VectorDB Indexing Benchmark                    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Load corpus
    if args.corpus_file:
        with open(args.corpus_file) as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} documents from {args.corpus_file}")
    else:
        from rag_corpus import build_corpus
        docs = build_corpus()
        print(f"Generated {len(docs)} synthetic documents in-memory")

    print()

    # Health checks
    if not args.skip_health:
        print("Checking embedding endpoint health...")
        for label, url in [("AMX embed", args.amx_embed_url),
                            ("No-AMX embed", args.no_amx_embed_url)]:
            ok, info = check_embed_health(url, label)
            status = "✅ READY" if ok else "❌ UNREACHABLE"
            print(f"  {label:<16} ({url}): {status}  {info}")
        print()

    # AMX embedding benchmark
    print("--- AMX Embedding (port 8002) ---")
    amx_result = benchmark_embeddings(
        url=args.amx_embed_url,
        label="AMX",
        model=args.embed_model,
        docs=docs,
    )
    print(f"  {amx_result.total_docs} docs | "
          f"avg {amx_result.avg_latency_ms:.1f}ms/doc | "
          f"{amx_result.docs_per_sec:.2f} docs/sec | "
          f"errors: {amx_result.errors}")

    no_amx_result = None
    if not args.skip_no_amx:
        print("\n--- No-AMX Embedding (port 8003) ---")
        no_amx_result = benchmark_embeddings(
            url=args.no_amx_embed_url,
            label="No-AMX",
            model=args.embed_model,
            docs=docs,
        )
        print(f"  {no_amx_result.total_docs} docs | "
              f"avg {no_amx_result.avg_latency_ms:.1f}ms/doc | "
              f"{no_amx_result.docs_per_sec:.2f} docs/sec | "
              f"errors: {no_amx_result.errors}")

        print_embed_comparison(amx_result, no_amx_result)

    # Milvus indexing
    if not args.skip_milvus:
        if not HAS_MILVUS:
            print("\n⚠️  pymilvus not installed — skipping Milvus insertion.")
            print("   Run: pip install pymilvus")
        elif amx_result.results:
            dim = len(amx_result.results[0].embedding)
            print(f"\n--- Milvus Indexing (HNSW, dim={dim}) ---")
            collection = setup_milvus_collection(
                host=args.milvus_host,
                port=args.milvus_port,
                dim=dim,
            )
            n = insert_into_milvus(collection, docs, amx_result.results)
            print(f"  ✅ Index ready: {n} documents in '{COLLECTION_NAME}' collection")
        else:
            print("\n⚠️  No embeddings to insert (AMX benchmark produced no results).")

    print("\nDone. ✅")


if __name__ == "__main__":
    main()
