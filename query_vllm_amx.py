#!/usr/bin/env python3
"""
query_vllm_amx.py
-----------------
Query two running vLLM containers (AMX vs no-AMX) with a prompt
and compare TTFT, throughput, and response quality.

Usage:
    python query_vllm_amx.py --prompt "Explain KV cache eviction strategies"
    python query_vllm_amx.py --prompt "What is Intel AMX?" --runs 5
    python query_vllm_amx.py --amx-url http://localhost:8000 \
                              --no-amx-url http://localhost:8001

Prerequisites:
    pip install openai rich
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Run: pip install openai rich")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_AMX_URL    = "http://localhost:8000"
DEFAULT_NO_AMX_URL = "http://localhost:8001"
#DEFAULT_MODEL      = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL      = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_MAX_TOKENS = 256
DEFAULT_RUNS       = 3

SAMPLE_PROMPTS = [
    "Explain KV cache eviction strategies in distributed LLM inference.",
    "What are the performance benefits of Intel AMX over AVX-512 for GEMM?",
    "Describe RDMA-based KV cache transfer between heterogeneous GPU clusters.",
    "Compare prefill vs decode phases in autoregressive transformer inference.",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    ttft_ms: float          # time to first token (ms)
    total_ms: float         # total request time (ms)
    tokens_generated: int
    tokens_per_sec: float
    prompt_tokens: int      # approximate prompt token count
    prefill_tps: float      # prompt tokens / TTFT — measures AMX GEMM speed
    response_text: str
    error: Optional[str] = None

@dataclass
class BenchResult:
    label: str
    url: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def successful_runs(self):
        return [r for r in self.runs if r.error is None]

    @property
    def avg_ttft_ms(self):
        vals = [r.ttft_ms for r in self.successful_runs]
        return statistics.mean(vals) if vals else float("nan")

    @property
    def p50_ttft_ms(self):
        vals = sorted(r.ttft_ms for r in self.successful_runs)
        return vals[len(vals) // 2] if vals else float("nan")

    @property
    def p95_ttft_ms(self):
        vals = sorted(r.ttft_ms for r in self.successful_runs)
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)] if vals else float("nan")

    @property
    def avg_total_ms(self):
        vals = [r.total_ms for r in self.successful_runs]
        return statistics.mean(vals) if vals else float("nan")

    @property
    def avg_prefill_tps(self):
        vals = [r.prefill_tps for r in self.successful_runs]
        return statistics.mean(vals) if vals else float("nan")

    @property
    def prompt_tokens(self):
        vals = [r.prompt_tokens for r in self.successful_runs]
        return int(statistics.mean(vals)) if vals else 0

    @property
    def avg_tps(self):
        vals = [r.tokens_per_sec for r in self.successful_runs]
        return statistics.mean(vals) if vals else float("nan")

    @property
    def success_rate(self):
        return len(self.successful_runs) / len(self.runs) * 100 if self.runs else 0

# ---------------------------------------------------------------------------
# Core query function (streaming for accurate TTFT)
# ---------------------------------------------------------------------------
def query_with_streaming(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    system_prompt: str = "You are a helpful AI assistant specializing in computer architecture and systems software.",
) -> RunResult:
    """Query vLLM using streaming to measure true TTFT."""

    first_token_time = None
    full_text = []
    token_count = 0

    # Approximate prompt token count (chars / 4 is a reasonable heuristic)
    prompt_tokens = len((system_prompt + prompt)) // 4

    t_start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # Capture actual prompt token count from usage if available
            if hasattr(chunk, "usage") and chunk.usage is not None:
                if chunk.usage.prompt_tokens:
                    prompt_tokens = chunk.usage.prompt_tokens
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text.append(delta.content)
                token_count += 1  # approximate; 1 chunk ≈ 1 token

        t_end = time.perf_counter()

        if first_token_time is None:
            first_token_time = t_end  # no tokens generated

        ttft_ms     = (first_token_time - t_start) * 1000
        total_ms    = (t_end - t_start) * 1000
        tps         = token_count / (t_end - t_start) if (t_end - t_start) > 0 else 0
        prefill_tps = prompt_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0

        return RunResult(
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_generated=token_count,
            tokens_per_sec=tps,
            prompt_tokens=prompt_tokens,
            prefill_tps=prefill_tps,
            response_text="".join(full_text),
        )

    except Exception as e:
        t_end = time.perf_counter()
        return RunResult(
            ttft_ms=0,
            total_ms=(t_end - t_start) * 1000,
            tokens_generated=0,
            tokens_per_sec=0,
            prompt_tokens=prompt_tokens,
            prefill_tps=0,
            response_text="",
            error=str(e),
        )

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def check_health(url: str, label: str) -> tuple[bool, str]:
    """Check /health endpoint and return (ok, model_name)."""
    import urllib.request
    import json

    try:
        # Check /health
        req = urllib.request.urlopen(f"{url}/health", timeout=5)
        if req.getcode() != 200:
            return False, ""

        # Get model name from /v1/models
        req2 = urllib.request.urlopen(f"{url}/v1/models", timeout=5)
        data = json.loads(req2.read())
        model = data["data"][0]["id"] if data.get("data") else "unknown"
        return True, model

    except Exception as e:
        return False, str(e)

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    url: str,
    label: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_runs: int,
    cooldown: int = 2,
) -> BenchResult:
    client = OpenAI(base_url=f"{url}/v1", api_key="dummy")
    result = BenchResult(label=label, url=url)

    iterator = track(range(num_runs), description=f"[cyan]{label}[/cyan]") \
               if HAS_RICH else range(num_runs)

    for i in iterator:
        if i > 0 and cooldown > 0:
            time.sleep(cooldown)
        # Append a unique suffix per run to defeat prefix caching,
        # ensuring each run triggers a real prefill computation.
        busted_prompt = f"{prompt} [run {i+1}]"
        run = query_with_streaming(client, model, busted_prompt, max_tokens)
        result.runs.append(run)

        if run.error:
            print(f"  Run {i+1}: ERROR — {run.error}")
        else:
            print(f"  Run {i+1}: TTFT={run.ttft_ms:.1f}ms  "
                  f"Total={run.total_ms:.1f}ms  "
                  f"Prefill={run.prefill_tps:.0f}tok/s  "
                  f"TPS={run.tokens_per_sec:.1f}  "
                  f"tokens={run.tokens_generated}")

    return result

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
def print_comparison(amx: BenchResult, no_amx: BenchResult, prompt: str):
    print("\n" + "="*72)
    print("  AMX vs NO-AMX vLLM BENCHMARK RESULTS")
    print("="*72)
    print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()

    def speedup(a, b):
        """How much faster is a vs b (as Nx)."""
        if b == 0 or a == 0:
            return float("nan")
        return b / a  # lower TTFT = better, so b/a for TTFT

    if HAS_RICH:
        table = Table(title="Performance Comparison", show_header=True,
                      header_style="bold magenta")
        table.add_column("Metric",          style="dim",    width=24)
        table.add_column("AMX ✅",          style="cyan",   justify="right")
        table.add_column("No AMX (AVX-512)",style="yellow", justify="right")
        table.add_column("AMX Speedup",     style="green",  justify="right")

        ttft_speedup    = speedup(amx.avg_ttft_ms, no_amx.avg_ttft_ms)
        total_speedup   = speedup(amx.avg_total_ms, no_amx.avg_total_ms)
        prefill_speedup = amx.avg_prefill_tps / no_amx.avg_prefill_tps if no_amx.avg_prefill_tps > 0 else float("nan")
        tps_speedup     = amx.avg_tps / no_amx.avg_tps if no_amx.avg_tps > 0 else float("nan")

        table.add_row(
            "Prompt tokens (approx)",
            f"{amx.prompt_tokens}",
            f"{no_amx.prompt_tokens}",
            "",
        )
        table.add_row(
            "Avg TTFT (ms)",
            f"{amx.avg_ttft_ms:.1f}ms",
            f"{no_amx.avg_ttft_ms:.1f}ms",
            f"{ttft_speedup:.1f}x faster" if not (ttft_speedup != ttft_speedup) else "N/A",
        )
        table.add_row(
            "P50 TTFT (ms)",
            f"{amx.p50_ttft_ms:.1f}ms",
            f"{no_amx.p50_ttft_ms:.1f}ms",
            "",
        )
        table.add_row(
            "P95 TTFT (ms)",
            f"{amx.p95_ttft_ms:.1f}ms",
            f"{no_amx.p95_ttft_ms:.1f}ms",
            "",
        )
        table.add_row(
            "Prefill throughput (tok/s)",
            f"{amx.avg_prefill_tps:.0f}",
            f"{no_amx.avg_prefill_tps:.0f}",
            f"{prefill_speedup:.1f}x higher" if not (prefill_speedup != prefill_speedup) else "N/A",
        )
        table.add_row(
            "Avg Total Time (ms)",
            f"{amx.avg_total_ms:.1f}ms",
            f"{no_amx.avg_total_ms:.1f}ms",
            f"{total_speedup:.1f}x faster" if not (total_speedup != total_speedup) else "N/A",
        )
        table.add_row(
            "Decode throughput (tok/s)",
            f"{amx.avg_tps:.1f}",
            f"{no_amx.avg_tps:.1f}",
            f"{tps_speedup:.1f}x higher" if not (tps_speedup != tps_speedup) else "N/A",
        )
        table.add_row(
            "Success rate",
            f"{amx.success_rate:.0f}%",
            f"{no_amx.success_rate:.0f}%",
            "",
        )

        console.print(table)

        # Show last response from AMX
        if amx.successful_runs:
            console.print(Panel(
                amx.successful_runs[-1].response_text[:600],
                title="[cyan]AMX Response (last run)[/cyan]",
                border_style="cyan",
            ))
        if no_amx.successful_runs:
            console.print(Panel(
                no_amx.successful_runs[-1].response_text[:600],
                title="[yellow]No-AMX Response (last run)[/yellow]",
                border_style="yellow",
            ))
    else:
        # Plain text fallback
        ttft_speedup    = speedup(amx.avg_ttft_ms, no_amx.avg_ttft_ms)
        total_speedup   = speedup(amx.avg_total_ms, no_amx.avg_total_ms)
        prefill_speedup = amx.avg_prefill_tps / no_amx.avg_prefill_tps if no_amx.avg_prefill_tps > 0 else float("nan")
        tps_speedup     = amx.avg_tps / no_amx.avg_tps if no_amx.avg_tps > 0 else float("nan")

        rows = [
            ("Prompt tokens (approx)",     f"{amx.prompt_tokens}",           f"{no_amx.prompt_tokens}",          ""),
            ("Avg TTFT (ms)",              f"{amx.avg_ttft_ms:.1f}",          f"{no_amx.avg_ttft_ms:.1f}",        f"{ttft_speedup:.1f}x"),
            ("P50 TTFT (ms)",              f"{amx.p50_ttft_ms:.1f}",          f"{no_amx.p50_ttft_ms:.1f}",        ""),
            ("P95 TTFT (ms)",              f"{amx.p95_ttft_ms:.1f}",          f"{no_amx.p95_ttft_ms:.1f}",        ""),
            ("Prefill throughput (tok/s)", f"{amx.avg_prefill_tps:.0f}",      f"{no_amx.avg_prefill_tps:.0f}",    f"{prefill_speedup:.1f}x"),
            ("Avg Total Time (ms)",        f"{amx.avg_total_ms:.1f}",         f"{no_amx.avg_total_ms:.1f}",       f"{total_speedup:.1f}x"),
            ("Decode throughput (tok/s)",  f"{amx.avg_tps:.1f}",              f"{no_amx.avg_tps:.1f}",            f"{tps_speedup:.1f}x"),
            ("Success rate",               f"{amx.success_rate:.0f}%",        f"{no_amx.success_rate:.0f}%",      ""),
        ]
        print(f"{'Metric':<24} {'AMX':>12} {'No-AMX':>14} {'Speedup':>12}")
        print("-" * 64)
        for label, a, b, s in rows:
            print(f"{label:<24} {a:>12} {b:>14} {s:>12}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM AMX vs No-AMX containers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with default prompt
  python query_vllm_amx.py

  # Custom prompt, 5 warmup+bench runs
  python query_vllm_amx.py --prompt "What is RDMA?" --runs 5

  # Custom endpoints
  python query_vllm_amx.py \\
      --amx-url http://node1:8000 \\
      --no-amx-url http://node2:8001 \\
      --model meta-llama/Llama-3.1-8B-Instruct

  # List sample prompts and exit
  python query_vllm_amx.py --list-prompts
        """,
    )
    parser.add_argument("--amx-url",    default=DEFAULT_AMX_URL,
                        help=f"AMX container URL (default: {DEFAULT_AMX_URL})")
    parser.add_argument("--no-amx-url", default=DEFAULT_NO_AMX_URL,
                        help=f"No-AMX container URL (default: {DEFAULT_NO_AMX_URL})")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt",     default=SAMPLE_PROMPTS[0],
                        help="Prompt to send to both containers")
    parser.add_argument("--max-tokens", default=DEFAULT_MAX_TOKENS, type=int,
                        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--runs",       default=DEFAULT_RUNS, type=int,
                        help=f"Number of benchmark runs per container (default: {DEFAULT_RUNS})")
    parser.add_argument("--skip-health", action="store_true",
                        help="Skip health check before benchmarking")
    parser.add_argument("--list-prompts", action="store_true",
                        help="Print sample prompts and exit")
    parser.add_argument("--cooldown", default=2, type=int,
                        help="Seconds to wait between runs to reduce cross-container interference (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_prompts:
        print("Sample prompts:")
        for i, p in enumerate(SAMPLE_PROMPTS, 1):
            print(f"  {i}. {p}")
        return

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      vLLM AMX vs No-AMX Benchmark                    ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"  AMX endpoint:    {args.amx_url}")
    print(f"  No-AMX endpoint: {args.no_amx_url}")
    print(f"  Model:           {args.model}")
    print(f"  Prompt:          {args.prompt[:60]}...")
    print(f"  Runs per image:  {args.runs}")
    print(f"  Max tokens:      {args.max_tokens}")
    print()

    # Health checks
    if not args.skip_health:
        print("Checking container health...")
        for label, url in [("AMX", args.amx_url), ("No-AMX", args.no_amx_url)]:
            ok, info = check_health(url, label)
            status = "✅ READY" if ok else "❌ UNREACHABLE"
            print(f"  {label:8s} ({url}): {status}  {info}")
        print()

    # Run benchmarks
    print("--- AMX Container ---")
    amx_result = run_benchmark(
        url=args.amx_url,
        label="AMX (AMX-BF16 + SGL kernels)",
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        num_runs=args.runs,
        cooldown=args.cooldown,
    )

    print("\n--- No-AMX Container ---")
    no_amx_result = run_benchmark(
        url=args.no_amx_url,
        label="No-AMX (AVX-512 BF16 only)",
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        num_runs=args.runs,
        cooldown=args.cooldown,
    )

    # Print comparison
    print_comparison(amx_result, no_amx_result, args.prompt)


if __name__ == "__main__":
    main()
