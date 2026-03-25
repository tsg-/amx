"""
PWI-Flask-2vLLM-v2.py
---------------------
Improved AMX vs No-AMX demo app incorporating lessons from query_vllm_amx.py:
  - Long-context prompts so prefill (compute-bound GEMM) dominates and AMX advantage is visible
  - Cache busting per run (unique prefix) to defeat vLLM prefix caching
  - stream_options include_usage for real prompt token counts
  - Multiple runs with Avg TTFT, P95 TTFT, Prefill tok/s, Decode tok/s
  - max_tokens=50 (RAG sweet spot: prefill ~46% of total, ~3x end-to-end speedup)
  - Sequential execution: AMX runs first, then No-AMX — avoids DRAM contention noise

Usage:
    python3 PWI-Flask-2vLLM-v2.py
    open http://localhost:5001
"""

from flask import Flask, render_template_string, request, Response, stream_with_context
import requests
import time
import json
import statistics

app = Flask(__name__)

VLLM_SERVICES = {
    "amx":   "http://localhost:8000/v1/chat/completions",
    "noamx": "http://localhost:8001/v1/chat/completions",
}

VLLM_MODEL  = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_RUNS       = 3
DEFAULT_MAX_TOKENS = 50
DEFAULT_COOLDOWN   = 2   # seconds between runs

# Detect CPU model at startup
def _get_cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    # Replace (R) and (TM) with proper Unicode symbols
                    name = name.replace("(R)", "\u00ae").replace("(TM)", "\u2122")
                    return name
    except Exception:
        pass
    return "Unknown CPU"

CPU_NAME = _get_cpu_name()

# ---------------------------------------------------------------------------
# Long-context questions — prefill must be substantial for AMX advantage
# to show in TTFT. Each question prepends a ~350-token technical document.
# ---------------------------------------------------------------------------
CONTEXT_DOC = """Intel Advanced Matrix Extensions (AMX\u2122) is an x86 ISA extension introduced in
Intel Xeon Scalable processors (Sapphire Rapids and later). AMX\u2122 adds eight 2D tile registers
(each 16 rows x 64 bytes = 1 KB) and TMUL instructions that perform a full 16x16 BF16
matrix multiply-accumulate in a single instruction, delivering up to 2048 BF16 MACs per
cycle per core. This is a significant improvement over AVX-512 BF16 VNNI, which processes
4x16 BF16 dot products per instruction.

In transformer-based LLM inference there are two distinct phases. The prefill phase processes
the entire input prompt in parallel: it computes query, key, and value projections (large GEMMs
over hidden_size x sequence_length), self-attention scores, and all feed-forward network layers.
Because the full sequence is processed simultaneously, prefill consists of large matrix
multiplications — exactly the compute-bound workload AMX\u2122 tile instructions are designed for.

The decode phase generates one token at a time via autoregressive sampling. Each step performs
a matrix-vector multiply against the full weight matrices, which is memory-bandwidth bound:
the bottleneck is loading weights from DRAM, not arithmetic throughput. AMX\u2122 provides no
benefit here; both AMX\u2122 and non-AMX\u2122 cores are equally constrained by DRAM bandwidth.

As a result, AMX\u2122 delivers its advantage primarily through TTFT (time to first token), which
directly measures prefill latency. For workloads with long input contexts and short outputs —
such as retrieval-augmented generation (RAG), document summarisation, code review, and
classification — prefill is a large fraction of total request time. Benchmarks on
Granite-3.3-8B with ~2600-token prompts show approximately 6x faster TTFT and 3x faster
total time with AMX\u2122 vs AVX-512 BF16 at 50 output tokens.\n\n"""

QUESTION_LABELS = [
    "Why does AMX\u2122 improve TTFT but not decode throughput?",
    "What LLM workloads benefit most from AMX\u2122?",
    "Compare prefill vs decode phases in transformer inference.",
    "What is TTFT and why does it matter to users?",
    "How do AMX\u2122 tile registers differ from AVX-512 VNNI?",
]

QUESTIONS = [
    CONTEXT_DOC + "In 2-3 sentences, explain why AMX improves TTFT but not decode throughput.",
    CONTEXT_DOC + "In 2-3 sentences, what LLM inference workloads benefit most from AMX?",
    CONTEXT_DOC + "In 2-3 sentences, compare the prefill and decode phases of transformer inference.",
    CONTEXT_DOC + "In 2-3 sentences, explain what TTFT measures and why it matters to end users.",
    CONTEXT_DOC + "In 2-3 sentences, describe how AMX tile registers differ from AVX-512 VNNI.",
]

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AMX&#8482; on vLLM</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background-color: #0071C5; color: white; }
        .page-wrap { width: 1080px; }
        h1 { color: white; }
        select, button, input[type=number] { padding: 8px; margin-top: 8px; font-size: 1em; font-family: Arial, sans-serif; }
        button { cursor: pointer; background: #33ff56; font-weight: bold; border: none; border-radius: 6px; padding: 10px 20px; }
        button:disabled { background: #888; cursor: not-allowed; }

        .rounded { border-radius: 15px; }

        /* Side-by-side columns */
        .cols { display: flex; gap: 20px; margin-top: 16px; width: 100%; }
        .col  { flex: 1; }
        .col h2 { margin: 0 0 8px 0; font-size: 1.1em; }

        /* Streaming text box */
        .response-box {
            background: #d0e8ff;
            color: #000;
            border: 1px solid #aaa;
            border-radius: 6px;
            padding: 10px;
            min-height: 80px;
            max-height: 160px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-size: 0.9em;
        }

        /* Metric cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }
        .metric-card {
            background: rgba(255,255,255,0.15);
            border-radius: 8px;
            padding: 8px 12px;
            text-align: center;
        }
        .metric-label { font-size: 0.75em; opacity: 0.85; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #ffff00; }
        .metric-value.neutral { color: #ffffff; }

        /* Info line */
        .info-line {
            margin-top: 8px;
            font-size: 0.8em;
            opacity: 0.85;
        }

        /* Run progress */
        .run-progress { font-size: 0.85em; margin-top: 6px; min-height: 20px; }

        /* Speedup banner */
        #speedup-banner {
            display: none;
            margin-top: 20px;
            width: 100%;
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 26px 20px;
            font-size: 1em;
        }
        #speedup-banner h3 { margin: 0 0 8px 0; }
        .speedup-grid { display: flex; gap: 20px; }
        .speedup-item { text-align: center; flex: 1; }
        .speedup-num { font-size: 2em; font-weight: bold; color: #33ff56; }
        .speedup-desc { font-size: 1em; opacity: 0.85; font-weight: bold; }

        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .blink { animation: blink 1s infinite; }

        /* controls row */
        .controls { display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
        .ctrl-group label { display: block; font-size: 0.9em; margin-bottom: 2px; font-family: Arial, sans-serif; }
        .ctrl-group input[type=number] { width: 60px; }
    </style>
</head>
<body>
<div class="page-wrap">
    <table width="100%"><tr>
        <td width="200" align="left" valign="middle"><img src="/static/intel-logo.jpg" width="200" class="rounded" alt="Intel Logo"></td>
        <td align="center">
            <h1 style="margin-bottom:4px;">
                <b style="font-size:1.4em;">AMX&#8482; DEMO</b><br>
                <b style="font-size:0.9em;">Intel Advanced Matrix eXtensions with vLLM</b>
            </h1>
            <p style="font-size:1.2em; margin-top:6px; opacity:0.95;">
                <table style="margin:8px auto 0 auto; font-size:1.05em; border-collapse:collapse;">
                    <tr>
                        <td style="text-align:right; padding:2px 8px 2px 0; white-space:nowrap; width:160px;"><b>Model in use:</b></td>
                        <td style="text-align:left; padding:2px 0; color:#33ff56; white-space:nowrap;">{{ model }}</td>
                    </tr>
                    <tr>
                        <td style="text-align:right; padding:2px 8px 2px 0; white-space:nowrap;"><b>CPU under Test:</b></td>
                        <td style="text-align:left; padding:2px 0; color:#33ff56; white-space:nowrap;">{{ cpu }}</td>
                    </tr>
                </table>
            </p>
        </td>
        <td width="160" align="right" valign="middle"><img src="/static/vLLM-logo.jpg" width="160" class="rounded" alt="vLLM Logo"></td>
    </tr></table>

    <div class="controls">
        <div class="ctrl-group">
            <label>Question:</label>
            <select id="question" style="width:340px">
                <option value="">-- Select a question --</option>
                {% for label, q in question_pairs %}
                    <option value="{{ q }}">{{ label }}</option>
                {% endfor %}
            </select>
        </div>
        <div class="ctrl-group">
            <label>Iterations:</label>
            <input type="number" id="runs" value="{{ default_runs }}" min="1" max="10">
        </div>
        <div class="ctrl-group">
            <label>Max tokens:</label>
            <input type="number" id="max_tokens" value="{{ default_max_tokens }}" min="1" max="512">
        </div>
        <div class="ctrl-group">
            <label>&nbsp;</label>
            <button id="askBtn">&#9654; RUN BENCHMARK</button>
        </div>
    </div>

    <div class="cols">
        <!-- AMX column -->
        <div class="col">
            <h2>&#9989; AMX&#8482; &mdash; AVX512 + Tile GEMM (Port 8000)</h2>
            <div id="text-amx" class="response-box">Waiting...</div>
            <div id="progress-amx" class="run-progress"></div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Avg TTFT</div>
                    <div class="metric-value" id="amx-avg-ttft">&mdash;</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">P95 TTFT</div>
                    <div class="metric-value" id="amx-p95-ttft">&mdash;</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Prefill throughput</div>
                    <div class="metric-value" id="amx-prefill">&mdash;</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Decode throughput</div>
                    <div class="metric-value neutral" id="amx-decode">&mdash;</div>
                </div>
            </div>
            <div class="info-line" id="amx-info">&nbsp;</div>
        </div>

        <!-- No-AMX column -->
        <div class="col">
            <h2>&#128994; No AMX &mdash; AVX-512 BF16 only (Port 8001)</h2>
            <div id="text-noamx" class="response-box">Waiting...</div>
            <div id="progress-noamx" class="run-progress"></div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Avg TTFT</div>
                    <div class="metric-value" id="noamx-avg-ttft">&mdash;</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">P95 TTFT</div>
                    <div class="metric-value" id="noamx-p95-ttft">&mdash;</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Prefill throughput</div>
                    <div class="metric-value" id="noamx-prefill">&mdash;</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Decode throughput</div>
                    <div class="metric-value neutral" id="noamx-decode">&mdash;</div>
                </div>
            </div>
            <div class="info-line" id="noamx-info">&nbsp;</div>
        </div>
    </div>

    <!-- Speedup summary banner -->
    <div id="speedup-banner">
        <h3>&#128640; AMX Speedup Summary</h3>
        <div class="speedup-grid">
            <div class="speedup-item">
                <div class="speedup-num" id="sp-ttft">&mdash;</div>
                <div class="speedup-desc">Avg TTFT speedup</div>
            </div>
            <div class="speedup-item">
                <div class="speedup-num" id="sp-p95">&mdash;</div>
                <div class="speedup-desc">P95 TTFT speedup</div>
            </div>
            <div class="speedup-item">
                <div class="speedup-num" id="sp-prefill">&mdash;</div>
                <div class="speedup-desc">Prefill throughput gain</div>
            </div>
            <div class="speedup-item">
                <div class="speedup-num" id="sp-decode" style="color:#ffffff">&mdash;</div>
                <div class="speedup-desc">Decode throughput<br>(memory-BW bound)</div>
            </div>
        </div>
    </div>

    <script>
    const btn = document.getElementById("askBtn");

    function ms(val) { return val !== null ? val.toFixed(0) + "ms" : "—"; }
    function tps(val) { return val !== null ? val.toFixed(0) + " tok/s" : "—"; }
    function speedupStr(a, b) {
        if (!a || !b) return "—";
        return (b / a).toFixed(1) + "x";
    }

    function setMetrics(prefix, summary) {
        document.getElementById(prefix+"-avg-ttft").textContent  = ms(summary.avg_ttft_ms);
        document.getElementById(prefix+"-p95-ttft").textContent  = ms(summary.p95_ttft_ms);
        document.getElementById(prefix+"-prefill").textContent   = tps(summary.avg_prefill_tps);
        document.getElementById(prefix+"-decode").textContent    = tps(summary.avg_decode_tps);
        document.getElementById(prefix+"-info").textContent =
            `Prompt tokens: ${summary.prompt_tokens} | ` +
            `Avg output tokens: ${summary.avg_output_tokens} | ` +
            `Success rate: ${summary.success_rate.toFixed(0)}% | ` +
            `Iterations: ${summary.runs}`;
    }

    function runBenchmark(service, question, runs, maxTokens, textEl, progressEl) {
        return new Promise((resolve) => {
            const url = `/benchmark?service=${encodeURIComponent(service)}` +
                        `&question=${encodeURIComponent(question)}` +
                        `&runs=${runs}&max_tokens=${maxTokens}`;
            const es = new EventSource(url);
            let summary = null;

            es.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.run_start !== undefined) {
                    progressEl.textContent = `Iteration ${data.run_start} / ${data.total_runs}...`;
                }
                if (data.token) {
                    // First token: clear the "Running..." placeholder
                    if (textEl.textContent === "Running...") textEl.textContent = "";
                    textEl.textContent += data.token;
                    textEl.scrollTop = textEl.scrollHeight;
                }
                if (data.run_metrics) {
                    const r = data.run_metrics;
                    const info = r.error
                        ? `Run ${r.run}: ERROR — ${r.error}`
                        : `Run ${r.run}: TTFT=${r.ttft_ms.toFixed(0)}ms  Prefill=${r.prefill_tps.toFixed(0)}tok/s`;
                    progressEl.textContent = info;
                }
                if (data.summary) {
                    summary = data.summary;
                }
                if (data.done) {
                    es.close();
                    progressEl.textContent = "Done.";
                    resolve(summary);
                }
                if (data.error) {
                    es.close();
                    progressEl.textContent = "Error: " + data.error;
                    resolve(null);
                }
            };

            es.onerror = function() {
                es.close();
                progressEl.textContent = "Connection error.";
                resolve(null);
            };
        });
    }

    btn.addEventListener("click", async function() {
        const question = document.getElementById("question").value;
        const runs     = parseInt(document.getElementById("runs").value) || {{ default_runs }};
        const maxTok   = parseInt(document.getElementById("max_tokens").value) || {{ default_max_tokens }};

        if (!question) {
            alert("Please select a question.");
            return;
        }

        btn.disabled = true;
        btn.textContent = "Running...";
        document.getElementById("speedup-banner").style.display = "none";

        // Reset displays
        ["amx","noamx"].forEach(p => {
            document.getElementById("progress-"+p).textContent = "";
            document.getElementById(p+"-avg-ttft").textContent = "—";
            document.getElementById(p+"-p95-ttft").textContent = "—";
            document.getElementById(p+"-prefill").textContent = "—";
            document.getElementById(p+"-decode").textContent = "—";
            document.getElementById(p+"-info").textContent = "";
        });
        document.getElementById("text-amx").textContent = "Running...";
        document.getElementById("text-noamx").textContent = "Waiting...";

        // Run AMX first, then No-AMX (sequential = no DRAM contention)
        const amxSummary = await runBenchmark(
            "amx", question, runs, maxTok,
            document.getElementById("text-amx"),
            document.getElementById("progress-amx")
        );
        if (amxSummary) setMetrics("amx", amxSummary);

        document.getElementById("text-noamx").textContent = "Running...";
        const noamxSummary = await runBenchmark(
            "noamx", question, runs, maxTok,
            document.getElementById("text-noamx"),
            document.getElementById("progress-noamx")
        );
        if (noamxSummary) setMetrics("noamx", noamxSummary);

        // Show speedup banner
        if (amxSummary && noamxSummary) {
            document.getElementById("sp-ttft").textContent    = speedupStr(amxSummary.avg_ttft_ms,   noamxSummary.avg_ttft_ms);
            document.getElementById("sp-p95").textContent     = speedupStr(amxSummary.p95_ttft_ms,   noamxSummary.p95_ttft_ms);
            document.getElementById("sp-prefill").textContent = speedupStr(noamxSummary.avg_prefill_tps, amxSummary.avg_prefill_tps) + " higher";
            document.getElementById("sp-decode").textContent  = speedupStr(noamxSummary.avg_decode_tps,  amxSummary.avg_decode_tps);
            // Colour decode neutral if within 20%
            const decodeRatio = amxSummary.avg_decode_tps / (noamxSummary.avg_decode_tps || 1);
            document.getElementById("sp-decode").style.color =
                (decodeRatio > 0.8 && decodeRatio < 1.2) ? "#ffffff" : "#33ff56";
            document.getElementById("speedup-banner").style.display = "block";
        }

        btn.disabled = false;
        btn.textContent = "▶ RUN BENCHMARK";
    });
    </script>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(
        HTML_PAGE,
        question_pairs=list(zip(QUESTION_LABELS, QUESTIONS)),
        default_runs=DEFAULT_RUNS,
        default_max_tokens=DEFAULT_MAX_TOKENS,
        model=VLLM_MODEL,
        cpu=CPU_NAME,
    )


@app.route("/benchmark", methods=["GET"])
def benchmark():
    service_key = request.args.get("service", "amx")
    question    = request.args.get("question", "").strip()
    num_runs    = max(1, min(10, int(request.args.get("runs", DEFAULT_RUNS))))
    max_tokens  = max(1, min(512, int(request.args.get("max_tokens", DEFAULT_MAX_TOKENS))))

    if service_key not in VLLM_SERVICES:
        return "Invalid service", 400
    if not question:
        return "Invalid question", 400

    api_url = VLLM_SERVICES[service_key]

    def generate():
        ttft_list    = []
        prefill_list = []
        decode_list  = []
        success_count = 0
        prompt_tokens_last = 0
        output_tokens_total = 0

        run_id_base = int(time.time() * 1000)

        for run in range(num_runs):
            # Cache bust at front of message — defeats vLLM prefix caching
            bust = f"[uid:{run_id_base}_{run}] "
            busted_question = bust + question

            yield f"data: {json.dumps({'run_start': run + 1, 'total_runs': num_runs})}\n\n"

            t_start        = time.time()
            first_token_t  = None
            token_count    = 0
            prompt_tokens  = 0
            error          = None

            payload = {
                "model":    VLLM_MODEL,
                "messages": [{"role": "user", "content": busted_question}],
                "max_tokens":     max_tokens,
                "temperature":    0,
                "seed":           12345,
                "stream":         True,
                "stream_options": {"include_usage": True},
            }

            try:
                with requests.post(api_url, json=payload, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            dj = json.loads(data_str)
                            # Capture actual prompt token count from usage chunk
                            if dj.get("usage") and dj["usage"].get("prompt_tokens"):
                                prompt_tokens = dj["usage"]["prompt_tokens"]
                            token = ""
                            if dj.get("choices"):
                                token = dj["choices"][0]["delta"].get("content", "")
                            if token:
                                token_count += 1
                                if first_token_t is None:
                                    first_token_t = time.time()
                                # Only stream text on the last run
                                if run == num_runs - 1:
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                        except Exception:
                            continue
            except Exception as e:
                error = str(e)

            t_end = time.time()
            if first_token_t is None:
                first_token_t = t_end

            # Fallback prompt token estimate if API didn't return usage
            if prompt_tokens == 0:
                prompt_tokens = len(busted_question) // 4

            ttft_ms     = (first_token_t - t_start) * 1000
            total_ms    = (t_end - t_start) * 1000
            prefill_tps = prompt_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0
            decode_tps  = token_count / (total_ms / 1000) if total_ms > 0 else 0

            if not error:
                ttft_list.append(ttft_ms)
                prefill_list.append(prefill_tps)
                decode_list.append(decode_tps)
                success_count += 1
                prompt_tokens_last  = prompt_tokens
                output_tokens_total += token_count

            run_metrics = {
                "run":         run + 1,
                "ttft_ms":     ttft_ms,
                "total_ms":    total_ms,
                "prefill_tps": prefill_tps,
                "decode_tps":  decode_tps,
                "prompt_tokens":  prompt_tokens,
                "output_tokens":  token_count,
                "error":          error,
            }
            yield f"data: {json.dumps({'run_metrics': run_metrics})}\n\n"

            if run < num_runs - 1:
                time.sleep(DEFAULT_COOLDOWN)

        # Aggregate
        def avg(lst):
            return statistics.mean(lst) if lst else 0.0

        def p95(lst):
            if not lst:
                return 0.0
            s = sorted(lst)
            return s[min(int(len(s) * 0.95), len(s) - 1)]

        summary = {
            "avg_ttft_ms":      avg(ttft_list),
            "p95_ttft_ms":      p95(ttft_list),
            "avg_prefill_tps":  avg(prefill_list),
            "avg_decode_tps":   avg(decode_list),
            "prompt_tokens":    prompt_tokens_last,
            "avg_output_tokens": output_tokens_total // success_count if success_count else 0,
            "success_rate":     success_count / num_runs * 100,
            "runs":             num_runs,
        }
        yield f"data: {json.dumps({'done': True, 'summary': summary})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
