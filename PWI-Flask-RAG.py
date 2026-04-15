"""
PWI-Flask-RAG.py
----------------
Full RAG demo with AMX performance comparison.

Three sections:
  1. Indexing Demo   — show AMX vs no-AMX embedding throughput while indexing corpus
  2. RAG Query Demo  — ask a question, see retrieved docs + LLM answer with AMX metrics
  3. LLM Inference   — existing direct LLM comparison (no retrieval)

Usage:
    python3 PWI-Flask-RAG.py
    open http://localhost:5002

Prerequisites:
    pip install flask requests pymilvus openai
    All services running via start_rag_demo.sh
    Corpus indexed via rag_index_amx.py
"""

from flask import Flask, render_template_string, request, Response, stream_with_context
import requests as _requests
import time
import json
import statistics
import threading

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------
VLLM_LLM_SERVICES = {
    "amx":   "http://localhost:8000/v1/chat/completions",
    "noamx": "http://localhost:8001/v1/chat/completions",
}
VLLM_EMBED_SERVICES = {
    "amx":   "http://localhost:8002/v1/embeddings",
    "noamx": "http://localhost:8003/v1/embeddings",
}
MILVUS_URL   = "http://localhost:9091"

LLM_MODEL   = "ibm-granite/granite-3.3-8b-instruct"
EMBED_MODEL = "BAAI/bge-m3"

DEFAULT_RUNS       = 3
DEFAULT_MAX_TOKENS = 50
DEFAULT_COOLDOWN   = 2
DEFAULT_TOP_K      = 5

COLLECTION_NAME = "amx_rag_demo"

# ---------------------------------------------------------------------------
# Pre-load corpus in memory (used for indexing demo)
# ---------------------------------------------------------------------------
try:
    from rag_corpus import build_corpus as _build_corpus
    CORPUS = _build_corpus()
except ImportError:
    CORPUS = []

# ---------------------------------------------------------------------------
# Lazy Milvus connection (connect on first use)
# ---------------------------------------------------------------------------
_milvus_collection = None
_milvus_lock = threading.Lock()

def get_milvus_collection():
    global _milvus_collection
    with _milvus_lock:
        if _milvus_collection is not None:
            return _milvus_collection
        try:
            from pymilvus import connections, Collection, utility
            connections.connect("default", host="localhost", port=19530)
            if utility.has_collection(COLLECTION_NAME):
                _milvus_collection = Collection(COLLECTION_NAME)
                _milvus_collection.load()
        except Exception:
            pass
        return _milvus_collection


def milvus_search(query_vector, top_k=5):
    """Return list of {title, topic, text, score} dicts."""
    coll = get_milvus_collection()
    if coll is None:
        return []
    try:
        results = coll.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["id", "title", "topic", "text"],
        )
        return [
            {
                "doc_id": hit.entity.get("id", ""),
                "title":  hit.entity.get("title", ""),
                "topic":  hit.entity.get("topic", ""),
                "text":   hit.entity.get("text", ""),
                "score":  round(float(hit.score), 4),
            }
            for hit in results[0]
        ]
    except Exception:
        return []


def get_embedding(url, text):
    """Return (vector, latency_ms) or (None, 0) on error."""
    t0 = time.perf_counter()
    try:
        resp = _requests.post(
            url,
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        vector = data["data"][0]["embedding"]
        return vector, (time.perf_counter() - t0) * 1000.0
    except Exception:
        return None, 0.0


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------
def _get_cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    return name.replace("(R)", "\u00ae").replace("(TM)", "\u2122")
    except Exception:
        pass
    return "Unknown CPU"

CPU_NAME = _get_cpu_name()

# ---------------------------------------------------------------------------
# Questions for each demo section
# ---------------------------------------------------------------------------
LLM_QUESTION_LABELS = [
    "Why does AMX™ improve TTFT but not decode throughput?",
    "What LLM workloads benefit most from AMX™?",
    "Compare prefill vs decode phases in transformer inference.",
    "What is TTFT and why does it matter to users?",
    "How do AMX™ tile registers differ from AVX-512 VNNI?",
]

CONTEXT_DOC = (
    "Intel Advanced Matrix Extensions (AMX™) is an x86 ISA extension introduced in "
    "Intel Xeon Scalable processors (Sapphire Rapids and later). AMX™ adds eight 2D tile registers "
    "(each 16 rows × 64 bytes = 1 KB) and TMUL instructions that perform a full 16×16 BF16 "
    "matrix multiply-accumulate in a single instruction, delivering up to 2048 BF16 MACs per "
    "cycle per core — ~8× more than AVX-512 BF16 VNNI.\n\n"
    "In transformer-based LLM inference the prefill phase processes the entire input prompt in "
    "parallel via large GEMMs — exactly what AMX™ is designed for. The decode phase generates "
    "one token at a time (matrix-vector multiply), which is memory-bandwidth bound; AMX™ provides "
    "no benefit there. Benchmarks on Granite-3.3-8B with ~2600-token prompts show approximately "
    "6× faster TTFT and 3× faster total time with AMX™ vs AVX-512 BF16 at 50 output tokens.\n\n"
)

LLM_QUESTIONS = [
    CONTEXT_DOC + "In 2-3 sentences, explain why AMX improves TTFT but not decode throughput.",
    CONTEXT_DOC + "In 2-3 sentences, what LLM inference workloads benefit most from AMX?",
    CONTEXT_DOC + "In 2-3 sentences, compare the prefill and decode phases of transformer inference.",
    CONTEXT_DOC + "In 2-3 sentences, explain what TTFT measures and why it matters to end users.",
    CONTEXT_DOC + "In 2-3 sentences, describe how AMX tile registers differ from AVX-512 VNNI.",
]

RAG_QUESTIONS = [
    "What is Intel AMX and how does it accelerate LLM inference?",
    "Why does AMX improve time-to-first-token but not decode throughput?",
    "What LLM serving workloads benefit most from Intel AMX?",
    "How does a Milvus HNSW index work?",
    "Compare BF16 and INT8 quantization for CPU inference.",
    "Explain how oneDNN dispatches AMX vs AVX-512 kernels.",
    "What is retrieval-augmented generation and how does it help LLMs?",
    "How does the AMX TDPBF16PS instruction differ from AVX-512 VDPBF16PS?",
]

RAG_SYSTEM_PROMPT = (
    "You are a helpful technical assistant answering questions based strictly on the "
    "provided context documents. If the context does not contain sufficient information "
    "to answer, say so clearly. Do not fabricate information."
)

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AMX RAG Demo</title>
  <style>
    :root {
      --intel-blue: #0071C5;
      --intel-dark: #003C71;
      --intel-light: #E6F2FB;
      --amx-green:  #00C853;
      --noamx-gold: #FFB300;
    }
    * { box-sizing: border-box; }
    body {
      font-family: Arial, sans-serif;
      background: var(--intel-blue);
      margin: 0; padding: 20px;
      color: white;
    }
    h1 { margin: 0 0 4px; font-size: 1.5em; }
    .subtitle { opacity: .85; font-size: .9em; margin-bottom: 16px; }
    .cpu-badge {
      display: inline-block;
      background: rgba(255,255,255,.15);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: .8em;
      margin-bottom: 16px;
    }

    /* Tab bar */
    .tabs { display: flex; gap: 4px; margin-bottom: 0; }
    .tab {
      padding: 10px 22px;
      border-radius: 8px 8px 0 0;
      cursor: pointer;
      background: rgba(255,255,255,.15);
      border: none;
      color: white;
      font-size: .95em;
      font-weight: bold;
      transition: background .2s;
    }
    .tab.active { background: white; color: var(--intel-blue); }
    .tab:hover:not(.active) { background: rgba(255,255,255,.25); }

    /* Card */
    .card {
      background: white;
      color: #222;
      border-radius: 0 8px 8px 8px;
      padding: 20px;
    }
    .section { display: none; }
    .section.active { display: block; }

    /* Controls */
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 16px; }
    .ctrl-group { display: flex; flex-direction: column; gap: 4px; }
    .ctrl-group label { font-size: .8em; font-weight: bold; color: #555; }
    select, button, input[type=number] {
      padding: 8px 12px;
      border: 1px solid #ccc;
      border-radius: 6px;
      font-size: .9em;
    }
    button.run-btn {
      background: var(--intel-blue);
      color: white;
      border: none;
      cursor: pointer;
      font-weight: bold;
      padding: 8px 20px;
      border-radius: 6px;
      transition: background .2s;
    }
    button.run-btn:hover { background: var(--intel-dark); }
    button.run-btn:disabled { background: #aaa; cursor: not-allowed; }

    /* Metric bars */
    .metric-row {
      display: grid;
      grid-template-columns: 200px 1fr 1fr 100px;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
      font-size: .9em;
    }
    .bar-wrap { background: #f0f0f0; border-radius: 4px; height: 26px; position: relative; overflow: hidden; }
    .bar {
      height: 100%;
      border-radius: 4px;
      transition: width .6s ease;
      display: flex;
      align-items: center;
      padding-left: 8px;
      font-size: .8em;
      font-weight: bold;
      color: white;
      white-space: nowrap;
    }
    .bar.amx   { background: var(--amx-green); }
    .bar.noamx { background: var(--noamx-gold); color: #333; }
    .speedup-badge {
      background: var(--intel-blue);
      color: white;
      border-radius: 12px;
      padding: 3px 10px;
      font-size: .85em;
      font-weight: bold;
      text-align: center;
    }

    /* Response boxes */
    .response-wrap {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .response-box {
      border-radius: 6px;
      padding: 10px;
      min-height: 80px;
      font-size: .85em;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-y: auto;
      max-height: 220px;
    }
    .response-box.amx   { background: #E8F5E9; border: 1px solid var(--amx-green); }
    .response-box.noamx { background: #FFF8E1; border: 1px solid var(--noamx-gold); }
    .resp-label { font-size: .8em; font-weight: bold; margin-bottom: 4px; }
    .resp-label.amx   { color: #2E7D32; }
    .resp-label.noamx { color: #F57F17; }

    /* Chunks */
    .chunk-list { font-size: .8em; margin-top: 8px; }
    .chunk-item {
      background: #f8f9fa;
      border-left: 3px solid var(--intel-blue);
      padding: 6px 10px;
      margin-bottom: 4px;
      border-radius: 0 4px 4px 0;
    }
    .chunk-score { color: var(--intel-blue); font-weight: bold; }

    /* Status */
    .status { font-size: .85em; color: #555; margin-top: 8px; min-height: 1.5em; }
    .status.running { color: var(--intel-blue); font-weight: bold; }
    .status.done    { color: #2E7D32; }
    .status.error   { color: #c00; }

    /* Legend */
    .legend { display: flex; gap: 16px; margin-bottom: 12px; font-size: .85em; }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
    .legend-dot.amx   { background: var(--amx-green); }
    .legend-dot.noamx { background: var(--noamx-gold); }

    /* Progress bar (indexing) */
    #indexProgress {
      display: none;
      height: 8px;
      background: #f0f0f0;
      border-radius: 4px;
      margin: 8px 0;
      overflow: hidden;
    }
    #indexProgressBar {
      height: 100%;
      background: var(--intel-blue);
      width: 0%;
      transition: width .3s;
      border-radius: 4px;
    }

    @media (max-width: 700px) {
      .response-wrap { grid-template-columns: 1fr; }
      .metric-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>🔷 Intel AMX™ — RAG &amp; VectorDB Performance Demo</h1>
  <div class="subtitle">
    Quantifies AMX™ benefit across the full RAG pipeline: embedding throughput,
    vector indexing, and LLM inference — same hardware, same model, AMX on vs off.
  </div>
  <div class="cpu-badge">🖥 {{ cpu_name }}</div>

  <div class="tabs">
    <button class="tab active" onclick="showTab('indexing',this)">① Embedding &amp; Indexing</button>
    <button class="tab"        onclick="showTab('rag',this)">② RAG Query</button>
    <button class="tab"        onclick="showTab('llm',this)">③ LLM Inference</button>
  </div>

  <div class="card">

    <!-- ======================= SECTION 1: INDEXING ======================= -->
    <div id="tab-indexing" class="section active">
      <h2 style="margin-top:0; color: var(--intel-blue);">Embedding &amp; Corpus Indexing</h2>
      <p style="font-size:.9em; color:#555;">
        Encodes the {{ corpus_size }}-document corpus using the AMX and no-AMX embedding endpoints.
        Demonstrates how AMX tile instructions accelerate transformer-based embedding generation
        (the same GEMM speedup as LLM prefill).
      </p>

      <div class="controls">
        <div class="ctrl-group">
          <label>Sample size</label>
          <input type="number" id="idxSampleSize" value="20" min="1" max="{{ corpus_size }}"
                 style="width:90px">
        </div>
        <button class="run-btn" id="idxRunBtn" onclick="runIndexBenchmark()">
          ▶ Run Benchmark
        </button>
      </div>

      <div id="indexProgress"><div id="indexProgressBar"></div></div>
      <div class="status" id="idxStatus"></div>

      <div class="legend" style="margin-top:12px;">
        <div class="legend-item"><div class="legend-dot amx"></div> AMX™ (AVX512_CORE_AMX)</div>
        <div class="legend-item"><div class="legend-dot noamx"></div> No-AMX (AVX512_CORE_BF16)</div>
      </div>

      <div id="idxMetrics" style="margin-top:8px;"></div>
    </div>

    <!-- ======================= SECTION 2: RAG QUERY ======================= -->
    <div id="tab-rag" class="section">
      <h2 style="margin-top:0; color: var(--intel-blue);">RAG Query Pipeline</h2>
      <p style="font-size:.9em; color:#555;">
        Embeds your question, retrieves relevant documents from Milvus, and generates
        a grounded answer. Shows AMX speedup at every stage.
      </p>

      <div class="controls">
        <div class="ctrl-group" style="flex:1; min-width:300px;">
          <label>Question</label>
          <select id="ragQuestion">
            {% for q in rag_questions %}
            <option value="{{ q }}">{{ q }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="ctrl-group">
          <label>Top-k docs</label>
          <input type="number" id="ragTopK" value="5" min="1" max="20" style="width:70px">
        </div>
        <div class="ctrl-group">
          <label>Max tokens</label>
          <input type="number" id="ragMaxTokens" value="50" min="1" max="500" style="width:80px">
        </div>
        <div class="ctrl-group">
          <label>Runs</label>
          <input type="number" id="ragRuns" value="1" min="1" max="5" style="width:60px">
        </div>
        <button class="run-btn" id="ragRunBtn" onclick="runRAGQuery()">
          ▶ Ask
        </button>
      </div>

      <div class="status" id="ragStatus"></div>

      <div id="ragChunks" class="chunk-list"></div>

      <div class="legend" style="margin-top:12px;">
        <div class="legend-item"><div class="legend-dot amx"></div> AMX™</div>
        <div class="legend-item"><div class="legend-dot noamx"></div> No-AMX</div>
      </div>
      <div id="ragMetrics" style="margin-top:8px;"></div>

      <div class="response-wrap" id="ragResponses" style="display:none;">
        <div>
          <div class="resp-label amx">✅ AMX™ Answer</div>
          <div class="response-box amx" id="ragAmxAnswer"></div>
        </div>
        <div>
          <div class="resp-label noamx">⚡ No-AMX Answer</div>
          <div class="response-box noamx" id="ragNoamxAnswer"></div>
        </div>
      </div>
    </div>

    <!-- ======================= SECTION 3: LLM INFERENCE ======================= -->
    <div id="tab-llm" class="section">
      <h2 style="margin-top:0; color: var(--intel-blue);">Direct LLM Inference</h2>
      <p style="font-size:.9em; color:#555;">
        Sends a long-context prompt directly to the LLM (no retrieval). Shows the baseline
        AMX prefill speedup — the engine driving the TTFT improvement in the RAG demo above.
      </p>

      <div class="controls">
        <div class="ctrl-group" style="flex:1; min-width:300px;">
          <label>Question</label>
          <select id="llmQuestion">
            {% for label, q in llm_questions %}
            <option value="{{ q }}">{{ label }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="ctrl-group">
          <label>Runs</label>
          <input type="number" id="llmRuns" value="3" min="1" max="10" style="width:60px">
        </div>
        <div class="ctrl-group">
          <label>Max tokens</label>
          <input type="number" id="llmMaxTokens" value="50" min="1" max="500" style="width:80px">
        </div>
        <button class="run-btn" id="llmRunBtn" onclick="runLLMBenchmark()">
          ▶ Run
        </button>
      </div>

      <div class="status" id="llmStatus"></div>

      <div class="legend" style="margin-top:12px;">
        <div class="legend-item"><div class="legend-dot amx"></div> AMX™</div>
        <div class="legend-item"><div class="legend-dot noamx"></div> No-AMX</div>
      </div>
      <div id="llmMetrics" style="margin-top:8px;"></div>

      <div class="response-wrap" id="llmResponses" style="display:none;">
        <div>
          <div class="resp-label amx">✅ AMX™ Response</div>
          <div class="response-box amx" id="llmAmxAnswer"></div>
        </div>
        <div>
          <div class="resp-label noamx">⚡ No-AMX Response</div>
          <div class="response-box noamx" id="llmNoamxAnswer"></div>
        </div>
      </div>
    </div>

  </div><!-- /.card -->

  <script>
  // ---------------------------------------------------------------------------
  // Tab switching
  // ---------------------------------------------------------------------------
  function showTab(name, btn) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ---------------------------------------------------------------------------
  // Metric bar rendering
  // ---------------------------------------------------------------------------
  function renderMetrics(containerId, rows) {
    // rows: [{label, amx, noamx, unit, lowerBetter}]
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    rows.forEach(row => {
      const max = Math.max(row.amx, row.noamx) || 1;
      const amxPct   = (row.amx   / max * 100).toFixed(1);
      const noamxPct = (row.noamx / max * 100).toFixed(1);
      const speedup  = row.lowerBetter
        ? (row.noamx / row.amx).toFixed(1)
        : (row.amx   / row.noamx).toFixed(1);

      const div = document.createElement('div');
      div.className = 'metric-row';
      div.innerHTML = `
        <div style="font-size:.85em; font-weight:bold; color:#333;">${row.label}</div>
        <div>
          <div style="font-size:.75em; color:#2E7D32; margin-bottom:2px;">AMX™</div>
          <div class="bar-wrap">
            <div class="bar amx" style="width:${amxPct}%">
              ${row.amx.toFixed(row.decimals??1)}${row.unit}
            </div>
          </div>
        </div>
        <div>
          <div style="font-size:.75em; color:#F57F17; margin-bottom:2px;">No-AMX</div>
          <div class="bar-wrap">
            <div class="bar noamx" style="width:${noamxPct}%">
              ${row.noamx.toFixed(row.decimals??1)}${row.unit}
            </div>
          </div>
        </div>
        <div class="speedup-badge">${speedup}×</div>
      `;
      container.appendChild(div);
    });
  }

  // ---------------------------------------------------------------------------
  // Section 1: Embedding/Indexing benchmark
  // ---------------------------------------------------------------------------
  function runIndexBenchmark() {
    const sampleSize = parseInt(document.getElementById('idxSampleSize').value) || 20;
    const btn = document.getElementById('idxRunBtn');
    btn.disabled = true;
    document.getElementById('idxStatus').className = 'status running';
    document.getElementById('idxStatus').textContent = 'Running embedding benchmark...';
    document.getElementById('idxMetrics').innerHTML = '';
    document.getElementById('indexProgress').style.display = 'block';
    document.getElementById('indexProgressBar').style.width = '0%';

    const evtSource = new EventSource(`/index_benchmark?sample=${sampleSize}`);
    let prog = 0;
    evtSource.onmessage = (ev) => {
      const data = JSON.parse(ev.data);

      if (data.progress !== undefined) {
        prog = data.progress;
        document.getElementById('indexProgressBar').style.width = prog + '%';
        document.getElementById('idxStatus').textContent = data.message || '';
      }

      if (data.done) {
        evtSource.close();
        btn.disabled = false;
        document.getElementById('indexProgressBar').style.width = '100%';
        document.getElementById('idxStatus').className = 'status done';
        document.getElementById('idxStatus').textContent =
          `✅ Done — ${data.docs} docs embedded`;

        renderMetrics('idxMetrics', [
          { label: 'Avg latency / doc (ms)', amx: data.amx_avg_ms,  noamx: data.noamx_avg_ms,  unit: 'ms',      lowerBetter: true,  decimals: 1 },
          { label: 'Throughput (docs/sec)',  amx: data.amx_dps,     noamx: data.noamx_dps,     unit: ' d/s',    lowerBetter: false, decimals: 2 },
          { label: 'Total indexing time (s)',amx: data.amx_total_s, noamx: data.noamx_total_s, unit: 's',       lowerBetter: true,  decimals: 1 },
        ]);
      }

      if (data.error) {
        evtSource.close();
        btn.disabled = false;
        document.getElementById('idxStatus').className = 'status error';
        document.getElementById('idxStatus').textContent = 'Error: ' + data.error;
      }
    };
    evtSource.onerror = () => {
      evtSource.close();
      btn.disabled = false;
      document.getElementById('idxStatus').className = 'status error';
      document.getElementById('idxStatus').textContent = 'Connection error.';
    };
  }

  // ---------------------------------------------------------------------------
  // Section 2: RAG Query
  // ---------------------------------------------------------------------------
  function runRAGQuery() {
    const question = document.getElementById('ragQuestion').value;
    const topK     = parseInt(document.getElementById('ragTopK').value) || 5;
    const maxTok   = parseInt(document.getElementById('ragMaxTokens').value) || 50;
    const runs     = parseInt(document.getElementById('ragRuns').value) || 1;
    const btn = document.getElementById('ragRunBtn');
    btn.disabled = true;
    document.getElementById('ragStatus').className  = 'status running';
    document.getElementById('ragStatus').textContent = 'Embedding query and retrieving...';
    document.getElementById('ragMetrics').innerHTML = '';
    document.getElementById('ragChunks').innerHTML  = '';
    document.getElementById('ragResponses').style.display = 'none';
    document.getElementById('ragAmxAnswer').textContent   = '';
    document.getElementById('ragNoamxAnswer').textContent = '';

    const params = new URLSearchParams({ question, top_k: topK, max_tokens: maxTok, runs });
    const evtSource = new EventSource('/rag_stream?' + params);

    evtSource.onmessage = (ev) => {
      const data = JSON.parse(ev.data);

      if (data.status) {
        document.getElementById('ragStatus').textContent = data.status;
      }
      if (data.chunks) {
        const el = document.getElementById('ragChunks');
        el.innerHTML = '<b style="font-size:.8em;color:#555;">📚 Retrieved documents:</b>';
        data.chunks.forEach(c => {
          el.innerHTML += `<div class="chunk-item">
            <span class="chunk-score">[${c.score}]</span>
            <b>${c.title}</b>
            <span style="color:#888;font-size:.9em;"> (${c.topic})</span>
          </div>`;
        });
      }
      if (data.amx_token) {
        document.getElementById('ragResponses').style.display = 'grid';
        document.getElementById('ragAmxAnswer').textContent += data.amx_token;
      }
      if (data.noamx_token) {
        document.getElementById('ragNoamxAnswer').textContent += data.noamx_token;
      }
      if (data.done) {
        evtSource.close();
        btn.disabled = false;
        document.getElementById('ragStatus').className  = 'status done';
        document.getElementById('ragStatus').textContent = '✅ Complete';

        renderMetrics('ragMetrics', [
          { label: 'Query embed (ms)',      amx: data.amx_embed_ms,  noamx: data.noamx_embed_ms,  unit: 'ms', lowerBetter: true,  decimals: 1 },
          { label: 'LLM TTFT (ms)',         amx: data.amx_ttft_ms,   noamx: data.noamx_ttft_ms,   unit: 'ms', lowerBetter: true,  decimals: 1 },
          { label: 'Prefill tok/s',         amx: data.amx_pfill_tps, noamx: data.noamx_pfill_tps, unit: ' t/s', lowerBetter: false, decimals: 0 },
          { label: 'LLM total time (ms)',   amx: data.amx_total_ms,  noamx: data.noamx_total_ms,  unit: 'ms', lowerBetter: true,  decimals: 1 },
          { label: 'End-to-end RAG (ms)',   amx: data.amx_e2e_ms,    noamx: data.noamx_e2e_ms,    unit: 'ms', lowerBetter: true,  decimals: 1 },
        ]);
      }
      if (data.error) {
        evtSource.close();
        btn.disabled = false;
        document.getElementById('ragStatus').className  = 'status error';
        document.getElementById('ragStatus').textContent = 'Error: ' + data.error;
      }
    };
    evtSource.onerror = () => {
      evtSource.close();
      btn.disabled = false;
      document.getElementById('ragStatus').className  = 'status error';
      document.getElementById('ragStatus').textContent = 'Connection error.';
    };
  }

  // ---------------------------------------------------------------------------
  // Section 3: LLM benchmark (sequential: AMX first, then no-AMX)
  // ---------------------------------------------------------------------------
  function runLLMBenchmark() {
    const question = document.getElementById('llmQuestion').value;
    const runs     = parseInt(document.getElementById('llmRuns').value) || 3;
    const maxTok   = parseInt(document.getElementById('llmMaxTokens').value) || 50;
    const btn = document.getElementById('llmRunBtn');
    btn.disabled = true;
    document.getElementById('llmStatus').className  = 'status running';
    document.getElementById('llmStatus').textContent = 'Running AMX runs...';
    document.getElementById('llmMetrics').innerHTML = '';
    document.getElementById('llmResponses').style.display = 'none';
    document.getElementById('llmAmxAnswer').textContent   = '';
    document.getElementById('llmNoamxAnswer').textContent = '';

    const params = new URLSearchParams({ question, runs, max_tokens: maxTok });
    const evtSource = new EventSource('/llm_stream?' + params);

    evtSource.onmessage = (ev) => {
      const data = JSON.parse(ev.data);

      if (data.status) {
        document.getElementById('llmStatus').textContent = data.status;
      }
      if (data.amx_token) {
        document.getElementById('llmResponses').style.display = 'grid';
        document.getElementById('llmAmxAnswer').textContent += data.amx_token;
      }
      if (data.noamx_token) {
        document.getElementById('llmNoamxAnswer').textContent += data.noamx_token;
      }
      if (data.done) {
        evtSource.close();
        btn.disabled = false;
        document.getElementById('llmStatus').className  = 'status done';
        document.getElementById('llmStatus').textContent = '✅ Complete';

        renderMetrics('llmMetrics', [
          { label: 'Avg TTFT (ms)',          amx: data.amx_ttft_ms,   noamx: data.noamx_ttft_ms,   unit: 'ms',   lowerBetter: true,  decimals: 1 },
          { label: 'Prefill throughput (t/s)',amx: data.amx_pfill_tps, noamx: data.noamx_pfill_tps, unit: ' t/s', lowerBetter: false, decimals: 0 },
          { label: 'Avg total time (ms)',    amx: data.amx_total_ms,  noamx: data.noamx_total_ms,  unit: 'ms',   lowerBetter: true,  decimals: 1 },
        ]);
      }
      if (data.error) {
        evtSource.close();
        btn.disabled = false;
        document.getElementById('llmStatus').className  = 'status error';
        document.getElementById('llmStatus').textContent = 'Error: ' + data.error;
      }
    };
    evtSource.onerror = () => {
      evtSource.close();
      btn.disabled = false;
      document.getElementById('llmStatus').className  = 'status error';
      document.getElementById('llmStatus').textContent = 'Connection error.';
    };
  }
  </script>
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
        cpu_name=CPU_NAME,
        corpus_size=len(CORPUS),
        rag_questions=RAG_QUESTIONS,
        llm_questions=list(zip(LLM_QUESTION_LABELS, LLM_QUESTIONS)),
    )


# ---------------------------------------------------------------------------
# /index_benchmark  — SSE stream for embedding benchmark
# ---------------------------------------------------------------------------
@app.route("/index_benchmark")
def index_benchmark():
    sample_size = min(int(request.args.get("sample", 20)), len(CORPUS))
    sample_docs = CORPUS[:sample_size]

    def generate():
        amx_latencies, noamx_latencies = [], []
        total = len(sample_docs)

        for i, doc in enumerate(sample_docs):
            # AMX embed
            _, amx_ms = get_embedding(VLLM_EMBED_SERVICES["amx"], doc["text"])
            amx_latencies.append(amx_ms)

            # No-AMX embed
            _, noamx_ms = get_embedding(VLLM_EMBED_SERVICES["noamx"], doc["text"])
            noamx_latencies.append(noamx_ms)

            progress = int((i + 1) / total * 95)
            yield (
                f"data: {json.dumps({'progress': progress, 'message': f'Embedded {i+1}/{total} docs...'})}\n\n"
            )

        amx_avg   = statistics.mean(amx_latencies)   if amx_latencies   else 0
        noamx_avg = statistics.mean(noamx_latencies) if noamx_latencies else 0
        amx_total   = sum(amx_latencies)   / 1000.0
        noamx_total = sum(noamx_latencies) / 1000.0
        amx_dps     = len(amx_latencies)   / amx_total   if amx_total   > 0 else 0
        noamx_dps   = len(noamx_latencies) / noamx_total if noamx_total > 0 else 0

        payload = json.dumps({
            "done": True, "docs": total,
            "amx_avg_ms": amx_avg, "noamx_avg_ms": noamx_avg,
            "amx_dps": amx_dps, "noamx_dps": noamx_dps,
            "amx_total_s": amx_total, "noamx_total_s": noamx_total,
        })
        yield f"data: {payload}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# /rag_stream  — SSE stream for full RAG pipeline
# ---------------------------------------------------------------------------
@app.route("/rag_stream")
def rag_stream():
    question   = request.args.get("question", RAG_QUESTIONS[0])
    top_k      = int(request.args.get("top_k", DEFAULT_TOP_K))
    max_tokens = int(request.args.get("max_tokens", DEFAULT_MAX_TOKENS))
    runs       = int(request.args.get("runs", 1))

    def generate():
        amx_embed_ms_list, noamx_embed_ms_list = [], []
        amx_ttft_list, noamx_ttft_list = [], []
        amx_total_list, noamx_total_list = [], []
        amx_pfill_list, noamx_pfill_list = [], []
        amx_e2e_list, noamx_e2e_list = [], []
        amx_last_text, noamx_last_text = "", ""
        chunks = []
        search_ms = 0.0

        for run_idx in range(1, runs + 1):
            if run_idx > 1:
                time.sleep(DEFAULT_COOLDOWN)

            yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — embedding query (AMX)...'})}\n\n"

            # AMX embed query
            amx_vec, amx_embed_ms = get_embedding(
                VLLM_EMBED_SERVICES["amx"],
                question,
            )
            amx_embed_ms_list.append(amx_embed_ms)

            # Milvus search (once per run, using AMX embedding)
            if run_idx == 1 and amx_vec is not None:
                yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — searching Milvus (top {top_k})...'})}\n\n"
                t_search = time.perf_counter()
                chunks = milvus_search(amx_vec, top_k)
                search_ms = (time.perf_counter() - t_search) * 1000.0
                if chunks:
                    yield f"data: {json.dumps({'chunks': chunks})}\n\n"

            # No-AMX embed query
            yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — embedding query (no-AMX)...'})}\n\n"
            _, noamx_embed_ms = get_embedding(VLLM_EMBED_SERVICES["noamx"], question)
            noamx_embed_ms_list.append(noamx_embed_ms)

            # Build RAG context
            context = "\n\n".join(
                f"[Doc {i+1}: {c['title']}]\n{c['text']}"
                for i, c in enumerate(chunks)
            )
            user_content = (
                f"Context documents:\n\n{context}\n\n"
                f"Question: {question} [run {run_idx}]"
            )

            # AMX LLM generation (streaming)
            yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — AMX LLM generating...'})}\n\n"
            amx_result = _stream_llm(
                VLLM_LLM_SERVICES["amx"],
                LLM_MODEL,
                RAG_SYSTEM_PROMPT,
                user_content,
                max_tokens,
                token_event_key="amx_token",
            )
            # stream tokens
            for token_evt in amx_result["token_events"]:
                yield token_evt
            amx_ttft_list.append(amx_result["ttft_ms"])
            amx_total_list.append(amx_result["total_ms"])
            amx_pfill_list.append(amx_result["prefill_tps"])
            amx_e2e_list.append(amx_embed_ms + search_ms + amx_result["total_ms"])
            amx_last_text = amx_result["text"]

            # No-AMX LLM generation (streaming)
            yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — no-AMX LLM generating...'})}\n\n"
            noamx_result = _stream_llm(
                VLLM_LLM_SERVICES["noamx"],
                LLM_MODEL,
                RAG_SYSTEM_PROMPT,
                user_content,
                max_tokens,
                token_event_key="noamx_token",
            )
            for token_evt in noamx_result["token_events"]:
                yield token_evt
            noamx_ttft_list.append(noamx_result["ttft_ms"])
            noamx_total_list.append(noamx_result["total_ms"])
            noamx_pfill_list.append(noamx_result["prefill_tps"])
            noamx_e2e_list.append(noamx_embed_ms + search_ms + noamx_result["total_ms"])

        def _mean(lst): return statistics.mean(lst) if lst else 0

        payload = json.dumps({
            "done": True,
            "amx_embed_ms":    _mean(amx_embed_ms_list),
            "noamx_embed_ms":  _mean(noamx_embed_ms_list),
            "amx_ttft_ms":     _mean(amx_ttft_list),
            "noamx_ttft_ms":   _mean(noamx_ttft_list),
            "amx_pfill_tps":   _mean(amx_pfill_list),
            "noamx_pfill_tps": _mean(noamx_pfill_list),
            "amx_total_ms":    _mean(amx_total_list),
            "noamx_total_ms":  _mean(noamx_total_list),
            "amx_e2e_ms":      _mean(amx_e2e_list),
            "noamx_e2e_ms":    _mean(noamx_e2e_list),
        })
        yield f"data: {payload}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# /llm_stream  — SSE stream for direct LLM comparison (section 3)
# ---------------------------------------------------------------------------
@app.route("/llm_stream")
def llm_stream():
    question   = request.args.get("question", LLM_QUESTIONS[0])
    runs       = int(request.args.get("runs", DEFAULT_RUNS))
    max_tokens = int(request.args.get("max_tokens", DEFAULT_MAX_TOKENS))

    def generate():
        amx_ttft_list, noamx_ttft_list = [], []
        amx_total_list, noamx_total_list = [], []
        amx_pfill_list, noamx_pfill_list = [], []

        for run_idx in range(1, runs + 1):
            if run_idx > 1:
                time.sleep(DEFAULT_COOLDOWN)

            busted = f"{question} [run {run_idx}]"

            yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — AMX...'})}\n\n"
            amx_r = _stream_llm(
                VLLM_LLM_SERVICES["amx"], LLM_MODEL,
                "You are a helpful AI assistant specializing in computer architecture.",
                busted, max_tokens, "amx_token",
            )
            for evt in amx_r["token_events"]:
                yield evt
            amx_ttft_list.append(amx_r["ttft_ms"])
            amx_total_list.append(amx_r["total_ms"])
            amx_pfill_list.append(amx_r["prefill_tps"])

            yield f"data: {json.dumps({'status': f'Run {run_idx}/{runs} — no-AMX...'})}\n\n"
            noamx_r = _stream_llm(
                VLLM_LLM_SERVICES["noamx"], LLM_MODEL,
                "You are a helpful AI assistant specializing in computer architecture.",
                busted, max_tokens, "noamx_token",
            )
            for evt in noamx_r["token_events"]:
                yield evt
            noamx_ttft_list.append(noamx_r["ttft_ms"])
            noamx_total_list.append(noamx_r["total_ms"])
            noamx_pfill_list.append(noamx_r["prefill_tps"])

        def _mean(lst): return statistics.mean(lst) if lst else 0

        payload = json.dumps({
            "done": True,
            "amx_ttft_ms":     _mean(amx_ttft_list),
            "noamx_ttft_ms":   _mean(noamx_ttft_list),
            "amx_pfill_tps":   _mean(amx_pfill_list),
            "noamx_pfill_tps": _mean(noamx_pfill_list),
            "amx_total_ms":    _mean(amx_total_list),
            "noamx_total_ms":  _mean(noamx_total_list),
        })
        yield f"data: {payload}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# Shared: non-streaming LLM call that collects tokens then returns SSE events
# ---------------------------------------------------------------------------
def _stream_llm(
    api_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    token_event_key: str,
) -> dict:
    """
    Calls vLLM streaming endpoint, returns dict with:
      ttft_ms, total_ms, prefill_tps, text, token_events (list of SSE strings)
    """
    prompt_tokens = len((system_prompt + user_content)) // 4
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = []
    token_events = []

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        with _requests.post(api_url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk_str = line[len("data:"):].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    if chunk.get("usage") and chunk["usage"].get("prompt_tokens"):
                        prompt_tokens = chunk["usage"]["prompt_tokens"]
                    choices = chunk.get("choices", [])
                    if choices:
                        token = choices[0].get("delta", {}).get("content", "")
                        if token:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            full_text.append(token)
                            token_count += 1
                            token_events.append(
                                f"data: {json.dumps({token_event_key: token})}\n\n"
                            )
                except Exception:
                    continue

    except Exception as e:
        return {
            "ttft_ms": 0, "total_ms": 0, "prefill_tps": 0,
            "text": "", "token_events": [],
            "error": str(e),
        }

    t1 = time.perf_counter()
    if first_token_time is None:
        first_token_time = t1
    ttft_ms    = (first_token_time - t0) * 1000.0
    total_ms   = (t1 - t0) * 1000.0
    prefill_tps = prompt_tokens / (ttft_ms / 1000.0) if ttft_ms > 0 else 0

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "prefill_tps": prefill_tps,
        "text": "".join(full_text),
        "token_events": token_events,
    }


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5002)
