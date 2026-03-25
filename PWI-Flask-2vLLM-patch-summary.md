# PWI-Flask-2vLLM.py — Patch Summary

Patch file: `/tmp/PWI-Flask-2vLLM.patch`

To apply: `patch PWI-Flask-2vLLM.py < /tmp/PWI-Flask-2vLLM.patch`

## Changes

| Area | Change |
|---|---|
| **Questions** | Replace 5 short generic questions with a ~350-token `CONTEXT_DOC` prefix + focused 2-3 sentence questions — forces real prefill work |
| **Service dropdown** | Removed — both containers are always queried, no need to choose |
| **Side-by-side layout** | Two columns (✅ AMX / ❌ No-AMX) so the audience sees both responses and metrics simultaneously |
| **Metrics display** | Added `Prefill tok/s` = `prompt_tokens / TTFT`; renamed "Tokens/sec" → "Decode tok/s" |
| **Cache busting** | `bust=Date.now()` appended to each request URL, prepended to the question server-side |
| **`max_tokens`** | `300` → `50` (RAG sweet spot — prefill is ~46% of total time, delivers ~3x total speedup) |
| **`stream_options`** | Added `include_usage: True` to capture actual prompt token count for prefill metric |
| **Timeout** | `60s` → `120s` (long prompts need more headroom) |
| **`prompt_tokens`** | Parsed from usage chunk and returned in metrics payload to frontend |
