# LATENCY_AND_PERFORMANCE.md — TextInsight

## 1. Latency Budget (targets to validate, not guarantees — see PROJECT_SPEC.md §9)

| Stage | Target |
|---|---|
| Ingestion + profiling | < 1.5s |
| Single NLP tool (sentiment/NER, batch ~50 short texts) | 1–4s |
| Zero-shot classification (batch ~50 short texts) | 2–5s (flagged as slowest default tool) |
| Summarization (single doc) | 2–6s |
| Embedding generation (one-time indexing, ≤5k texts) | a few seconds |
| Semantic search query | < 300ms |
| LLM call (intent/plan/synthesis, each) | roughly 0.3–2s depending on Groq load |
| Full multi-step diagnostic workflow | single-digit seconds total |

All numbers above are **planning targets**; the actual measured numbers (from local runs) go into
`README_PLAN.md`'s benchmarks section once measured — never invented ahead of time.

## 2. What Should Be Measured

- Per-node LangGraph execution time (`understand_intent`, `plan_steps`, each `execute_tool` call,
  `synthesize`).
- Per-tool internal breakdown where relevant (e.g., `generate_embeddings`: model-load-if-cold vs. actual
  embedding vs. FAISS index build).
- LLM call latency, separated from tool latency (so it's clear how much of total time is "thinking" vs.
  "computing").
- Total end-to-end request latency (upload→profile is measured separately from query→response, since upload
  is a one-time cost per corpus).
- Semantic search: indexing latency (one-time) vs. query latency (repeated), reported separately — this
  distinction is the main proof point for the "precomputed embeddings enable low-latency repeated search"
  claim.

## 3. Instrumentation Points

- A small `@timed` decorator / context manager wraps every tool function and every LangGraph node function,
  writing `{name, start, end, duration_ms}` into `AgentState.latency` (a list or dict keyed by call order).
- The FastAPI response includes the full `latency` breakdown as structured JSON, not just a single total —
  the Streamlit UI renders this as a small table/bar chart per request (`UI_SPEC.md`).
- Model **load** time (first use, downloading/initializing weights) is measured and reported separately from
  model **inference** time, since load time is a one-time cost that shouldn't be blamed on "the tool is
  slow" in later requests.

## 4. Caching Strategy

- **Model instances**: loaded once at process startup or on first use, cached in a module-level registry for
  the life of the process (never reloaded per request).
- **Dataset profile**: cached per `corpus_ref`/corpus version in session state; invalidated only when the
  underlying corpus changes.
- **FAISS index**: built once per corpus version, persisted to a per-session directory on disk; reused across
  queries and, if the process restarts mid-session, reloadable from disk rather than rebuilt (nice-to-have,
  not MVP-blocking).
- **Research evidence**: optionally cached per (task_type, candidate_models) key within a session to avoid
  repeated identical search calls if the user asks a similar question twice in one session (nice-to-have).

## 5. Batching

- Sentiment/classification/NER tools accept a list of documents and run the HF `pipeline` in batches
  (`pipeline(..., batch_size=N)`), rather than looping one document at a time — this is the single biggest
  lever for these tools' latency on CPU.
- Summarization batch mode chunks the corpus into digestible groups rather than summarizing per-document
  when the document count is large (see `summarize_text`'s `batch_digest` mode).

## 6. Avoiding Unnecessary LLM Calls

- `understand_intent` and `plan_steps` are combined into a single LLM call wherever the plan is simple
  (single-tool intents), only using a second call for genuinely ambiguous/multi-step planning — reduces
  LLM round trips for the common case.
- Cached profile/tool_results are reused across a session instead of being reconstructed or re-explained by
  the LLM.
- `filter_documents` and other purely mechanical steps never invoke the LLM (see `TOOLS_AND_MODELS.md`).
- `synthesize` is skipped (or reduced to a thin pass-through) when a tool's own structured output is already
  sufficiently self-explanatory (e.g., `model_recommendation`'s output), avoiding a redundant LLM call.

## 7. Performance Testing Plan

- A small fixed set of benchmark inputs (e.g., a 200-row review CSV, a 5-page PDF, a 50-row support-ticket
  CSV) run through each workflow at least 3 times locally; median and range reported, not a single
  cherry-picked run.
- Cold-start (first model load) measured separately from warm-run numbers, and both are reported — hiding
  cold-start entirely would misrepresent real first-query latency.
- CPU-only run is the baseline reported number, since that's the project's stated deployability target; a
  GPU comparison is optional/nice-to-have if available.
- Zero-shot classification and multi-step diagnostic workflows are the two flows most likely to miss target
  latency — flagged for explicit measurement and, if needed, documented as a known limitation rather than
  silently left unmeasured.
