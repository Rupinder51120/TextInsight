# SECURITY_AND_RELIABILITY.md — TextInsight

## 1. File Validation

- File type restricted to `.csv`, `.txt`, `.pdf` by content sniffing (not just extension trust).
- Size limit enforced (`MAX_UPLOAD_MB` config) before any parsing begins.
- CSV parsing uses safe Pandas defaults (no arbitrary code execution paths like pickled objects); PDF
  extraction via PyMuPDF does not execute embedded scripts/JavaScript in the PDF.
- Empty or entirely non-text content (e.g., scanned-image PDF with no extractable text) is rejected with a
  clear error rather than silently producing an empty corpus.

## 2. Prompt Injection Considerations

- **Uploaded document content is treated strictly as data, never as instructions.** Tool functions pass
  document text into model pipelines (HF pipelines) as plain input; the only place document text reaches the
  LLM is inside clearly delimited context blocks in `synthesize`/`model_recommendation` prompts, with system
  instructions that explicitly tell the model to treat that content as data to analyze, not as commands to
  follow.
- **Research/search results are treated the same way** — snippets returned by the search API are external,
  untrusted content; they are quoted narrowly and attributed, never interpreted as instructions to the agent
  (matches the general instruction-source-boundary principle: only the user's actual chat input is a valid
  instruction source).
- Tested explicitly in `TESTING_STRATEGY.md` §10 with adversarial document content containing
  instruction-like text.

## 3. Malicious Documents

- No execution of any embedded content from uploaded files (macros, scripts, embedded objects) — extraction
  libraries used (Pandas CSV reader, PyMuPDF text extraction) do not execute document-embedded code.
- Extremely large or pathological files (e.g., zip-bomb-style CSVs, PDFs with excessive page counts) are
  bounded by the size limit and a page/row count cap, with a clear rejection message.

## 4. API Key Handling

- All API keys (`GROQ_API_KEY`, `TAVILY_API_KEY`, `LANGCHAIN_API_KEY`) loaded from environment variables /
  `.env`, never hardcoded, never logged, never included in any API response body or error message returned
  to the frontend.
- `.env` excluded via `.gitignore`; `.env.example` ships with placeholder values only.

## 5. Size Limits

- Upload size (`MAX_UPLOAD_MB`), row/document count cap for very large CSVs (with a clear "truncated to N
  rows for analysis" notice rather than silent truncation), and PDF page count cap.
- Batch sizes for NLP tools bounded to keep single-request latency and memory bounded (see
  `LATENCY_AND_PERFORMANCE.md` §5).

## 6. Timeouts

- `LLM_TIMEOUT_SECONDS` bounds every Groq call.
- Search API calls bounded by a short timeout (a few seconds) since `research_models` must degrade
  gracefully rather than stall a whole turn.
- Model inference calls, being local and CPU-bound, are bounded implicitly by input size caps rather than a
  wall-clock timeout (a genuinely hung local inference call would indicate a bug, not a transient condition
  worth retrying).

## 7. Retries

- LLM and search API calls use bounded retry (e.g., up to 2–3 attempts) with exponential backoff, only on
  retryable errors (timeouts, 429, 5xx) — never retried indefinitely, and never retried on 4xx
  client-error-type failures (e.g., invalid API key), which surface immediately instead.

## 8. Rate Limits

- Provider-side rate limits (Groq, search API) are respected via the backoff behavior in §7 — this protects
  against the *providers'* limits, not this API's own.
- This API's own rate limiting (added 2026-09-02, via `slowapi`): `POST /query` is capped at 10
  requests/minute per client IP (`backend/main.py`), returning `429` over the limit. Not covered:
  - **`/upload`, `/session/{id}/history`, and `/metrics` are unlimited** — only `/query` (the endpoint that
    triggers the LLM/model pipeline) is rate-limited.
  - **No per-user quotas** — there is no user/auth concept at all (single-user/demo scope, `PROJECT_SPEC.md`
    §10); the limit is purely per-client-IP.
  - **No distributed rate limiting.** The limiter is in-memory and per-process — accurate for this
    project's single-instance deployment target, but it would not share state across multiple backend
    replicas/workers if one were ever run (each would enforce its own independent 10/minute).

## 9. Unsafe Tool Execution

- Every tool has a fixed, typed (Pydantic) input schema — the agent cannot invoke a tool with arbitrary
  free-form arguments; malformed tool-call arguments from the LLM fail validation and route to
  `handle_error` rather than executing with bad inputs.
- No tool performs file-system writes outside its designated session directory (e.g., FAISS index files),
  and no tool shells out to arbitrary system commands.
- The system exposes no tool capable of training/fine-tuning, regardless of how a user phrases a request —
  this is enforced by the tool catalog itself (no such tool exists to call), not just by prompting.

## 10. External Research Reliability

- Evidence is source-attributed by construction (`research_models` output schema requires `source_url`
  alongside every claim) — there is no code path that produces an unattributed claim from research.
- If the search API is unavailable, misconfigured, or returns nothing usable, the system explicitly reports
  "no external evidence found/available" rather than falling back to an unattributed LLM guess dressed up as
  research.

## 11. Hallucination Mitigation

- Deterministic NLP outputs (sentiment/classification/NER/summarization/embeddings) come from actual model
  inference, not LLM generation — the LLM only narrates/reasons over these structured results.
- `synthesize`/`explain_results` prompts are constructed to only reference facts present in `tool_results`;
  numeric claims (counts, percentages, scores) are computed in Python and inserted into the prompt/response
  as pre-computed values, not left for the LLM to compute or recall.
- `model_recommendation`'s A/B/C structure (per `MODEL_RECOMMENDATION.md`) is the primary structural defense
  against the specific hallucination risk called out in the project brief (presenting external benchmarks as
  if measured on the user's data).

## 12. Source Attribution

- Every external claim surfaced anywhere in the UI carries a visible source (title + link), consistent
  end-to-end from `research_models`'s output schema through to the Streamlit evidence cards
  (`UI_SPEC.md` §6).

## 13. Known Limitations & Fixes Under Concurrent Requests

`backend/main.py`'s `/query` handler offloads agent execution to a thread pool
(`fastapi.concurrency.run_in_threadpool`, added 2026-09-02 — see `LOAD_TEST_RESULTS.md`) so that concurrent
requests genuinely run in parallel rather than queuing behind one another. That surfaced two independent,
pre-existing risks that real concurrency was never reachable enough to trigger before — one now fixed with
verified evidence, one identified and left open.

### 13.1 Session State Race — identified, not fixed

- `SessionStore.update_profile` and `SessionStore.append_turn` are a plain read-modify-write against Redis
  (`get` → mutate the in-memory `SessionData` → `set`), not an atomic transaction (no `WATCH`/`MULTI`, no
  optimistic locking). If two requests for the *same* `session_id` are in flight at the same time — e.g.
  two browser tabs on the same session, or a retried request racing the original — the second `set` can
  overwrite the first, silently dropping a `chat_history` entry or a profile update rather than merging
  both.
- Documented here as an honestly-acknowledged gap, per the same standard used throughout this project (see
  `LOAD_TEST_RESULTS.md`'s own caveats section, where this was first noted) — not something a workaround
  was attempted for. A real fix would need either a Redis transaction (`WATCH`/`MULTI`/`EXEC`) around each
  read-modify-write, or moving `chat_history` to a Redis list (`RPUSH`, naturally append-only and
  race-free) instead of a field inside one JSON blob.

### 13.2 Concurrent First-Time Model Loading — fixed for the tested path, not for every path

- `models/registry.py`'s and `models/faiss_index.py`'s lazy model caches were not safe against concurrent
  first-time construction: `transformers`' own internal lazy-import state can be corrupted when two threads
  both try to load a model for the first time at once, which crashed the process outright under load
  (`LOAD_TEST_RESULTS.md`, runs 2–3). Fixed by eagerly loading every default model, plus every model named
  anywhere in `model_recommendation.py`'s candidate shortlists, once at process startup
  (`backend/main.py`'s `_warm_up_all_models`) — see `LOAD_TEST_RESULTS.md`'s 2026-09-03 update for the
  full startup-cost numbers and verification detail.
- **What's verified with a passing concurrent load test**: the 4 default models, and
  `cardiffnlp/twitter-roberta-base-sentiment-latest` (the one non-default candidate model reachable today
  through `evaluate_candidates`, since `agent/nodes.py` currently hardcodes the sentiment task type).
- **What's warmed but not exercised by any test**: the other 8 model names in `model_recommendation.py`'s
  classification/NER/summarization/embeddings shortlists. No code path reaches them through
  `evaluate_candidates` today (the hardcoded task type), so there is nothing to concurrency-test yet;
  warming them is defensive, in case that hardcoding changes later. Several of them, notably
  `facebook/bart-large-cnn` and both `sentence-transformers/*` embedding models, load with a **randomly-
  initialized classification head** when forced through the `sentiment-analysis` pipeline task
  `evaluate_candidates` always uses — a separate, pre-existing limitation (README's "only meaningful for
  sentiment-style labeled datasets today" note on `evaluate_candidates`), unrelated to and not fixed by
  this warm-up.
