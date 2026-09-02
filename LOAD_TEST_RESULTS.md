# Load Test Results

Item 5 of the 2026-09-02 scope revision (CLAUDE.md §3.5 / docs/TECH_STACK.md — see also
`docs/TECH_STACK.md`'s Redis section for the rest of that revision). Every number below is copied verbatim
from one real run of `scripts/load_test.py` against a locally running backend (`uvicorn backend.main:app`,
real Redis, real Groq key) — nothing here is tuned, retried to get a better number, or hand-edited.
Per CLAUDE.md §5, this is a real measurement, not an estimate.

## Run configuration

- Date: 2026-09-02
- Target: `POST /query`, 15 concurrent requests (`python scripts/load_test.py --n 15`), one shared session
  (fixture corpus `tests/fixtures/csv/reviews.csv`), 3 queries cycled round-robin across the 15 requests.
- Backend: single `uvicorn` process, no `--workers`, macOS, CPU-only — the project's documented single-
  instance deployment target (`docs/TECH_STACK.md`), not a multi-worker/production topology.
- Rate limit in effect: 10 requests/minute per client IP (item 4), Groq's own 8,000 TPM per-minute cap
  (`openai/gpt-oss-20b`, on-demand tier) also in effect and independent of this app's limiter.

## Raw output

```
Uploaded fixture corpus, session_id=e7112b53a32d4cc0a8cee899d0c8ea37
Firing 15 concurrent POST /query requests...

Wall time for all 15 requests: 31163 ms
Successes (200, body error=null): 9
Degraded (200, body error set — agent caught an internal failure): 1
  duration_ms=25772 error=I couldn't process that request right now (Groq call failed: Error code: 429 -
  {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization
  `org_01ksw6dk8bfzkt50knsq19dzx2` service tier `on_demand` on tokens per minute (TPM): Limit 8000,
  Used 7297, Requested 974. Please try again in 2.0325s. ...'}}). Please try again.
Rate-limited (429): 5
Other failures: 0

Successful request latency: median=14363 ms, min=5119 ms, max=31154 ms
```

`GET /metrics` immediately after the run (server-side, includes the upload call and this metrics call
itself): `{"request_count": 17, "error_count": 5, "average_latency_ms": 2139.92, "error_rate": 0.2941}`.
The average is much lower than the client-observed successful-request median because it blends in the 5
near-instant 429 rejections (a few ms each) with the slow successful ones.

## What actually happened (honest read, not softened)

**The rate limiter worked as designed.** All 15 requests came from one client (one IP), so item 4's
10/minute cap engaged partway through: 5 of 15 got a `429` — that's the intended behavior of a per-IP
limiter under a single-client burst, not a bug.

**Of the 10 requests that got past the limiter, request-level concurrency was much lower than 10x.**
Server-side per-request durations (from the structured `request` log, not the client's wall-clock number)
for the 10 non-429 responses: 5112, 9107, 2566, 1363, 1318, 1554, 2020, 4522, 3306, 5383 ms — summing to
~36.3s of server-side processing time inside a ~31.2s wall-clock window. That's only possible if the
backend was processing requests close to one-at-a-time, not 10-way in parallel: `backend/main.py`'s
`/query` handler is `async def` but calls the (synchronous) LangGraph agent and Hugging Face pipelines
directly, without offloading to a thread pool — so a CPU-bound tool call or a blocking Groq HTTP call holds
the single event loop for its full duration, and other requests queue behind it rather than running
alongside it. The one "degraded" (LLM-rate-limited) request took 25.8s server-side — consistent with it
sitting queued behind several other requests before it even got to make its own Groq call, at which point
Groq's own TPM cap (shared across all in-flight requests, not per-request) had already been mostly consumed
by the requests ahead of it.

**This is a genuine, currently-unaddressed limitation, not something this load test fixed.** Per the scope
for item 5 ("run it once and record the honest results... don't tune anything to make them look better
first"), no changes were made to the backend to improve concurrency after seeing these numbers. If real
concurrent-user throughput becomes a goal, the two independent bottlenecks visible here — this project's own
in-process request handling (no thread-pool offload for blocking work) and Groq's shared per-minute token
budget — would need to be addressed separately; neither is solved by the rate limiter added in item 4, which
protects against abuse but does not add concurrency.

## Update 2026-09-02: async-blocking fix, re-tested (before/after)

`backend/main.py`'s `/query` handler now offloads `run_agent(...)` to FastAPI's thread pool
(`run_in_threadpool`) instead of calling it inline in the `async def` handler — directly addressing the
bottleneck identified above. Two more runs, same 15-concurrent-request setup, same fixture corpus, same
rate limit config. Numbers below are raw, unedited script output, same as the original run.

**Run 2 (threadpool fix only) — the server crashed partway through:**

```
Wall time for all 15 requests: 2677 ms
Successes (200, body error=null): 0
Degraded (200, body error set — agent caught an internal failure): 2
  duration_ms=1074 error='generate_embeddings' failed: cannot import name 'AutoConfig' from 'transformers'
  duration_ms=1824 error='generate_embeddings' failed: 'module' object is not callable
Rate-limited (429): 5
Other failures: 8
  status=None duration_ms=2672 error=Server disconnected without sending a response.
  (x8, all "Server disconnected without sending a response")
```

The `uvicorn` process was no longer listening on its port after this run — it crashed outright, not just a
request-level error.

**Root cause**: offloading to a thread pool means multiple threads can now call
`models/faiss_index.py`'s `_get_embedding_model()` at the same time on first use. That function had no
lock around its lazy `from sentence_transformers import SentenceTransformer` construction — unlike
`models/registry.py`'s `get_pipeline()`, which already double-checked-locks the equivalent path. Under the
old, effectively-single-threaded request handling this race could never fire (nothing ever ran
concurrently); the threadpool fix is what exposed it. This is a **pre-existing latent bug**, not something
introduced by the fix — the fix just made concurrency real enough to trigger it.

**Applied the same lock pattern already used in `models/registry.py` to `models/faiss_index.py`, then
re-ran the identical test — it crashed again, with the identical error:**

```
Wall time for all 15 requests: 3217 ms
Successes (200, body error=null): 0
Degraded (200, body error set — agent caught an internal failure): 1
  duration_ms=1486 error='generate_embeddings' failed: cannot import name 'AutoConfig' from 'transformers'
Rate-limited (429): 5
Other failures: 9
  status=None duration_ms=3212-3216 error=Server disconnected without sending a response. (x9)
```

Locking `_get_embedding_model()`'s own cache was necessary but not sufficient: `models/registry.py`'s
`get_pipeline()` (used by `sentiment_analysis`/`summarize_text`) and `models/faiss_index.py`'s
`_get_embedding_model()` (used by `generate_embeddings`) guard *different* locks over *different* caches.
Nothing stops both from constructing a model for the first time at the same moment — and the actual failure
point is inside `transformers`' own internal lazy-attribute-loading mechanism (`_LazyModule`), which is
shared, global, mutable process state that neither lock protects. Two per-cache locks aren't enough when the
race is in a piece of shared state neither cache owns.

**What this run confirms nonetheless — the threadpool fix itself works as intended.** The structured
request log for run 3 shows six separate requests' `understand_intent` LLM calls firing within about 4
milliseconds of each other (`09:55:42.040738Z` through `09:55:42.045209Z`) — genuinely concurrent execution
that was structurally impossible before this fix (the original run's server-side durations *summed to more
than the wall-clock window*, proving near-total serialization; see above). The async-blocking problem
named in this task is fixed. It surfaced a second, independent, pre-existing bug in doing so.

**Honest bottom line**: the requested fix (`run_in_threadpool`) is correct and did what it was supposed to
do — concurrent requests now actually run in parallel at the asyncio level. It did not make this load test
pass, because it exposed a real crash-causing thread-safety bug in first-time Hugging Face model loading
that existed in this codebase before today and was simply never reachable until now. A one-line lock fix
(applied above) was not enough to resolve it, because the unsafe shared state lives inside a third-party
library's lazy-import mechanism, not in either of this project's own caches. Fully fixing it needs one of:
(a) a single process-wide lock serializing *every* first-time model construction across both
`models/registry.py` and `models/faiss_index.py` (simple, but forces first-time loads of unrelated tools
to queue behind each other), or (b) eagerly loading all models once at process startup instead of on first
use — which is what `docs/LATENCY_AND_PERFORMANCE.md` §4 already describes as one of the two intended
options ("loaded once at process startup **or** on first use") but only the lazy half was ever built. Not
implemented here — this goes beyond this task's scope of offloading the blocking call, and choosing between
those two options is a real tradeoff worth a decision, not something to pick silently.

## Update 2026-09-02, part 2: startup warm-up fix, re-tested (run 4, the actual before/after)

Implemented option (b) above: `backend/main.py` now loads every default model (sentiment, zero-shot
classification, summarization, embeddings) once via a FastAPI `lifespan` startup hook
(`_warm_up_all_models`), before the server accepts any traffic. `models/registry.py`'s and
`models/faiss_index.py`'s per-cache locks stay in place as defense-in-depth, but they are no longer the
thing doing the real work — by the time a second concurrent request could ever reach a lazy-load path, the
cache is already populated and neither lock is ever contended for a first-time load.

**Startup time — measured, not estimated.** Same machine, same command (`uvicorn backend.main:app`), timed
from process launch to the first successful `GET /metrics` response (i.e. the server is actually ready to
serve, which for the new code requires the full warm-up to finish first):

| | Startup time |
|---|---|
| Before (lazy-on-first-use, no warm-up) | 0.77 s |
| After (eager warm-up at startup) | 7.20 s |
| Increase | **+6.44 s** |

The app's own `model_warmup` startup log line independently reports `duration_ms: 6464.5` for loading all
four models — consistent with the externally-measured +6.44 s increase. This is a real, one-time cost paid
once per process start, traded for eliminating a crash under concurrent load.

**Load test, run 4 (threadpool + startup warm-up) — same 15-concurrent-request setup, unedited output:**

```
Uploaded fixture corpus, session_id=8a390039ee354d97993ab1e9677d8805
Firing 15 concurrent POST /query requests...

Wall time for all 15 requests: 5919 ms
Successes (200, body error=null): 10
Degraded (200, body error set — agent caught an internal failure): 0
Rate-limited (429): 5
Other failures: 0

Successful request latency: median=4986 ms, min=1730 ms, max=5919 ms
```

`GET /metrics` immediately after: `{"request_count": 18, "error_count": 5, "average_latency_ms": 2350.03,
"error_rate": 0.2778}`.

**Is the crash actually gone, not just less frequent?** The server process was still listening and healthy
after this run (verified directly — `lsof` still showed it on its port), unlike both run 2 and run 3, which
both took the whole process down. All 10 non-rate-limited requests completed cleanly (0 degraded, 0 other
failures) — no `AutoConfig` import error, no `'module' object is not callable`, on any of them. That's a
single clean run, not dozens, so "gone" here means "did not reproduce in this run," not "mathematically
impossible" — but it's consistent with the fix actually addressing the mechanism (no two requests can ever
race to load the same model first, because nothing is ever loaded at request time anymore).

**Concurrency, compared directly against the very first (pre-fix) run:**

| | Run 1 (original, no fix) | Run 4 (threadpool + warm-up) |
|---|---|---|
| Wall time, 15 requests | 31,163 ms | 5,919 ms |
| Successes | 9 | 10 |
| Successful latency (median) | 14,363 ms | 4,986 ms |
| Successful latency (min–max) | 5,119–31,154 ms | 1,730–5,919 ms |
| Sum of server-side durations (non-429 requests) | ~36,251 ms | ~42,140 ms |
| Wall-clock window those durations fit into | 31,163 ms | 5,906 ms |
| Effective concurrency (sum ÷ wall time) | **~1.16x** (near-total serialization) | **~7.13x** |

Per-request server-side durations for run 4's 10 successes (from the structured `request` log): 1718,
1816, 1861, 4979, 4974, 4976, 5023, 5367, 5520, 5906 ms — summing to ~42.1s of work completed inside a
5.9s window. That is only possible with real parallel execution, not queued serialization.

**Honest caveats, not glossed over:**
- Still only 10/15 succeeded, not 15/15 — the other 5 hit the per-IP rate limiter (item 4), which is
  working exactly as designed for a single-client burst, not a new problem.
- ~7.1x effective concurrency on 10 concurrent successful requests, not close to a theoretical 10x — some
  of that gap is expected (Python's GIL still serializes pure-Python bytecode — JSON parsing, control flow,
  structlog calls — even across threads; only native-code sections like PyTorch tensor ops and blocking
  HTTP I/O actually overlap), and some of it may just be this being a single run on a shared laptop CPU
  during other background work, not a controlled benchmark environment.
- Session state (`session_store`) is read-modify-written per request (get → mutate → save, not an atomic
  Redis transaction) — with all 15 requests sharing one session in this test, concurrent writes to the same
  session's `chat_history` could in principle lose an update under real concurrency. Not observed as a
  test failure here (nothing asserts on final chat_history ordering/completeness), and not investigated or
  fixed — flagged here because it's the kind of thing today's fix makes newly possible, not because it was
  demonstrated.
- This was one run, not a statistical sample. The crash in runs 2 and 3 was itself reproduced twice in a
  row before the warm-up fix, which is why a single clean run after the fix is reasonably convincing; it
  is not proof the race can never recur under different timing.
