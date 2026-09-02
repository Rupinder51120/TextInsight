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
