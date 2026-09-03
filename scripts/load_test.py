"""Concurrent load test against POST /query (item 5, 2026-09-02 scope revision). A plain concurrent script,
not Locust: the user's own scope note allowed either, and a single-run "record the honest numbers once"
need doesn't warrant Locust's separate runner/web UI. Every number in LOAD_TEST_RESULTS.md comes directly
from this script's own output (CLAUDE.md §5) — never retyped by hand, never tuned to look better.

Requires a running backend (`uvicorn backend.main:app`) and a reachable Redis (both already required to run
the app at all). Uploads one real fixture corpus, then fires N concurrent POST /query requests against it
and reports per-request status/latency plus aggregate stats.

Run: KMP_DUPLICATE_LIB_OK=TRUE python scripts/load_test.py [--base-url http://localhost:8000] [--n 15]
      [--query "..."] [--fixture reviews.csv]

--query targets a single query at every concurrent request instead of round-robining the default 3 —
used to target a specific tool/path deliberately, e.g. "Should I use BERT or DistilBERT?" (routes through
evaluate_candidates with a non-default candidate model, per docs/MODEL_RECOMMENDATION.md — see
LOAD_TEST_RESULTS.md's evaluate_candidates warm-up verification run).
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "csv"

_QUERIES = [
    "Analyze the sentiment",
    "Summarize these documents",
    "Find complaints about delayed delivery",
]


def _upload_session(base_url: str, fixture_filename: str) -> str:
    content = (FIXTURES_DIR / fixture_filename).read_bytes()
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{base_url}/upload", files={"file": (fixture_filename, content, "text/csv")})
        resp.raise_for_status()
        return resp.json()["session_id"]


def _one_query(base_url: str, session_id: str, query: str) -> dict:
    """A 200 status alone isn't "success" here: /query's own contract (backend/schemas.py) returns 200 with
    a non-null body["error"] when the agent degrades gracefully (e.g. an LLM-side rate limit) rather than
    raising an HTTP error — reporting HTTP status alone would silently count those as successes."""
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{base_url}/query", json={"session_id": session_id, "query": query})
        duration_ms = (time.perf_counter() - start) * 1000
        body_error = None
        if resp.status_code == 200:
            body_error = resp.json().get("error")
        return {"status": resp.status_code, "duration_ms": duration_ms, "body_error": body_error}
    except httpx.HTTPError as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        return {"status": None, "duration_ms": duration_ms, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument(
        "--query", default=None, help="fire this single query for every request, instead of round-robin"
    )
    parser.add_argument("--fixture", default="reviews.csv", help="fixture CSV filename under tests/fixtures/csv/")
    args = parser.parse_args()

    session_id = _upload_session(args.base_url, args.fixture)
    print(f"Uploaded fixture corpus ({args.fixture}), session_id={session_id}")

    queries = [args.query] if args.query else _QUERIES
    print(
        f"Firing {args.n} concurrent POST /query requests (query={args.query!r})..."
        if args.query
        else f"Firing {args.n} concurrent POST /query requests..."
    )
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.n) as pool:
        futures = [pool.submit(_one_query, args.base_url, session_id, queries[i % len(queries)]) for i in range(args.n)]
        results = [f.result() for f in as_completed(futures)]
    wall_ms = (time.perf_counter() - start) * 1000

    successes = [r for r in results if r["status"] == 200 and r["body_error"] is None]
    degraded = [r for r in results if r["status"] == 200 and r["body_error"] is not None]
    rate_limited = [r for r in results if r["status"] == 429]
    other_failures = [r for r in results if r["status"] not in (200, 429)]

    print(f"\nWall time for all {args.n} requests: {wall_ms:.0f} ms")
    print(f"Successes (200, body error=null): {len(successes)}")
    print(f"Degraded (200, body error set — agent caught an internal failure): {len(degraded)}")
    for r in degraded:
        print(f"  duration_ms={r['duration_ms']:.0f} error={r['body_error']}")
    print(f"Rate-limited (429): {len(rate_limited)}")
    print(f"Other failures: {len(other_failures)}")
    for r in other_failures:
        print(f"  status={r['status']} duration_ms={r['duration_ms']:.0f} error={r.get('error')}")

    if successes:
        durations = [r["duration_ms"] for r in successes]
        print(
            f"\nSuccessful request latency: median={statistics.median(durations):.0f} ms, "
            f"min={min(durations):.0f} ms, max={max(durations):.0f} ms"
        )

    print("\nDone. Copy the numbers above into LOAD_TEST_RESULTS.md verbatim.")


if __name__ == "__main__":
    main()
