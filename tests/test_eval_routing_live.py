"""Live counterpart to tests/eval_routing.py — actually exercises Groq's routing judgement (not mocked).
Per docs/TESTING_STRATEGY.md §3: "a manual/optional run against the live Groq model to catch prompt drift."
Skipped automatically when no real GROQ_API_KEY is configured, so it never breaks CI in an environment
without one.

Floor: docs/FIVE_DAY_BUILD_PLAN.md Day 3 asks for "the routing eval set passing at an acceptable rate"
without pinning a number. 60% is chosen here as a floor that would still catch a real routing regression
(e.g. the agent picking wrong tools entirely) while tolerating the kind of over-elaborate-but-not-wrong
plans real LLM output produces (measured at 80% on 2026-09-01 against openai/gpt-oss-20b — see this
session's actual run for the full per-case report).
"""

import pytest

from config import settings
from tests.eval_routing import format_report, run_eval

_ACCEPTABLE_PASS_RATE = 0.6

pytestmark = pytest.mark.skipif(
    not settings.groq_api_key or settings.groq_api_key == "your-groq-api-key-here",
    reason="no real GROQ_API_KEY configured",
)


def test_live_routing_pass_rate_meets_floor():
    results = run_eval()
    pass_rate = sum(r.passed for r in results) / len(results)

    print("\n" + format_report(results))

    assert pass_rate >= _ACCEPTABLE_PASS_RATE, (
        f"live routing pass rate {pass_rate:.0%} fell below the {_ACCEPTABLE_PASS_RATE:.0%} floor"
    )
