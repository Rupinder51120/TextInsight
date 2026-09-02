"""SessionStore unit tests — docs/TESTING_STRATEGY.md §1/§2 pattern applied to the Redis-backed store
(CLAUDE.md §3.5 revised: session state persists in Redis, not an in-process dict).

Uses a real local Redis (REDIS_URL, default redis://localhost:6379/0) on a dedicated logical DB so it never
collides with dev data in DB 0, and flushes that DB before/after each test for isolation. Skipped when no
Redis is reachable, mirroring the skip pattern used for the live Groq/Tavily tests.
"""

import pytest
import redis as redis_lib

from backend.session import SessionData, SessionNotFoundError, SessionStore
from config import settings

_TEST_REDIS_URL = settings.redis_url.rsplit("/", 1)[0] + "/15"


def _redis_available() -> bool:
    try:
        redis_lib.Redis.from_url(_TEST_REDIS_URL, socket_connect_timeout=1).ping()
        return True
    except redis_lib.exceptions.RedisError:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="no Redis reachable at REDIS_URL")


@pytest.fixture
def store():
    client = redis_lib.Redis.from_url(_TEST_REDIS_URL, decode_responses=True)
    client.flushdb()
    yield SessionStore(client=client)
    client.flushdb()


def test_create_returns_a_retrievable_empty_session(store):
    session_id = store.create()

    session = store.get(session_id)
    assert session == SessionData(session_id=session_id)


def test_get_unknown_session_raises(store):
    with pytest.raises(SessionNotFoundError):
        store.get("does-not-exist")


def test_exists_reflects_creation(store):
    assert store.exists("does-not-exist") is False

    session_id = store.create()
    assert store.exists(session_id) is True


def test_set_corpus_persists_across_a_fresh_store_instance(store):
    """The point of the Redis swap: state must survive the SessionStore object being torn down and
    recreated (simulating a process restart), not just multiple calls on the same instance."""
    session_id = store.create()
    store.set_corpus(session_id, corpus_ref="corpus-1", source_filename="reviews.csv", profile={"rows": 5})

    reloaded = SessionStore(client=redis_lib.Redis.from_url(_TEST_REDIS_URL, decode_responses=True))
    session = reloaded.get(session_id)

    assert session.corpus_ref == "corpus-1"
    assert session.source_filename == "reviews.csv"
    assert session.profile == {"rows": 5}


def test_update_profile_overwrites_only_the_profile_field(store):
    session_id = store.create()
    store.set_corpus(session_id, corpus_ref="corpus-1", source_filename="reviews.csv", profile={"rows": 5})

    store.update_profile(session_id, {"rows": 6})

    session = store.get(session_id)
    assert session.profile == {"rows": 6}
    assert session.corpus_ref == "corpus-1"  # untouched


def test_append_turn_accumulates_chat_history_in_order(store):
    session_id = store.create()

    store.append_turn(session_id, "first question", "first answer")
    store.append_turn(session_id, "second question", "second answer")

    session = store.get(session_id)
    assert session.chat_history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
