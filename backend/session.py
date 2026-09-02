"""SessionStore — session-scoped state, keyed by session_id, backed by Redis.

CLAUDE.md §3.5 (revised): session state now persists in Redis rather than an in-process dict, for
restart-safety (a server restart no longer drops every active session) and multi-process readiness (a
module-level dict is only visible to the worker process that created it — Redis is not). Per
docs/DATA_FLOW.md §3: each upload creates/replaces a corpus_ref tied to session_id, and profile/chat_history
persist for the life of the session so follow-up turns don't re-profile from scratch. A new upload in the
same session explicitly invalidates the previous corpus's cached profile (not silently — callers are told a
replacement happened).

The public interface (create/get/exists/set_corpus/update_profile/append_turn) is unchanged from the
in-memory version — this is a storage-backend swap behind the existing abstraction, not a rewrite of
callers. See docs/TECH_STACK.md for the rationale behind choosing Redis over SQLite here.

Same encapsulated-class pattern as ingestion/store.py's CorpusStore — not the module-level-`df` anti-pattern
CLAUDE.md §4 warns against: state lives inside Redis, addressed only through this class, never as a
module-level mutable Python object.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field

import redis as redis_lib

from config import settings

_KEY_PREFIX = "textinsight:session:"


@dataclass
class SessionData:
    session_id: str
    corpus_ref: str | None = None
    source_filename: str | None = None
    profile: dict | None = None
    chat_history: list[dict] = field(default_factory=list)


class SessionNotFoundError(KeyError):
    pass


class SessionStore:
    def __init__(self, client: "redis_lib.Redis | None" = None) -> None:
        self._redis = (
            client if client is not None else redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
        )

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    def _save(self, session: SessionData) -> None:
        self._redis.set(self._key(session.session_id), json.dumps(asdict(session)))

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        self._save(SessionData(session_id=session_id))
        return session_id

    def get(self, session_id: str) -> SessionData:
        raw = self._redis.get(self._key(session_id))
        if raw is None:
            raise SessionNotFoundError(f"No session found for session_id={session_id!r}")
        return SessionData(**json.loads(raw))

    def exists(self, session_id: str) -> bool:
        return self._redis.exists(self._key(session_id)) == 1

    def set_corpus(self, session_id: str, corpus_ref: str, source_filename: str, profile: dict) -> None:
        """A new upload replaces the active corpus and its cached profile — explicit, not silent (the
        caller is expected to tell the user this happened, per docs/DATA_FLOW.md §3)."""
        session = self.get(session_id)
        session.corpus_ref = corpus_ref
        session.source_filename = source_filename
        session.profile = profile
        self._save(session)

    def update_profile(self, session_id: str, profile: dict) -> None:
        session = self.get(session_id)
        session.profile = profile
        self._save(session)

    def append_turn(self, session_id: str, user_query: str, final_answer: str | None) -> None:
        session = self.get(session_id)
        session.chat_history.append({"role": "user", "content": user_query})
        session.chat_history.append({"role": "assistant", "content": final_answer})
        self._save(session)


session_store = SessionStore()
