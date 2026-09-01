"""SessionStore — in-process, session-scoped state, keyed by session_id.

Per CLAUDE.md §3.5: no database, no external cache — state lives in-process for the life of the server
process; this is a documented, accepted limitation, not a gap. Per docs/DATA_FLOW.md §3: each upload
creates/replaces a corpus_ref tied to session_id, and profile/chat_history persist for the life of the
session so follow-up turns don't re-profile from scratch. A new upload in the same session explicitly
invalidates the previous corpus's cached profile (not silently — callers are told a replacement happened).

Same encapsulated-class pattern as ingestion/store.py's CorpusStore — not the module-level-`df` anti-pattern
CLAUDE.md §4 warns against.
"""

import uuid
from dataclasses import dataclass, field


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
    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = SessionData(session_id=session_id)
        return session_id

    def get(self, session_id: str) -> SessionData:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"No session found for session_id={session_id!r}") from None

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def set_corpus(self, session_id: str, corpus_ref: str, source_filename: str, profile: dict) -> None:
        """A new upload replaces the active corpus and its cached profile — explicit, not silent (the
        caller is expected to tell the user this happened, per docs/DATA_FLOW.md §3)."""
        session = self.get(session_id)
        session.corpus_ref = corpus_ref
        session.source_filename = source_filename
        session.profile = profile

    def update_profile(self, session_id: str, profile: dict) -> None:
        self.get(session_id).profile = profile

    def append_turn(self, session_id: str, user_query: str, final_answer: str | None) -> None:
        session = self.get(session_id)
        session.chat_history.append({"role": "user", "content": user_query})
        session.chat_history.append({"role": "assistant", "content": final_answer})


session_store = SessionStore()
