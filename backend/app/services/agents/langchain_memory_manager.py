"""LangChain-backed Chat Memory Manager.

Replaces the custom `ChatMemoryManager` with the idiomatic
`langchain_postgres.PostgresChatMessageHistory` backend. Preserves the
async API consumed by `AgentBrain` (`add_message`, `get_history`) so the
swap is transparent to upstream code.

Design (Phase 2.1 — unified pool):
- All Postgres I/O is checked out from the SHARED `psycopg_pool.ConnectionPool`
  defined in `app.core.database` (no per-call `psycopg.connect`, no private
  long-lived connection).
- A `PostgresChatMessageHistory` is constructed *per operation* with a
  freshly-borrowed connection; the connection is automatically returned to
  the pool when the `with pool.connection():` block exits.
- Sync calls are wrapped in `asyncio.to_thread` to keep the API non-blocking
  for FastAPI handlers.
- Table is auto-created once on first use via `create_tables`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg  # type: ignore  # noqa: F401  (still needed for type hints downstream)
    from langchain_postgres import PostgresChatMessageHistory  # type: ignore
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # type: ignore
    HAS_LANGCHAIN_POSTGRES = True
except ImportError as _e:  # pragma: no cover
    PostgresChatMessageHistory = None  # type: ignore
    HumanMessage = AIMessage = SystemMessage = None  # type: ignore
    psycopg = None  # type: ignore
    HAS_LANGCHAIN_POSTGRES = False
    logger.warning(f"langchain_postgres / psycopg unavailable: {_e}")


_DEFAULT_TABLE = "chat_message_history"


class ChatMemoryManager:
    """LangChain-backed chat memory manager — pooled Postgres edition.

    Public API (kept compatible with agent_brain.py):
        - await add_message(session_id, user_id, role, content, metadata=None)
        - await get_history(session_id, max_tokens=2000) -> str | None
        - await clear(session_id)
    """

    def __init__(
        self,
        table_name: str = _DEFAULT_TABLE,
    ) -> None:
        if not HAS_LANGCHAIN_POSTGRES:
            raise RuntimeError(
                "langchain_postgres is required. Install with `pip install "
                "langchain-postgres psycopg[binary] psycopg-pool`."
            )

        self.table_name = table_name
        self._tables_ready = False
        self._init_table()
        logger.info(
            f"✅ ChatMemoryManager (LangChain Postgres + shared pool) ready "
            f"— table={self.table_name}"
        )

    # ------------------------------------------------------------------ #
    # Connection helpers (delegated to app.core.database singleton pool)  #
    # ------------------------------------------------------------------ #
    @contextmanager
    def _checkout(self) -> Iterator[Any]:
        """Borrow a connection from the shared pool for the lifetime of the
        `with` block. The connection is returned automatically on exit.
        """
        from app.core.database import pg_connection  # local import: avoids
        # importing `psycopg_pool` at module-load when running tests w/o the
        # extension installed.
        with pg_connection() as conn:
            yield conn

    def _init_table(self) -> None:
        """Create the chat-history table once (idempotent)."""
        try:
            with self._checkout() as conn:
                PostgresChatMessageHistory.create_tables(conn, self.table_name)
            self._tables_ready = True
        except Exception as e:
            # `create_tables` is idempotent in newer langchain-postgres; old
            # versions raise if the table already exists. Either way, don't
            # block startup.
            logger.warning(f"create_tables skipped/failed (likely exists): {e}")
            self._tables_ready = True  # assume table exists

    def _history_with(
        self, conn: Any, session_id: str
    ) -> Any:
        """Build a short-lived `PostgresChatMessageHistory` bound to a
        pool-checked-out connection. NOT cached: the connection lifetime is
        tied to the surrounding `with self._checkout()` block.
        """
        return PostgresChatMessageHistory(
            self.table_name,
            session_id,
            sync_connection=conn,
        )

    # ------------------------------------------------------------------ #
    # Public async API                                                    #
    # ------------------------------------------------------------------ #
    async def add_message(
        self,
        session_id: str,
        user_id: str,  # accepted for API compatibility; LangChain history is keyed by session_id
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Append a single message to the session's history."""
        role = (role or "user").lower()
        if role == "user":
            msg = HumanMessage(content=content)
        elif role in ("assistant", "ai"):
            msg = AIMessage(content=content)
        elif role == "system":
            msg = SystemMessage(content=content)
        else:
            msg = HumanMessage(content=content)

        def _do_add() -> None:
            with self._checkout() as conn:
                history = self._history_with(conn, session_id)
                history.add_messages([msg])

        await asyncio.to_thread(_do_add)

    async def get_history(
        self,
        session_id: str,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """Return a flattened text history for prompt injection.

        We approximate token budgeting via 4 chars/token and trim from the
        oldest end to keep the most recent context.
        """
        def _load_messages() -> list:
            with self._checkout() as conn:
                history = self._history_with(conn, session_id)
                return list(history.messages)

        try:
            messages = await asyncio.to_thread(_load_messages)
        except Exception as e:
            logger.error(f"Failed to load history for {session_id}: {e}")
            return None

        if not messages:
            return None

        char_budget = max_tokens * 4
        rendered: list[str] = []
        total = 0
        # Walk newest -> oldest, prepend, stop on budget
        for m in reversed(messages):
            role_label = "User" if isinstance(m, HumanMessage) else (
                "Assistant" if isinstance(m, AIMessage) else "System"
            )
            line = f"{role_label}: {m.content}"
            total += len(line)
            if total > char_budget and rendered:
                break
            rendered.append(line)
        rendered.reverse()
        return "\n".join(rendered) if rendered else None

    async def clear(self, session_id: str) -> None:
        """Clear all messages for a session."""
        def _do_clear() -> None:
            with self._checkout() as conn:
                history = self._history_with(conn, session_id)
                history.clear()

        await asyncio.to_thread(_do_clear)
