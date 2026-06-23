"""
Long-term memory store — pgvector-backed cross-session fact retrieval.

Each entry represents a fact extracted from a past conversation: a module
the user was debugging, a preference they stated, an architectural decision
they described.  At query time the store returns the top-k facts most
similar to the current query so the agent has relevant context without the
user having to repeat themselves across sessions.

Hard security rule: every SELECT is scoped to a single user_id.  There is
no code path that returns rows from a different user.  This is enforced at
the SQL level (WHERE user_id = $1) and verified in the startup test suite.

Schema lives in the ltm_migration script.  The VECTOR(768) dimension matches
all-mpnet-base-v2, which is also used for retrieval embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── SQL constant (tested by C-010) ───────────────────────────────────────────
_RETRIEVE_QUERY: str = (
    "SELECT content FROM agent_long_term_memory "
    "WHERE user_id = $1 "
    "ORDER BY embedding <=> $2::vector "
    "LIMIT $3"
)

_INSERT_QUERY: str = """
INSERT INTO agent_long_term_memory
    (user_id, org_id, content, entity_type, embedding, source_session)
VALUES ($1, $2, $3, $4, $5::vector, $6)
ON CONFLICT DO NOTHING
"""

_UPDATE_ACCESS_QUERY: str = """
UPDATE agent_long_term_memory
   SET last_accessed = NOW(), access_count = access_count + 1
 WHERE user_id = $1 AND content = $2
"""


# ─────────────────────────────────────────────────────────────────────────────
# MemoryEntry dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single long-term memory record."""
    user_id: str
    content: str
    entity_type: str = "user_fact"
    org_id: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    source_session: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 1
    relevance_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LongTermStore
# ─────────────────────────────────────────────────────────────────────────────

class LongTermStore:
    """pgvector-backed long-term memory store.

    All operations are user-scoped — there is no way to query across users.

    Usage:
        store = LongTermStore()
        facts = await store.retrieve(user_id="alice", query="auth system", top_k=5)
        await store.store(entry)
    """

    def __init__(self) -> None:
        self._embedder = None  # lazy-loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_embedder(self):
        """Lazy-load the shared embedder singleton."""
        if self._embedder is None:
            try:
                from app.core.database import get_embedder
                self._embedder = get_embedder()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[LTM] embedder unavailable: %s", exc)
        return self._embedder

    def _get_pool(self):
        """Lazy-load the shared pg_pool singleton."""
        try:
            from app.core.database import get_pg_pool
            return get_pg_pool()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LTM] pg_pool unavailable: %s", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        org_id: Optional[str] = None,
    ) -> List[str]:
        """Return the top-k most relevant memory facts for *user_id*.

        Performs a pgvector cosine similarity search scoped to user_id.
        Returns an empty list (never raises) when DB or embedder unavailable.

        Args:
            user_id: The user whose memories to retrieve. REQUIRED — never omit.
            query:   Natural-language query to embed for similarity search.
            top_k:   Number of results (default 5).
            org_id:  Optional org scope (future use — not yet applied as filter).

        Returns:
            List of content strings, most-relevant first.
        """
        pool = self._get_pool()
        embedder = self._get_embedder()

        if pool is None or embedder is None:
            logger.debug("[LTM] retrieve skipped — pool=%s, embedder=%s", pool, embedder)
            return []

        try:
            # Embed the query
            embedding: List[float] = embedder.embed_query(query)
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

            # Execute user-scoped cosine search
            with pool.connection() as conn:
                try:
                    from pgvector.psycopg import register_vector  # type: ignore
                    register_vector(conn)
                except Exception:  # noqa: BLE001
                    pass
                rows = conn.execute(
                    _RETRIEVE_QUERY, (user_id, embedding_str, top_k)
                ).fetchall()

            facts = [row[0] for row in rows]
            logger.info("[LTM] retrieved %d facts for user=%s", len(facts), user_id)
            return facts

        except Exception as exc:  # noqa: BLE001
            logger.warning("[LTM] retrieve failed for user=%s: %s", user_id, exc)
            return []

    async def store(self, entry: MemoryEntry) -> bool:
        """Persist a MemoryEntry to the long-term memory table.

        Returns True on success, False on failure (never raises).
        """
        pool = self._get_pool()
        embedder = self._get_embedder()

        if pool is None or embedder is None:
            logger.debug("[LTM] store skipped — pool or embedder unavailable")
            return False

        try:
            if not entry.embedding:
                entry.embedding = embedder.embed_query(entry.content)

            embedding_str = "[" + ",".join(str(v) for v in entry.embedding) + "]"

            with pool.connection() as conn:
                try:
                    from pgvector.psycopg import register_vector  # type: ignore
                    register_vector(conn)
                except Exception:  # noqa: BLE001
                    pass
                conn.execute(
                    _INSERT_QUERY,
                    (
                        entry.user_id,
                        entry.org_id,
                        entry.content,
                        entry.entity_type,
                        embedding_str,
                        entry.source_session,
                    ),
                )
                conn.commit()

            logger.info("[LTM] stored fact for user=%s: %s…", entry.user_id, entry.content[:60])
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning("[LTM] store failed for user=%s: %s", entry.user_id, exc)
            return False

    async def store_batch(self, entries: List[MemoryEntry]) -> int:
        """Store multiple entries, returning the count of successful writes."""
        count = 0
        for entry in entries:
            if await self.store(entry):
                count += 1
        return count


# ── Module-level singleton ────────────────────────────────────────────────────
_ltm_store: Optional[LongTermStore] = None


def get_ltm_store() -> LongTermStore:
    """Return the process-wide LongTermStore singleton."""
    global _ltm_store
    if _ltm_store is None:
        _ltm_store = LongTermStore()
    return _ltm_store
