"""
Migration: add agent_long_term_memory table
============================================
Creates the pgvector-backed long-term memory table used by LongTermStore.

Run standalone:
    python -m app.db.migrations.ltm_migration

Or import and call:
    from app.db.migrations.ltm_migration import run_migration
    await run_migration()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── DDL statements ────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_long_term_memory (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    org_id          TEXT,
    content         TEXT NOT NULL,
    entity_type     TEXT NOT NULL DEFAULT 'user_fact',
    embedding       VECTOR(768) NOT NULL,
    source_session  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_accessed   TIMESTAMPTZ DEFAULT NOW(),
    access_count    INTEGER DEFAULT 1
);
"""

CREATE_USER_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ltm_user_idx
    ON agent_long_term_memory (user_id);
"""

CREATE_EMBEDDING_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ltm_embedding_idx
    ON agent_long_term_memory
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

# ── Migration runner ──────────────────────────────────────────────────────────


def run_migration_sync() -> bool:
    """Run the LTM migration synchronously using psycopg ConnectionPool.

    Returns True on success, False on failure.
    Idempotent — safe to run multiple times (all DDL uses IF NOT EXISTS).
    """
    try:
        from app.core.database import get_pg_pool
        pool = get_pg_pool()

        with pool.connection() as conn:
            # Ensure pgvector extension is installed
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logger.info("[LTM migration] pgvector extension ensured")

            # Create table
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()
            logger.info("[LTM migration] agent_long_term_memory table created/verified")

            # Create indexes
            conn.execute(CREATE_USER_INDEX_SQL)
            conn.commit()
            logger.info("[LTM migration] user index created/verified")

            try:
                conn.execute(CREATE_EMBEDDING_INDEX_SQL)
                conn.commit()
                logger.info("[LTM migration] ivfflat embedding index created/verified")
            except Exception as idx_exc:  # noqa: BLE001
                # ivfflat requires data rows to build — non-fatal on empty table
                logger.warning("[LTM migration] ivfflat index skipped (needs rows): %s", idx_exc)

        logger.info("[LTM migration] ✅ agent_long_term_memory ready")
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("[LTM migration] ❌ Failed: %s", exc)
        return False


async def run_migration() -> bool:
    """Async-compatible wrapper — runs migration in a threadpool executor."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_migration_sync)


# ── Standalone entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(run_migration())
    sys.exit(0 if success else 1)
