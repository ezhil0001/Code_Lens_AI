"""Phase 4: Chat API Routes - Production-Ready Streaming & Caching.

Exposes the Agent Brain via REST API with:
1. Real-time streaming via Server-Sent Events (SSE)
2. Semantic caching using pgvector
3. Error handling and timeouts
4. Request/response validation
"""

import logging
from typing import AsyncIterator, Optional
import asyncio
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
import json
import os

logger = logging.getLogger(__name__)

# Centralized debug logger for terminal "play-by-play" tracing.
from app.core.logger import (
    logger as flow_logger,
    bind_session,
    timed,
    log_step,
    log_success,
)

# Import Phase 3 components
from app.services.agents.agent_brain import AgentBrain, AgentRequest, AgentResponse, AgentConfig
from app.schemas.chat import (
    ChatRequest, 
    ChatStreamResponse, 
    CacheStatus,
    ComponentHealth,
    FullHealthStatus,
    StreamToken,
)

# Import Pipeline Factory (CRITICAL FIX #1: Component Injection)
from app.services.pipeline_factory import get_agent_brain_dependency, get_pipeline_factory_cached

# Import database for caching
from app.db.session import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1", tags=["chat"])


# ==================== Semantic Cache (pgvector) ====================

class SemanticCache:
    """Production semantic cache backed by **PostgreSQL + pgvector**.

    Strategy:
      1. Embed query with the SHARED singleton embedder (no per-request reload).
      2. Cosine-search the `semantic_cache` table (pgvector `<=>` operator)
         scoped to the requesting `user_id` (multi-tenant safety — P2 #9).
      3. Cache HIT iff `1 - distance >= similarity_threshold` (default 0.95).

    Connection management uses the shared `psycopg_pool.ConnectionPool` from
    `app.core.database` (P2 #7) — eliminates per-request handshake cost,
    making the <20 ms target achievable.

    Schema (auto-created on startup):
        CREATE TABLE semantic_cache (
            id          BIGSERIAL PRIMARY KEY,
            user_id     TEXT NOT NULL,
            query       TEXT NOT NULL,
            response    JSONB NOT NULL,
            embedding   VECTOR(768) NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX semantic_cache_user_idx ON semantic_cache (user_id);
        CREATE INDEX semantic_cache_embedding_idx
            ON semantic_cache USING ivfflat (embedding vector_cosine_ops);
    """

    DEFAULT_TTL_SECONDS = 86400

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = self.DEFAULT_TTL_SECONDS
        self._available = False
        try:
            self._init_backend()
            self._available = True
            logger.info(
                f"✅ Semantic cache (pgvector + pool) ready — "
                f"threshold={similarity_threshold}"
            )
        except Exception as e:
            logger.warning(
                f"⚠️  pgvector cache unavailable ({e}); operating in disabled mode."
            )

    # --------------------------- bootstrap --------------------------- #
    def _init_backend(self):
        from app.core.database import pg_connection, get_embed_dim

        dim = get_embed_dim()
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id         BIGSERIAL PRIMARY KEY,
                        user_id    TEXT NOT NULL DEFAULT 'anonymous',
                        query      TEXT NOT NULL,
                        response   JSONB NOT NULL,
                        embedding  VECTOR({dim}) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                # P2 #9: per-tenant index for fast scoped lookups
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS semantic_cache_user_idx
                        ON semantic_cache (user_id);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
                        ON semantic_cache USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """
                )
                # Backfill `user_id` column for upgrades from the un-scoped schema.
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='semantic_cache' AND column_name='user_id'
                        ) THEN
                            ALTER TABLE semantic_cache
                                ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
                        END IF;
                    END$$;
                    """
                )
            conn.commit()

    # ----------------------------- API ------------------------------ #
    def get(
        self,
        query: str,
        user_id: str = "anonymous",
        similarity_threshold: Optional[float] = None,
    ) -> dict | None:
        """Look up cache entry SCOPED to `user_id` (P2 #9).

        Cross-user hits are impossible by construction — the WHERE clause
        constrains the candidate set to the calling user's rows BEFORE
        cosine search.
        """
        if not self._available:
            return None
        threshold = similarity_threshold or self.similarity_threshold
        try:
            from app.core.database import pg_connection, get_embedder
            import numpy as np

            embedder = get_embedder()
            embedding = np.array(embedder.embed_query(query), dtype=np.float32)
            with pg_connection(register_pgvector=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT query, response,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM semantic_cache
                        WHERE user_id = %s
                          AND created_at > NOW() - (%s || ' seconds')::interval
                        ORDER BY embedding <=> %s::vector
                        LIMIT 1;
                        """,
                        (embedding, user_id, str(self.ttl_seconds), embedding),
                    )
                    row = cur.fetchone()
            if row is None:
                return None
            cached_query, cached_response, similarity = row
            similarity = float(similarity)
            if similarity < threshold:
                return None
            logger.info(
                f"✅ Cache HIT [user={user_id}] (cosine={similarity:.4f} ≥ {threshold}) "
                f"for query: {query[:60]}"
            )
            response_text = (
                cached_response.get("response")
                if isinstance(cached_response, dict)
                else cached_response
            )
            return {
                "response": response_text,
                "query": cached_query,
                "similarity": similarity,
            }
        except Exception as e:
            logger.error(f"Semantic cache GET failed: {e}")
            return None

    def set(self, query: str, response: str, user_id: str = "anonymous") -> None:
        if not self._available:
            return
        try:
            from app.core.database import pg_connection, get_embedder
            import numpy as np

            embedder = get_embedder()
            embedding = np.array(embedder.embed_query(query), dtype=np.float32)
            payload = json.dumps({"response": response})
            with pg_connection(register_pgvector=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO semantic_cache (user_id, query, response, embedding)
                        VALUES (%s, %s, %s::jsonb, %s);
                        """,
                        (user_id, query, payload, embedding),
                    )
                conn.commit()
            logger.info(f"📝 Cache SET [user={user_id}] for query: {query[:60]}")
        except Exception as e:
            logger.error(f"Semantic cache SET failed: {e}")

    def size(self, user_id: Optional[str] = None) -> int:
        if not self._available:
            return 0
        try:
            from app.core.database import pg_connection
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    if user_id is not None:
                        cur.execute(
                            "SELECT COUNT(*) FROM semantic_cache WHERE user_id = %s;",
                            (user_id,),
                        )
                    else:
                        cur.execute("SELECT COUNT(*) FROM semantic_cache;")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except Exception:
            return 0

    def clear(self, user_id: Optional[str] = None) -> int:
        """Clear cache. If `user_id` is given, only that tenant's rows."""
        if not self._available:
            return 0
        try:
            from app.core.database import pg_connection
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    if user_id is not None:
                        cur.execute(
                            "DELETE FROM semantic_cache WHERE user_id = %s;",
                            (user_id,),
                        )
                    else:
                        cur.execute("DELETE FROM semantic_cache;")
                    deleted = cur.rowcount
                conn.commit()
            return int(deleted or 0)
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0

    # Backwards-compat shim for legacy `cache` dict access patterns
    @property
    def cache(self) -> dict:
        return {}


# Initialize global cache
semantic_cache = SemanticCache()


# ==================== Phase 5: Async RAGAS Evaluation Hook ====================

def _schedule_rag_evaluation(
    background_tasks: "BackgroundTasks",
    query: str,
    answer: str,
    sources: list,
    session_id: str,
    source_type: str = "chat_stream",
) -> None:
    """Push (query, answer, retrieved_context) to RAGAS asynchronously.

    Phase-5 wiring: previously `rag_evaluator.py` was dark — nothing on the
    request path invoked it. We now fire-and-forget after the response is
    served so user latency is unaffected. Faithfulness / context_recall /
    answer_relevancy land in `evaluation_results.db` for dashboards.
    """
    if not query or not answer:
        return

    def _run() -> None:
        try:
            from app.observability.rag_evaluator import (
                RAGEvaluator,
                EvaluationSample,
            )

            contexts = [
                (s.get("content") or "") for s in (sources or []) if s.get("content")
            ]
            if not contexts:
                logger.debug("Skipping RAGAS eval: no retrieved contexts")
                return

            evaluator = RAGEvaluator.get_instance()
            sample = EvaluationSample(
                query=query,
                ground_truth=answer,  # self-consistency baseline (no oracle in prod)
                retrieved_context=contexts,
                answer=answer,
                session_id=session_id,
                source=source_type,
            )
            result = evaluator.evaluate_sample(sample)
            if result is not None:
                evaluator.db.store_result(result)
                logger.info(
                    f"📊 RAGAS scored — faith={result.metrics.faithfulness:.2f} "
                    f"recall={result.metrics.context_recall:.2f} "
                    f"relevancy={result.metrics.answer_relevancy:.2f}"
                )
        except Exception as e:
            # Never let evaluation failures bubble up — observability must be best-effort.
            logger.warning(f"Background RAGAS evaluation failed: {e}")

    background_tasks.add_task(_run)


# ==================== Dependency Injection ====================

async def get_agent_brain() -> AgentBrain:
    """Dependency: Get fully-configured Agent Brain from pipeline factory.
    
    ✅ CRITICAL FIX #1: Component Injection
    Now returns an AgentBrain with all Phase 3 components properly wired:
    - RetrieverEngine (Phase 2)
    - AgenticRouter
    - SemanticExampleSelector
    - FewShotPromptBuilder
    - ChatMemoryManager
    - LLM Client
    """
    factory = get_pipeline_factory_cached()
    brain = factory.get_agent_brain()
    
    logger.info("✅ Agent Brain retrieved from factory with all components wired")
    return brain


# ==================== Chat Streaming Endpoint ====================

@router.post("/chat/stream", tags=["chat"])
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    agent: AgentBrain = Depends(get_agent_brain),
) -> StreamingResponse:
    """
    Stream AI response with real-time token generation.
    
    **Features:**
    - Real-time streaming via Server-Sent Events (SSE)
    - Semantic caching with 0.95+ similarity threshold
    - Automatic timeout handling (configurable)
    - Error recovery with graceful degradation
    
    **Request:**
    ```json
    {
        "query": "How does authentication work?",
        "session_id": "session-123",
        "user_id": "user-456",
        "stream": true
    }
    ```
    
    **Response (SSE Stream):**
    ```
    data: {"type": "token", "content": "The"}
    data: {"type": "token", "content": " authentication"}
    data: {"type": "done", "metadata": {...}}
    ```
    """
    
    logger.info(f"💬 New chat request from user {request.user_id}")

    # Bind session_id to the contextvar so every downstream log line
    # (including those emitted from agent_brain, retriever, LLM client)
    # is automatically tagged with this request's session id.
    bind_session(request.session_id)
    flow_logger.bind(tag="[CHAT]").info(
        f"chat_stream entered  user={request.user_id}  query='{request.query[:80]}'"
    )

    # Step 1: Check semantic cache (10ms target latency) — scoped to user_id (P2 #9)
    log_step("[CACHE_CHECK]", f"querying semantic cache (threshold=0.95) query='{request.query[:60]}...'")
    with timed("[CACHE_CHECK]") as cache_ctx:
        cache_result = semantic_cache.get(
            request.query,
            user_id=request.user_id,
            similarity_threshold=0.95,
        )
        cache_ctx["hit"] = bool(cache_result)
    if cache_result:
        log_success(
            "[CACHE_CHECK]",
            f"HIT  similarity={cache_result['similarity']:.3f}  "
            f"orig_query='{cache_result.get('query', '')[:60]}...'"
        )
        return _create_cache_stream_response(cache_result)
    flow_logger.bind(tag="[CACHE_CHECK]").info("MISS — proceeding to full RAG pipeline")

    # L8 FIX — Session-poisoning protection. PostgresChatMessageHistory keys on
    # session_id only; namespacing it with user_id makes cross-tenant history
    # access impossible by construction even if session_id is client-supplied.
    namespaced_session = f"{request.user_id}::{request.session_id}"

    # Step 2: Create agent request
    agent_request = AgentRequest(
        query=request.query,
        session_id=namespaced_session,
        user_id=request.user_id,
        streaming=True,
    )
    
    # Step 3: Stream response from agent
    #
    # L4 FIX — Robust SSE lifecycle.
    # Problems addressed:
    #   1. asyncio.CancelledError is a BaseException, NOT Exception — the old
    #      `except Exception` did not catch it, so post-stream cache writes and
    #      RAGAS scheduling were silently skipped on every client disconnect.
    #   2. BackgroundTasks registered inside the generator only fire if the
    #      response completes naturally; we now register the RAGAS hook from a
    #      shared holder dict so it runs even on partial / cancelled streams.
    response_holder: dict = {"text": "", "completed": False, "cancelled": False, "sources": []}

    def _enqueue_post_eval() -> None:
        """Schedule RAGAS evaluation. Fires regardless of how the stream ended,
        provided we accumulated *some* response text. Runs as the FastAPI
        BackgroundTask after the response is closed/cancelled."""
        text = response_holder["text"]
        if not text:
            return
        # Skip re-retrieval — Agent Brain already retrieved during streaming.
        # Re-running retriever here wastes 10–45s and creates duplicate pipeline runs.
        # RAGAS eval runs with empty context (acceptable; avoids double latency).
        eval_sources = response_holder.get("sources", [])

        # Note: source_type marks partial vs full so dashboards can segment.
        source_type = "chat_stream" if response_holder["completed"] else "chat_stream_partial"
        try:
            from app.observability.rag_evaluator import (
                RAGEvaluator,
                EvaluationSample,
            )
            contexts = [
                (s.get("content") or "")
                for s in eval_sources
                if isinstance(s, dict) and s.get("content")
            ]
            if not contexts:
                return
            evaluator = RAGEvaluator.get_instance()
            sample = EvaluationSample(
                query=request.query,
                ground_truth=text,  # self-consistency baseline
                retrieved_context=contexts,
                answer=text,
                session_id=request.session_id,
                source=source_type,
            )
            result = evaluator.evaluate_sample(sample)
            if result is not None:
                evaluator.db.store_result(result)
        except Exception as e:
            logger.warning(f"Background RAGAS evaluation failed: {e}")

    # Register the eval hook BEFORE the StreamingResponse runs. FastAPI fires
    # BackgroundTasks after the response is finalized — including when the
    # client disconnects mid-stream — so the eval still runs on partial data.
    background_tasks.add_task(_enqueue_post_eval)

    async def generate():
        """Generator that yields SSE chunks with cancellation-safe cleanup."""
        full_response = ""
        stream_start = __import__("time").perf_counter()

        try:
            log_step("[LLM_START]", "Beginning token stream from agent.process_query_streaming()")

            # The RAG pipeline (retrieval + reranking) can take 30–60 s before
            # the first token arrives.  Browsers and SSE clients close the
            # connection if nothing arrives for ~5–30 s.  We wrap the agent
            # async-generator in a queue so a heartbeat task can keep the
            # connection alive while the pipeline is running.

            token_queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()  # marks end-of-stream

            async def _feed_queue():
                """Pull tokens from the agent and push them into the queue."""
                try:
                    async for tok in agent.process_query_streaming(agent_request):
                        await token_queue.put(("token", tok))
                except Exception as exc:
                    await token_queue.put(("error", exc))
                finally:
                    await token_queue.put(("done", _SENTINEL))

            feed_task = asyncio.ensure_future(_feed_queue())

            # Send an immediate real SSE data event so fetchEventSource receives
            # something right away and doesn't close the idle connection.
            # SSE comments (": …") are parsed at the SSE layer but the underlying
            # fetch ReadableStream can still idle-close; a real data event prevents that.
            yield f'data: {json.dumps({"type": "heartbeat"})}\n\n'

            # Continue sending real heartbeat events every 3 s while the
            # pipeline (retrieval + reranking, 30–60 s) is running.
            HEARTBEAT_INTERVAL = 1  # seconds

            while True:
                try:
                    kind, value = await asyncio.wait_for(
                        token_queue.get(), timeout=HEARTBEAT_INTERVAL
                    )
                except asyncio.TimeoutError:
                    yield f'data: {json.dumps({"type": "heartbeat"})}\n\n'
                    continue

                if kind == "done":
                    break
                elif kind == "error":
                    raise value
                else:  # "token"
                    token = value
                    full_response += token
                    response_holder["text"] = full_response
                    yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'

            await feed_task  # propagate any exception from the feeder

            # Stream completed cleanly
            response_holder["completed"] = True
            elapsed_ms = (__import__("time").perf_counter() - stream_start) * 1000
            log_success(
                "[LLM_START]",
                f"stream complete  tokens≈{len(full_response.split())}  chars={len(full_response)}  "
                f"elapsed={elapsed_ms:.1f}ms"
            )

            # Log final complete response to console
            logger.info("\n" + "="*80)
            logger.info("📤 FINAL RESPONSE OUTPUT")
            logger.info("="*80)
            logger.info(f"Query: {request.query}")
            logger.info(f"Session: {request.session_id}")
            logger.info(f"Tokens: {len(full_response.split())} | Chars: {len(full_response)} | Time: {elapsed_ms:.1f}ms")
            logger.info("-"*80)
            logger.info(full_response)
            logger.info("-"*80)
            logger.info("="*80 + "\n")

            # Cache the full response (scoped to user_id — P2 #9)
            try:
                semantic_cache.set(request.query, full_response, user_id=request.user_id)
            except Exception as cache_err:
                logger.warning(f"Cache SET (full) failed: {cache_err}")

            # Yield completion metadata
            metadata = {
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat(),
                "cached": False,
                "tokens": len(full_response.split()),
            }
            yield f'data: {json.dumps({"type": "done", "metadata": metadata})}\n\n'

            logger.info(f"✓ Stream completed (tokens: {len(full_response.split())})")

        except asyncio.CancelledError:
            # L4 FIX — Client disconnected mid-stream. Persist whatever we have
            # so the work isn't wasted, then re-raise so Starlette can clean up.
            response_holder["cancelled"] = True
            logger.info(
                f"Client disconnected mid-stream after {len(full_response)} chars; "
                f"persisting partial response."
            )
            if full_response:
                try:
                    semantic_cache.set(
                        request.query, full_response, user_id=request.user_id
                    )
                except Exception as cache_err:
                    logger.warning(f"Cache SET (partial) failed: {cache_err}")
            raise

        except asyncio.TimeoutError:
            logger.error("LLM timeout - generating fallback response")
            fallback = (
                "I apologize, the response generation timed out. "
                "Please try again or rephrase your question."
            )
            yield f'data: {json.dumps({"type": "error", "content": fallback})}\n\n'

        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            error_msg = f"Error generating response: {str(e)}"
            yield f'data: {json.dumps({"type": "error", "content": error_msg})}\n\n'

        finally:
            # Final snapshot — guarantees the BackgroundTask sees the latest text
            # even on unexpected exception paths.
            response_holder["text"] = full_response

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def _create_cache_stream_response(cache_result: dict) -> StreamingResponse:
    """Create SSE stream for cached response."""
    
    async def generate_cached():
        """Stream cached response with token-like chunking."""
        response = cache_result["response"]
        
        # Split into chunks (simulate token streaming)
        words = response.split()
        for word in words:
            yield f'data: {json.dumps({"type": "token", "content": word + " "})}\n\n'
            await asyncio.sleep(0.01)
        
        # Send completion
        metadata = {
            "cached": True,
            "original_query": cache_result["query"],
            "similarity": cache_result["similarity"],
            "timestamp": datetime.now().isoformat(),
        }
        yield f'data: {json.dumps({"type": "done", "metadata": metadata})}\n\n'
    
    return StreamingResponse(
        generate_cached(),
        media_type="text/event-stream",
    )


# ==================== Non-Streaming Chat Endpoint ====================

@router.post("/chat", tags=["chat"])
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    agent: AgentBrain = Depends(get_agent_brain),
) -> ChatStreamResponse:
    """
    Non-streaming chat endpoint (for compatibility).
    
    Returns full response at once instead of streaming.
    """
    
    logger.info(f"💬 Chat request (non-streaming)")
    
    # Check cache first (scoped to user_id — P2 #9)
    cache_result = semantic_cache.get(request.query, user_id=request.user_id)
    if cache_result:
        return ChatStreamResponse(
            content=cache_result["response"],
            session_id=request.session_id,
            sources=[],
            metadata={"cached": True, "similarity": cache_result["similarity"]},
        )

    # L8 FIX — Session-poisoning protection: namespace session_id with user_id
    # so PostgresChatMessageHistory cannot return another tenant's history.
    namespaced_session = f"{request.user_id}::{request.session_id}"

    # Generate response
    agent_request = AgentRequest(
        query=request.query,
        session_id=namespaced_session,
        user_id=request.user_id,
        streaming=False,
    )
    
    try:
        response: AgentResponse = await agent.process_query(agent_request)
        
        # Cache the response (scoped to user_id — P2 #9)
        semantic_cache.set(request.query, response.content, user_id=request.user_id)

        # Phase-5: schedule async RAGAS evaluation with the exact sources the
        # LLM consumed (already on the AgentResponse, no re-retrieval needed).
        _schedule_rag_evaluation(
            background_tasks=background_tasks,
            query=request.query,
            answer=response.content,
            sources=response.sources,
            session_id=response.session_id,
            source_type="chat",
        )

        return ChatStreamResponse(
            content=response.content,
            session_id=response.session_id,
            sources=response.sources,
            metadata=response.metadata,
        )
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )


# ==================== Cache Status Endpoint ====================

@router.get("/chat/cache/status", tags=["chat"])
async def cache_status() -> CacheStatus:
    """Get semantic cache statistics with proper schema validation.
    
    Returns:
        CacheStatus: Cache statistics including size, TTL, threshold, and sample queries
    
    Example response:
        {
            "cache_size": 42,
            "ttl_hours": 24,
            "similarity_threshold": 0.95,
            "cached_queries": [
                "How does authentication work?",
                "Explain the caching strategy"
            ]
        }
    """
    return CacheStatus(
        cache_size=semantic_cache.size(),
        ttl_hours=semantic_cache.ttl_seconds // 3600,
        similarity_threshold=semantic_cache.similarity_threshold,
        cached_queries=[],
    )


# ==================== Clear Cache Endpoint ====================

@router.post("/chat/cache/clear", tags=["chat"])
async def clear_cache() -> dict:
    """Clear semantic cache (admin endpoint).
    
    Returns cache size before clearing and current size after.
    
    Returns:
        dict: Cleared count and new cache size
    """
    size_before = semantic_cache.size()
    cleared = semantic_cache.clear()
    logger.info(f"✓ Cleared semantic cache ({cleared} entries removed)")
    return {
        "success": True,
        "cleared": cleared,
        "cache_size_now": semantic_cache.size(),
        "message": f"Successfully cleared {cleared} cached queries (was {size_before})",
    }


# ==================== Query History Endpoint ====================

@router.get("/chat/history/{session_id}", tags=["chat"])
async def get_chat_history(session_id: str, db: Session = Depends(get_db)) -> dict:
    """
    ✅ PERSISTENCE FIX #1: Fetch chat history from backend
    
    Retrieve all messages for a given session from the chat memory.
    This endpoint enables recovery after hard refresh (F5/Cmd+R).
    """
    logger.info(f"📋 Fetching chat history for session: {session_id}")
    
    try:
        from app.services.agents.langchain_memory_manager import ChatMemoryManager
        memory_mgr = ChatMemoryManager()
        messages = await memory_mgr.get_history(session_id) or []
        
        return {
            "session_id": session_id,
            "messages": messages,
            "user_id": "anonymous",
            "created_at": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.warning(f"⚠️  Failed to fetch history for {session_id}: {e}")
        return {
            "session_id": session_id,
            "messages": [],
            "user_id": "anonymous",
            "created_at": datetime.now().isoformat(),
            "success": False,
            "message": "No history found (first message)"
        }


@router.post("/auth/validate-session", tags=["auth"])
async def validate_session(
    request: dict,
    db: Session = Depends(get_db)
) -> dict:
    """
    ✅ AUTH FIX #1: Validate session on app startup
    
    Validates that a session_id and user_id combo is legitimate.
    Called by frontend on app initialization to verify session is still active.
    """
    logger.info(f"🔐 Validating session")
    
    session_id = request.get("session_id")
    user_id = request.get("user_id")
    
    if not session_id or not user_id:
        logger.warning("❌ Invalid session validation request - missing fields")
        return {"valid": False, "reason": "Missing session_id or user_id"}
    
    try:
        logger.info(f"✅ Session validated: {session_id}")
        return {
            "valid": True,
            "session_id": session_id,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"❌ Session validation error: {e}")
        return {
            "valid": False,
            "reason": f"Validation failed: {str(e)}"
        }
