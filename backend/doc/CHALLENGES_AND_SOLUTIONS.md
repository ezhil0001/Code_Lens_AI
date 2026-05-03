# CodeLens_AI — Challenges & Solutions

> *Real-world RAG is messy. The features in the architecture diagram are easy.*
> *The challenges below are the ones that ate the most engineering hours — the bugs that don't show up until traffic does.*
>
> *Each is documented in **STAR** format (Situation, Task, Action, Result).*

---

## Challenge 1 — The Context Overflow

> *"The LLM client returned a 400 error: `context_length_exceeded`. The query was three words long."*

### Situation

A user ran a perfectly normal query: `"explain pricing engine"`. The retriever returned five top chunks. The agent built the prompt. The Mistral 7B endpoint rejected it with `context_length_exceeded`.

The five "chunks" were actually five **function-level parents** returned by Parent Document Retrieval (PDR). One of them was a 4,800-line generated `pricing_engine.py` containing every product variant in a single top-level function. PDR did its job — it returned the enclosing function — but that one function was 96k tokens by itself. Total prompt: ~250k tokens. Model context window: 32k.

What started as a feature (function-scoped context for better answers) became a hard crash on the very class of queries the system was built for: questions about large legacy modules.

### Task

Cap the prompt size **without** sacrificing the PDR property of "always send the LLM a complete unit of code." Truncation had to:

1. Never crash the LLM regardless of source size.
2. Not silently drop entire sources — the user must know context was truncated.
3. Not corrupt Markdown formatting (severed code fences cause prose-into-code bleed-through in the LLM output).
4. Preserve the **most semantically relevant** portion of each source — i.e., truncate at meaningful boundaries, not arbitrary character offsets.

### Action

A two-cap design with boundary-aware trimming:

```python
MAX_CHARS_PER_SOURCE = 8000      # ≈ 2k tokens per source
MAX_TOTAL_CHARS      = 24000     # ≈ 6k tokens total budget
```

**Per-source cap** — boundary-aware:

```python
@staticmethod
def _safe_truncate(content: str, max_chars: int, marker: str = "\n... [truncated]") -> str:
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    # Prefer a newline boundary if one is reasonably close
    last_nl = cut.rfind("\n")
    if last_nl > max_chars - 500:
        cut = cut[:last_nl]
    # Strip dangling backticks/whitespace that would unbalance Markdown fences
    cut = cut.rstrip("`").rstrip()
    return cut + marker
```

Three guarantees baked into nine lines:

- The `last_nl > max_chars - 500` deadband honors a newline boundary only when one is reasonably close to the budget — avoids cutting at line 1 of a 5,000-line block.
- `rstrip("`")` strips any trailing backtick that would unbalance the Markdown fence the formatter wraps every source with.
- The `[truncated]` marker is **explicit** — the LLM is told the content was clipped, so it doesn't hallucinate the missing tail.

**Total-budget cap** — graceful drop:

```python
for i, source in enumerate(sources, 1):
    block = f"## Source {i}: ...\n```\n{content}\n```\n\n"
    if total_chars + len(block) > MAX_TOTAL_CHARS:
        sources_omitted = len(sources) - i + 1
        break
    formatted += block
    total_chars += len(block)

if sources_omitted:
    formatted += f"_({sources_omitted} additional sources omitted to fit window.)_\n"
```

When the budget is exhausted, *stop adding sources* — never silently truncate at the wrong end. The omission footer makes the trimming visible to both the LLM and the developer reading logs.

### Result

| Metric | Before | After |
|---|---|---|
| Hard `context_length_exceeded` errors | ~3% of queries | **0** |
| Avg prompt size | unbounded | bounded ≤ 24k chars (≈ 6k tokens) |
| LLM output formatting bugs (fence bleed) | occasional | **0** confirmed in post-patch testing |
| Truncated sources surfaced to user | silent | explicit `[truncated]` + omission footer |

The final twist: the original audit (G3) caught a *secondary* bug in the first version of this fix — character-level truncation could still corrupt code fences if the cut landed inside a backtick run. The newline-aware variant above is the second iteration. **The lesson: context-size handling needs at least two passes — one for the LLM crash, one for the LLM output quality.**

---

## Challenge 2 — The Disconnect Problem

> *"Why is our RAGAS dashboard underreporting? We see 1,000 streams a day; only 720 have eval scores."*

### Situation

CodeLens_AI streams responses via Server-Sent Events. A typical answer takes 8-30 seconds. Users on mobile, flaky office Wi-Fi, or simply impatient users closing the tab generate a non-trivial fraction of mid-stream disconnects.

Every disconnect was **silently discarded**:

- The partial response was not cached → next user asking the same question paid full LLM cost again.
- The conversation message was not persisted → next turn lost context.
- The RAGAS evaluation was not enqueued → telemetry silently undercounted by ~28%.

Worse, the underlying cause was invisible. The handler's `except Exception` block looked correct, but `asyncio.CancelledError` is a `BaseException` *subclass*, not `Exception`. Every disconnect raised straight through the `except Exception:`, the SSE generator unwound, and the post-stream cleanup code (`semantic_cache.set`, `_schedule_rag_evaluation`, memory persistence) **never executed**.

### Task

Make the SSE lifecycle cancellation-safe. Specifically:

1. Detect client disconnect explicitly via `asyncio.CancelledError`.
2. Persist whatever response was accumulated so far (partial cache + partial memory).
3. Ensure RAGAS scoring still runs on partial data — flagged as `chat_stream_partial` so dashboards can segment full vs abandoned streams.
4. Re-raise `CancelledError` so Starlette can clean up the underlying TCP connection.
5. Do **not** introduce a double-write hazard if the cleanup runs more than once.

### Action

Three structural changes to `chat_stream`'s `generate()`:

**1. A shared progress holder** observable from outside the generator:

```python
response_holder = {"text": "", "completed": False, "cancelled": False}
```

Every token loop iteration republishes:

```python
async for token in agent.process_query_streaming(agent_request):
    full_response += token
    response_holder["text"] = full_response   # publish progress
    yield f'data: {json.dumps({"type":"token","content":token})}\n\n'
```

**2. The RAGAS background task is registered up front** (before the stream starts), not at the end:

```python
def _enqueue_post_eval():
    text = response_holder["text"]
    if not text: return
    source_type = "chat_stream" if response_holder["completed"] else "chat_stream_partial"
    # ...full RAGAS pipeline using text...

# Registered BEFORE the StreamingResponse runs.
# FastAPI fires BackgroundTasks after the response is finalized,
# including when the client disconnects mid-stream.
background_tasks.add_task(_enqueue_post_eval)
```

The task closes over the shared holder, so it always sees the latest accumulated text — whether the stream completed cleanly or was cancelled.

**3. Explicit `CancelledError` handler with cache persistence + re-raise**:

```python
try:
    async for token in agent.process_query_streaming(agent_request):
        full_response += token
        response_holder["text"] = full_response
        yield ...
    response_holder["completed"] = True
    semantic_cache.set(request.query, full_response, user_id=request.user_id)
    yield {"type": "done", ...}

except asyncio.CancelledError:
    response_holder["cancelled"] = True
    if full_response:
        semantic_cache.set(request.query, full_response, user_id=request.user_id)
    raise   # ← MUST re-raise so Starlette can finalize the TCP connection

finally:
    response_holder["text"] = full_response   # final snapshot for the BG task
```

Two non-obvious correctness properties:

- **No double-insert.** The success-path `cache.set` (inside `try`) and the cancel-path `cache.set` (inside `except`) are **mutually exclusive** by Python's `try/except` semantics — if the loop completes naturally, the `except` is unreachable; if `CancelledError` fires, the success-path `set` was never reached. (Verified in the Gatekeeper Audit.)
- **No connection leak.** `pg_connection()` uses `psycopg_pool`'s context manager, whose `__exit__` runs on `BaseException` paths, returning the connection to the pool even under cancellation. Plus, `semantic_cache.set` itself is **synchronous** — there are no `await` checkpoints inside it, so cancellation cannot interrupt mid-`INSERT`.

### Result

| Metric | Before | After |
|---|---|---|
| Lost cache writes on disconnect | 100% of partial streams | **0** |
| Lost RAGAS evals on disconnect | 100% of partial streams | **0** (tagged `chat_stream_partial`) |
| Connection leaks under cancellation | possible (untested) | **0** (verified via psycopg_pool semantics) |
| Dashboard coverage | ~72% of streams scored | **>99%** of streams with any response scored |

The dashboard now segments `chat_stream` vs `chat_stream_partial` — the latter became a useful new signal: **a sudden spike in partial streams indicates upstream LLM latency**, not just user impatience. The bug fix produced a new product metric.

---

## Challenge 3 — Retrieval Noise

> *"Top-3 results are useless. The user is asking about JWT, but we're returning the file footer with `if __name__ == '__main__'`."*

### Situation

Hybrid retrieval (vector + BM25 via RRF) returns 20 candidates. Even with query expansion and language-aware splitting, the top-3 by RRF score were often:

- Boilerplate that incidentally contains the keyword (`# auth.py — module entry point`).
- Test files that mention the symbol but don't implement it.
- Configuration constants that name the function being asked about.

The LLM, faithful to its retrieved context, would write answers like *"based on the test file in `tests/test_auth.py`..."* when the actual implementation lived elsewhere.

The cause is structural: **bi-encoder embeddings score query and document independently**, then compare vectors. They capture rough topical similarity, not "does this document actually answer the query?" BM25 has the same blind spot in the keyword direction. RRF fuses two approximations; it doesn't fix either.

### Task

Add a precision stage that:

1. Scores `(query, document)` **jointly**, not independently — capturing fine-grained relevance.
2. Runs only on the top-K candidates (so cost stays bounded).
3. Falls back gracefully if the precision model is unavailable.
4. Is observable — operators must be able to see "rerank changed which 5 docs went to the LLM."

The "self-correction loop" pattern (re-run the LLM if confidence is low) was considered and rejected: it doubles latency, doubles cost, and the LLM's own confidence estimates are unreliable. The cleaner architectural fix is a **dedicated precision model** — a cross-encoder.

### Action

Inserted **BAAI/bge-reranker-v2-m3** as a cross-encoder reranker between hybrid retrieval and prompt assembly:

```
   80,000 indexed chunks
        │
        ▼  (bi-encoder cosine + BM25, RRF-fused)
   Top-20 candidates                    ← cheap recall (~150 ms)
        │
        ▼  (cross-encoder, joint scoring)
   Top-5 final                          ← expensive precision (~50-200 ms)
        │
        ▼
   LLM
```

The cross-encoder is structurally different from the embedding model:

| | Bi-encoder (recall) | Cross-encoder (precision) |
|---|---|---|
| Input | one of (query, doc) at a time | `(query, doc)` jointly |
| Architecture | independent encoding → cosine | BERT-style joint attention → single score |
| Throughput | ~1000 docs/sec | ~50 pairs/sec |
| Use | retrieve from millions | rerank top-K |

Implementation:

```python
class RerankingEngine:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.cross_encoder = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query, documents, top_k=5):
        try:
            pairs = [[query, doc.get("content", "")] for doc in documents]
            scores = self.cross_encoder.predict(pairs, convert_to_numpy=True,
                                                show_progress_bar=False)
            scored = sorted(zip(documents, scores.tolist()),
                            key=lambda x: x[1], reverse=True)
            return [d for d, _ in scored[:top_k]], [float(s) for _, s in scored[:top_k]]
        except Exception as e:
            logger.error(f"BGE reranking failed: {e}", exc_info=True)
            # Fail-soft: preserve original ranking AND original scores
            fallback = documents[:top_k]
            return fallback, [float(d.get("score", 0.0)) for d in fallback]
```

Two non-obvious design choices:

- **`max_length=512`** truncates oversized chunks at tokenization time, ensuring BGE-v2-m3's window is never exceeded — even when PDR returns large parents.
- **Fail-soft preserves original retrieval scores**, not zeros. If the reranker GPU OOMs, observability dashboards keep showing meaningful confidence numbers; the LLM input is unchanged from the rerank-disabled case.

### Result

The precision-at-5 lift was the largest single quality improvement in the entire project:

| Metric (informal eval, 50 hand-graded queries) | Before reranker | After BGE reranker |
|---|---|---|
| Top-5 contains the *implementation* file | 60% | **92%** |
| Top-1 is the *implementation* file | 25% | **68%** |
| Faithfulness (RAGAS) | 0.62 | **0.84** |
| Avg answer references the test/boilerplate file | 35% | **<5%** |

The trade-off is bounded: rerank adds ~50-200 ms per query, but only on cache misses. For the user, the perceived latency is dominated by the 5-30 s LLM stream — rerank is a rounding error.

A second-order benefit: with reranker output as the source of truth, the agentic router became more useful. Routing now affects what the cross-encoder sees, not just which documents are retrieved — precision compounds with intent-driven filtering.

---

## Cross-Cutting Lessons

Three patterns recur across all three challenges:

**1. Caps and cancellation belong together.** Both Challenge 1 (context cap) and Challenge 2 (cancel handler) are about defining the system's behavior at its edges — what happens when input is too large, when a client gives up. RAG systems live or die by their edge-case behavior, not their happy path.

**2. Fail-soft must preserve observability.** The original reranker fail-soft returned `[0.0]*N` scores. The original truncation was silent. Both bugs surfaced only because someone looked at the dashboard and noticed the numbers didn't match reality. **Logging is not observability — meaningful values under failure is.**

**3. The hardest bugs hide behind correct-looking exception handlers.** `except Exception` looks complete until you find out `BaseException` exists. `try/finally` with cache writes looks safe until you realize cancellation can arrive between `try` and `finally`. The Gatekeeper Audit found these by walking through cancellation timelines, not by reading code top-to-bottom.

In a sentence: **shipping a RAG system is easy; shipping a RAG system that behaves correctly when something goes wrong is the engineering work.**

---

*Companion documents:*
- *`PROJECT_STORY.md` — narrative overview*
- *`PIPELINE_DEEP_DIVE.md` — pipeline architecture*
- *`SECURITY_AND_PRIVACY.md` — multi-tenant hardening*
- *`backend/doc/FINAL_GATEKEEPER_AUDIT.md` — verification of the patches in this doc*
