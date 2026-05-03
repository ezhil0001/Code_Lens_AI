# CodeLens_AI — Pipeline Deep Dive

> A line-of-sight technical walkthrough of the five-phase RAG pipeline:
> **Ingestion → Hybrid Retrieval → Reranking → Agentic Reasoning → Semantic Cache.**

---

## Pipeline at a glance

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST LIFECYCLE                                │
└──────────────────────────────────────────────────────────────────────────────┘

   Client ─► /chat/stream
              │
              ▼
   ┌─────────────────────┐    HIT    ┌──────────────────────────────────┐
   │ ⓹ Semantic Cache    │ ────────► │ Stream cached tokens via SSE     │
   │   (pgvector + ANN)  │           │ (sub-100 ms total latency)       │
   └──────────┬──────────┘           └──────────────────────────────────┘
              │ MISS
              ▼
   ┌─────────────────────┐
   │ ⓸ Agentic Router    │ ◄────── conversation history + intent
   │   decides where to  │
   │   look (code/docs/  │
   │   hybrid)           │
   └──────────┬──────────┘
              │ metadata_filter
              ▼
   ┌─────────────────────┐
   │ ⓶ Hybrid Retrieval  │ ◄────── ⓵ Ingestion artifacts:
   │   Vector + BM25     │           ChromaDB (vectors)
   │   via weighted RRF  │           BM25 corpus
   │   → top-20          │           parent_store (function-level)
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ ⓷ BGE Cross-Encoder │
   │   rerank to top-5   │
   │   + parent context  │
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ ⓸ Prompt assembly   │ + few-shot examples (cosine-selected)
   │   ─► LLM stream     │
   └──────────┬──────────┘
              ▼
   Client ◄── SSE tokens
              │
              ▼
   ⓹ Cache.set(user_id-scoped)  +  ⓹ BackgroundTasks → RAGAS scoring
```

Each numbered phase is dissected below.

---

## Phase 1 — Advanced Ingestion: AST > Naive Splitting

### Why naive splitting breaks for code

A `RecursiveCharacterTextSplitter` with default separators (`\n\n`, `\n`, ` `, `""`) treats source code like prose. Run it on a 200-line Python file and you get:

```
Chunk 7 (last 30 lines):
    def authenticate(token: str) -> User:
        if not token:
            raise InvalidT  ← BOUNDARY HERE

Chunk 8 (first 30 lines):
    okenError("Empty token")
        decoded = jwt.decode(...
```

The LLM sees `InvalidT` and `okenError` as separate tokens. When prompted "explain the auth flow", it confidently hallucinates a different function. **The bug isn't the LLM — it's the chunker.**

### Language-aware splitting

LangChain's `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)` ships separators tuned per language:

```python
PYTHON_SEPARATORS = ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
JS_SEPARATORS     = ["\nfunction ", "\nconst ", "\nclass ", "\n\n", "\n", ...]
```

Splits prefer `class` and `def` boundaries over arbitrary character counts. Output:

```
Chunk 7 ends at "raise InvalidTokenError(...)"
Chunk 8 starts at "    decoded = jwt.decode(...)"
```

Whole statements survive. **This is the floor; AST-based parents are the ceiling.**

### AST-based parent extraction (the actual win)

Even language-aware splitting can produce a 30-line chunk that's *part of* a 150-line function. The LLM still gets fragments. The fix is **Parent Document Retrieval (PDR)**: store small chunks for retrieval, return the *enclosing function* for context.

```
┌──────────────────────────────────────────────────────────┐
│  Source file (full text)                                 │
└────────────────────┬─────────────────────────────────────┘
                     │
            ast.parse(source)          ← Python: stdlib, zero deps
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   tree.body iteration      regex + brace balancer
   (Python)                 (JS / TS)
        │                         │
        └────────────┬────────────┘
                     ▼
   For each top-level FunctionDef / ClassDef:
       parent_id = "parent::<source>::<name>::<start>-<end>"
       parent_store[parent_id] = full_function_body
                     │
                     ▼
   Children (split chunks) get metadata.parent_id
   pointing to their enclosing function.
```

**Concrete payoff:**

| Strategy | Avg context size | Hallucination rate (informal eyeball) |
|---|---|---|
| Naive char split | ~400 tokens — fragments | High (severed signatures) |
| Language-aware split | ~400 tokens — whole statements | Medium |
| **PDR with AST parents** | ~1500 tokens — whole functions | **Low** — LLM sees the unit a developer would read |

### Why `ast.parse` (Python) and regex (JS/TS), not tree-sitter

- **`ast`** ships with Python, parses identical to the interpreter, zero install.
- **tree-sitter** is more accurate for JS/TS but requires Node.js and language grammars in the container — operationally heavier.
- The brace-balancing regex covers `function foo(...)`, `class Bar`, and `const baz = (...) =>` — the three dominant idioms. Edge cases (braces inside strings) fall through to the **file-level parent fallback**, so nothing is ever lost.

---

## Phase 2 — Hybrid Retrieval: Vector + BM25 with Weighted RRF

### Why neither alone is sufficient

| Query | Best retriever | Why |
|---|---|---|
| `"authenticate(token: str)"` | **BM25** | exact symbol; vector embedding flattens type annotations |
| `"how do we handle expired sessions"` | **Vector** | conceptual; BM25 misses paraphrases |
| `"how does authenticate handle expired tokens"` | **Both** | needs symbol *and* concept |

Real developer queries are the third row. Hybrid is not a luxury — it's required for usable recall.

### The fusion mechanism — weighted Reciprocal Rank Fusion

```
   ┌──────────────────────┐         ┌──────────────────────┐
   │ Vector retriever     │         │ BM25 retriever       │
   │ (ChromaDB, k=20)     │         │ (in-memory corpus,   │
   │                      │         │  k=20)               │
   └──────────┬───────────┘         └──────────┬───────────┘
              │                                │
              ▼                                ▼
        Ranked list V₁..V₂₀           Ranked list B₁..B₂₀
              │                                │
              └──────────┬─────────────────────┘
                         ▼
              ┌────────────────────────────┐
              │  EnsembleRetriever (RRF)   │
              │                            │
              │  score(d) =                │
              │      w_v / (k + rank_v(d)) │
              │    + w_b / (k + rank_b(d)) │
              │                            │
              │  default: w_v=0.6, w_b=0.4 │
              │  k = 60 (LangChain default)│
              └────────────┬───────────────┘
                           ▼
                  Fused top-N candidates
```

**Why RRF over score averaging:** vector scores are cosine in `[-1, 1]`; BM25 scores are unbounded positive. They're not commensurate. RRF only uses *ranks*, which are comparable across any retriever. This is why every serious hybrid system (LangChain, Vespa, Elasticsearch RRF, Weaviate) uses it.

### Dynamic weight tuning

A `QueryIntentDetector` inspects the query and proposes per-query weights:

```
   Query: "authenticate(token)"
       └─► IntentDetector → exact_lookup → vector=0.4, bm25=0.6  (boost BM25)

   Query: "explain the architecture"
       └─► IntentDetector → conceptual    → vector=0.8, bm25=0.2  (boost vector)
```

If the proposed weights drift more than 0.05 from the running default, the `EnsembleRetriever` is rebuilt for that one query. The 0.05 deadband prevents pointless rebuilds.

### Routing-driven metadata filter

The *agentic router* (Phase 4) decides "this query is code-only" before retrieval runs. That decision becomes a Chroma `where=` filter:

```python
RoutingDecision.CODEBASE_ONLY → {"file_type": "code"}
RoutingDecision.KT_ONLY       → {"file_type": "kt_doc"}
RoutingDecision.HYBRID        → None
```

Vector retrieval applies it natively (`collection.query(where=...)`); BM25 has no native filter so we **post-filter** its hits in Python. The filter mutation is **lock-protected** — under concurrent load, two requests must not race on the shared retriever's filter attribute.

```
  with self._filter_lock:                       ◄── thread-safety guarantee
      previous_filter = vector_retriever.metadata_filter
      vector_retriever.metadata_filter = metadata_filter
      docs = ensemble.invoke(query)             ◄── INSIDE the lock
      vector_retriever.metadata_filter = previous_filter
  # BM25 post-filter happens outside the lock (operates on local list)
```

### Query expansion for recall

Before retrieval, a `QueryExpander` produces 3 variants:

```
   "authenticate token expiry"
        ├── variant 1: "authenticate token expiry"             (original)
        ├── variant 2: "JWT token validation expired"          (concept)
        └── variant 3: "verify_token expired_at decode_jwt"    (technical)
```

Each variant runs through hybrid retrieval; results are deduplicated and merged. Recall improves; precision is recovered in Phase 3.

---

## Phase 3 — Reranking with BGE Cross-Encoder

### Bi-encoder vs cross-encoder — the precision/cost trade-off

```
   BI-ENCODER (the embedding model)
   ┌──────────┐                          ┌──────────┐
   │  query   │ ── encode ─► v_query     │ document │ ── encode ─► v_doc
   └──────────┘                          └──────────┘
                       cosine(v_query, v_doc)
                       │
                       ▼
                 INDEPENDENT scoring — fast, approximate.
                 Trained for "what's roughly similar?"

   CROSS-ENCODER (BAAI/bge-reranker-v2-m3)
   ┌────────────────────────────────────────────────┐
   │  [CLS] query [SEP] document [SEP]              │
   └────────────────────┬───────────────────────────┘
                        ▼
                 BERT-style joint encoding
                        │
                        ▼
                 single relevance score
                        │
                        ▼
                 JOINT scoring — slow, accurate.
                 Trained for "is this document the answer?"
```

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Throughput | ~1000 docs/sec | ~50 pairs/sec |
| Accuracy on hard cases | Baseline | +20-30% NDCG@5 |
| Use for | Initial recall over millions | Precision over top-K candidates |

### The 20→5 pipeline

```
   Hybrid retrieval                   BGE Cross-Encoder
   ────────────────                   ─────────────────
   80,000 chunks ─► top-20            top-20 ─► (query, doc) pairs ─► top-5
                    (~150 ms)                   (~50-200 ms)

   Cheap recall                       Expensive precision

   Fail-soft on cross-encoder error: return top-5 by ORIGINAL retrieval score
   (preserves observability metrics — never zeros).
```

`max_length=512` truncates oversized chunks at tokenization time so BGE-v2-m3's window is never exceeded.

### Parent context attachment (PDR realization)

```
   For each top-5 chunk:
       ├── parent_id present? ─► fetch from parent_store
       │                          (the enclosing function/class body)
       │                          ─► becomes "content" sent to LLM
       └── parent_id missing? ─► fall back to literal chunk text
```

The LLM gets ~5 functions, not ~5 line fragments. **This is where retrieval quality becomes generation quality.**

---

## Phase 4 — The Agentic Layer

### Decision flow

```
  User query
      │
      ▼
  ┌───────────────────────┐
  │ Conversation history  │ ◄── ChatMemoryManager (Postgres-backed,
  │ (last ~2k tokens)     │      pooled connections, per-op checkout)
  └──────────┬────────────┘
             │
             ▼
  ┌───────────────────────┐
  │ Intent detection      │ ◄── QueryIntentDetector
  │  • exact_lookup       │      (regex + heuristics)
  │  • conceptual         │
  │  • code_navigation    │
  │  • mixed              │
  └──────────┬────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ Routing decision                        │
  │                                         │
  │  intent × history × surface →           │
  │      CODEBASE_ONLY                      │
  │      KT_ONLY                            │
  │      HYBRID                             │
  │      AGENT_TOOL                         │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ Translation to data-layer constraint    │
  │   routing_decision_to_metadata_filter() │
  │      ─► {"file_type": "code"}           │
  │      ─► {"file_type": "kt_doc"}         │
  │      ─► None  (HYBRID — no filter)      │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ Phase 2 retrieval with metadata_filter  │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ Few-shot example selection              │
  │   cosine(query_emb, example_emb)        │
  │   top-2 from curated Q&A bank           │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ Prompt assembly                         │
  │   + system instructions                 │
  │   + few-shot exemplars                  │
  │   + retrieved context (boundary-aware   │
  │      truncated; per-source 8 KB cap;    │
  │      total 24 KB cap)                   │
  │   + Pydantic format instructions        │
  │     (forces JSON output)                │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ LLM stream → SSE tokens to client       │
  │   AnswerSchema(answer, sources,         │
  │                confidence_score)        │
  └──────────┬──────────────────────────────┘
             │
             ▼
  Persist user + assistant messages to memory (pooled checkout)
```

### Why this is "agentic" and not just "LLM with retrieval"

- **The router is a real classifier** — its decision changes which corpus is searched, not just which prompt template is used.
- **The routing decision becomes a SQL/Chroma constraint** — soft "I'll think about it" routing is theatre; CodeLens_AI's routing flips a `where=` clause.
- **History changes routing** — a follow-up like "what about the refresh path?" inherits the prior turn's surface (KT vs code).
- **Structured output is enforced** — `PydanticOutputParser` parses every response into `AnswerSchema`; malformed JSON triggers a retry, not a silent failure.

### The single-pipeline contract

Both streaming and non-streaming endpoints converge on **`_run_core_pipeline`** — one function, one source of truth. Drift between the two paths was the bug class that produced silent regressions in the past; consolidating eliminated it.

---

## Phase 5 — Semantic Cache

### The latency story

Without cache, every request pays:

```
  cache lookup (skipped)             0 ms
  history fetch                    ~10 ms
  intent + routing                  ~1 ms
  hybrid retrieval                ~200 ms
  cross-encoder rerank             ~150 ms
  few-shot selection                ~15 ms
  LLM streaming               ~5,000-30,000 ms
  ──────────────────────────────────────
  total                       ~5-30 seconds
```

A cache hit on a paraphrased question collapses this to **<100 ms total**. The cache is the difference between "an AI demo" and "an interactive tool you'd use ten times an hour."

### Architecture — pgvector with multi-tenant scoping

```
   POST /chat/stream
       │
       ▼
   ┌───────────────────────────────────────────────────┐
   │  embedder = get_embedder()      ◄── singleton     │
   │  q_emb = embedder.embed_query(query)              │
   │  with pg_connection() as conn:  ◄── pooled        │
   │      SELECT response                              │
   │      FROM   semantic_cache                        │
   │      WHERE  user_id = %s                          │
   │        AND  created_at > NOW() - interval         │
   │      ORDER  BY embedding <=> %s::vector           │
   │      LIMIT  1;                                    │
   │                                                   │
   │  HIT iff (1 - cosine_distance) ≥ 0.95             │
   └───────────────────────────────────────────────────┘
```

### Schema and indexing

```sql
CREATE TABLE semantic_cache (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'anonymous',
    query       TEXT NOT NULL,
    response    JSONB NOT NULL,
    embedding   VECTOR(768) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX semantic_cache_user_idx
    ON semantic_cache (user_id);                       -- B-tree (tenant filter)

CREATE INDEX semantic_cache_embedding_idx
    ON semantic_cache USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);                                -- IVFFlat (ANN search)
```

### Why `WHERE` runs **before** `ORDER BY`

```
  PostgreSQL planner execution order:
       1. WHERE user_id = 'A'  ──► uses semantic_cache_user_idx (B-tree)
       2. Filtered candidate set fed to IVFFlat scan
       3. ORDER BY embedding <=> $q   (cosine distance)
       4. LIMIT 1
```

Cross-tenant rows are **unreachable by construction**. Even if user B's row would be a closer cosine match, the planner cannot return it — the candidate set never contains it.

This is why the cache is multi-tenant-safe: not because we trust the application code, but because the SQL planner enforces isolation at the index layer.

### Threshold tuning

```
  threshold = 0.95   ◄── default; only near-paraphrases hit
                       (low hit rate, high precision)

  threshold = 0.85   ◄── more hits, occasional stale answer
                       (suitable for high-volume read patterns)

  threshold = 0.75   ◄── too loose; risk serving wrong answer to
                       superficially similar queries
```

We hold 0.95 in production. Lowering to 0.85 was tested and showed ~3× hit rate at the cost of one user-visible "this doesn't quite answer my question" per ~50 hits — not worth it.

### Cache TTL

Default `created_at > NOW() - interval '86400 seconds'` (24 h). Code repos churn; a cached answer on yesterday's API isn't useful tomorrow. The TTL is enforced inside the SQL query so expired rows never participate in ranking.

### What the cache deliberately does NOT do

- **No LRU eviction.** Old rows stay until a vacuum job sweeps them — the IVFFlat index handles the size. Avoids cache stampede.
- **No write-on-stream-cancel.** If the LLM stream is cancelled mid-generation, the partial response *is* still cached (intentional — see audit) but tagged as `chat_stream_partial` for the RAGAS evaluator to flag.
- **No global cache.** Per-tenant always. Cross-tenant cache hits are a privacy violation, not a feature.

---

## Closing — How the phases compose

```
  Ingestion produces:                  Retrieval consumes:
  ──────────────────────               ───────────────────
  ChromaDB collection ─────────────►   vector retriever
  BM25 corpus         ─────────────►   BM25 retriever
  parent_store        ─────────────►   PDR attachment in Phase 2 → 3 boundary

  Routing produces:                    Retrieval respects:
  ──────────────────────               ───────────────────
  metadata_filter     ─────────────►   Chroma where= + BM25 post-filter

  Retrieval produces:                  Agent consumes:
  ──────────────────────               ───────────────────
  top-5 chunks +      ─────────────►   prompt context (truncated, fence-safe)
  parent contexts

  Agent produces:                      Cache + Eval consume:
  ──────────────────────               ───────────────────
  full response       ─────────────►   semantic_cache.set(user_id-scoped)
                      ─────────────►   BackgroundTasks → RAGAS scoring
```

Every phase's output is the next phase's input under a well-defined contract (metadata schema, score normalization, response shape). That contract — not the individual phase quality — is what makes the system maintainable. Replace any single phase (swap ChromaDB for Qdrant, swap BGE for Cohere rerank, swap Mistral for Claude) and the others don't notice.

That's the design goal: **modular phases, idiomatic primitives, contracts that don't lie.**
