# CodeLens_AI — Technical Deep Dive
**Audience:** Senior backend / RAG engineers joining the project.
**Scope:** Every critical code path across the 5-phase pipeline, line-by-line.
**Source of truth:** `backend/app/` as of 30 April 2026.

---

## Table of Contents

1. [High-Level Architecture & Data Flow](#1-high-level-architecture--data-flow)
2. [Phase-by-Phase File Breakdown](#2-phase-by-phase-file-breakdown)
   - [Phase 1 — Ingestion](#phase-1--ingestion)
   - [Phase 2 — Retrieval](#phase-2--retrieval)
   - [Phase 3 — Agentic Brain](#phase-3--agentic-brain)
   - [Phase 4 — API & Cache](#phase-4--api--cache)
   - [Phase 5 — Observability](#phase-5--observability)
3. [Infrastructure & Security](#3-infrastructure--security)
4. [Execution Logic — Request Lifecycle](#4-execution-logic--request-lifecycle)

---

## 1. High-Level Architecture & Data Flow

CodeLens_AI is a 5-phase RAG (Retrieval-Augmented Generation) system that answers developer questions over **two sources simultaneously**: Knowledge-Transfer (KT) PDFs/Markdown and source code repositories. Each phase has a single responsibility:

| Phase | Module(s) | Responsibility |
|---|---|---|
| **1. Ingestion** | `services/ingestion/*` | Load PDF/MD/code → split with language awareness → register function-level parents → embed children → persist in ChromaDB. |
| **2. Retrieval** | `services/retrieval/*` | Query expansion → hybrid (vector + BM25) → adaptive weight tuning → BGE rerank → parent context attach. |
| **3. Agentic Brain** | `services/agents/*` | Intent detection → routing → few-shot example selection → prompt assembly → LLM invocation. |
| **4. API & Cache** | `api/chat.py` | SSE streaming endpoint, semantic pgvector cache (multi-tenant), structured-output parsing. |
| **5. Observability** | `observability/rag_evaluator.py` | Async RAGAS scoring (faithfulness / context_recall / answer_relevancy) into SQLite. |

### Cross-cutting infrastructure

Two **process-wide singletons** in `app/core/database.py` are the load-bearing primitives that make the system performant:

```
                 ┌──────────────────────────────────────┐
                 │         app.core.database            │
                 │                                      │
                 │  ┌─ psycopg_pool.ConnectionPool ─┐  │
                 │  │   min=2, max=10              │  │
                 │  │   yields conn via            │  │
                 │  │   pg_connection() ctxmgr     │  │
                 │  └──────────────────────────────┘  │
                 │                                      │
                 │  ┌─ HuggingFaceEmbeddings ─────┐    │
                 │  │   all-mpnet-base-v2 (768d) │    │
                 │  │   loaded once per process   │    │
                 │  │   exposed via get_embedder()│    │
                 │  └─────────────────────────────┘    │
                 └──────────────────────────────────────┘
                          ▲              ▲
       ┌──────────────────┼──────────────┼──────────────────┐
       │                  │              │                  │
SemanticCache       ChatMemoryManager  RAGEvaluator      ExampleSelector
(pgvector)          (PostgresChat       (Postgres-      (cosine ranking
                     MessageHistory)    bound paths)    of few-shot ex.)
```

**Why singletons?** Two reasons:
1. The HuggingFace model load is ~500 MB and ~3-5 s warmup. Re-loading per request would add hundreds of milliseconds of latency.
2. Postgres TCP+auth handshake is ~10-30 ms. Without a pool, the `<20 ms` semantic-cache target is unreachable. Pooling amortizes the handshake.

### Request data flow (one diagram, end-to-end)

```
Angular EventSource ──► POST /api/v1/chat/stream
                          │ {query, session_id, user_id, stream:true}
                          ▼
              SemanticCache.get(query, user_id)            ◄── pgvector + WHERE user_id=...
              ├─ HIT  → stream cached words → SSE done
              └─ MISS
                          ▼
              AgentBrain.process_query_streaming
                          │
                _run_core_pipeline(request)
                  ├─ history       ← ChatMemoryManager.get_history (pooled conn)
                  ├─ intent        ← retriever.retrieve_with_intent
                  ├─ routing       ← AgenticRouter.route(intent, history)
                  ├─ where filter  ← routing_decision_to_metadata_filter(routing)
                  ├─ retrieval     ← RetrieverEngine.retrieve(
                  │                       use_dynamic_weights=True,
                  │                       metadata_filter=where,
                  │                   ) → RetrievalResult.chunks
                  ├─ few_shot      ← SemanticExampleSelector (singleton embedder)
                  └─ prompt        ← FewShotPromptBuilder + PydanticOutputParser fmt
                          ▼
              llm_client.stream(prompt) ──► SSE tokens to client
                          ▼
              ChatMemoryManager.add_message(user, assistant)   (pooled conn)
              SemanticCache.set(query, full_response, user_id) (pooled conn)
                          ▼
              BackgroundTasks ──► RAGEvaluator.evaluate_sample (Ollama judge)
                                   └── stored in evaluation_results.db
                          ▼
              SSE: data: {"type":"done","metadata":{...}}\n\n
```

---

## 2. Phase-by-Phase File Breakdown

### Phase 1 — Ingestion

#### 2.1 `services/ingestion/multi_modal_loader.py`

**Purpose.** Convert raw filesystem inputs (a code directory + a documentation directory) into uniformly-tagged LangChain `Document` objects. The crucial output contract is that every document carries `metadata["file_type"] ∈ {"code", "kt_doc"}` and `metadata["language"]` — these tags drive Phase-2 routing filters.

**Imports.**

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
```

`DirectoryLoader` is LangChain's glob-based loader. `PyPDFLoader` is selected explicitly via `loader_cls=` so PDFs are parsed page-by-page (each page becomes one `Document`). `Language` is the enum used downstream to pick a syntax-aware splitter.

**Class initialization.**

```python
CODE_EXTENSIONS = {
    ".js": Language.JS, ".ts": Language.TS, ".tsx": Language.TS,
    ".py": Language.PYTHON, ".java": Language.JAVA, ".cpp": Language.CPP,
    ".c": Language.C, ".go": Language.GO, ".rs": Language.RUST,
    ".cs": Language.CSHARP, ".php": Language.PHP,
}
KT_DOC_EXTENSIONS = {".pdf", ".md", ".txt"}
```

This static map is the **routing key for the entire pipeline**. Adding a new language requires extending this dict and the corresponding mapping in `language_aware_splitter.py`.

**Core method — `load_source_code()`.**

```python
loader = DirectoryLoader(
    path=str(directory_path),
    glob=f"**/{pattern}" if recursive else pattern,
    silent_errors=True,                # don't crash on a single unreadable file
    show_progress=True,
)
docs = loader.load()
for doc in docs:
    file_path = doc.metadata.get("source", "")
    ext = Path(file_path).suffix
    doc.metadata["file_type"] = "code"                        # ← Phase-2 filter key
    doc.metadata["language"] = self.CODE_EXTENSIONS.get(ext, "unknown")
    doc.metadata["line_count"] = len(doc.page_content.split("\n"))
```

`silent_errors=True` is intentional — one corrupted file should not halt ingestion of a 10,000-file repo. Errors accumulate in `self.document_metadata` for later inspection. The metadata enrichment that follows is **the contract** Phase 2 depends on.

**Core method — `load_kt_documents()`.**

```python
pdf_loader = DirectoryLoader(
    path=str(directory_path),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,            # one Document per PDF page
    silent_errors=True,
)
pdf_docs = pdf_loader.load()
for doc in pdf_docs:
    doc.metadata["file_type"] = "kt_doc"
    doc.metadata["language"] = "markdown"  # downstream splitter treats PDF text as MD
```

PDFs are forced into the `markdown` language bucket so the splitter uses prose-aware separators (`\n\n`, `\n`, ` `) rather than code-aware ones (`def `, `class `).

**Why the multi-modal approach matters.** Phase 2's `routing_decision_to_metadata_filter()` returns `{"file_type": "code"}` for `CODEBASE_ONLY` queries — but that filter only works if the ingestion contract is satisfied for every chunk. A missing or misspelled tag silently degrades retrieval.

---

#### 2.2 `services/ingestion/parent_document_retriever.py`

**Purpose.** Implement the **Parent Document Retrieval (PDR)** strategy: store small, semantically-tight child chunks (~400 tokens) in the vector index, but at retrieval time return the *enclosing function or class* as context to the LLM. This dramatically improves the signal-to-noise ratio.

**Imports.**

```python
import ast       # Python parent extraction
import re        # JS/TS regex boundary detection
import uuid      # parent_id namespacing
```

The choice of `ast` (Python's standard library) over `tree-sitter` is deliberate: zero extra dependency, and it parses Python identical to the interpreter. JS/TS uses regex with brace-balancing because shipping `@babel/parser` would require Node.js inside the Python container.

**Function-level parent extractors.**

```python
def _extract_python_parents(source: str) -> List[Tuple[str, str, int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # malformed file → fall back to file-scoped parents
    parents = []
    lines = source.splitlines()
    for node in tree.body:                         # only top-level definitions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = (getattr(node, "lineno", 1) or 1) - 1
            end = (getattr(node, "end_lineno", start + 1) or (start + 1))
            body = "\n".join(lines[start:end])
            parents.append((node.name, body, start + 1, end))
    return parents
```

Why **`tree.body`** and not a recursive walk? We want **top-level** parents only. Methods inside a class become part of the class's `body` text — the LLM gets the entire class, which is the correct unit of context for a question like "How does `AgentBrain.process_query` work?".

**JS/TS extractor.**

```python
_JS_BOUNDARY_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:function\s+(?P<fn>\w+)|class\s+(?P<cls>\w+)|"
    r"const\s+(?P<arrow>\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
    re.MULTILINE,
)
```

This matches three idioms: declared functions (`function foo(...)`), classes (`class Bar`), and arrow constants (`const baz = (...) =>`). The brace balancer that follows handles bodies:

```python
brace_idx = source.find("{", decl_start)
depth = 0
for i in range(brace_idx, len(source)):
    ch = source[i]
    if ch == "{": depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            end_idx = i
            break
```

It is **not** a real parser — it ignores `{` inside strings/comments. In practice this is acceptable because (a) we only need approximate function spans for chunk grouping, not compilation, and (b) the file-scoped fallback catches anything that misbehaves.

**Core method — `create_child_parent_pairs()`.**

```python
by_source: Dict[str, List[Dict]] = {}
for chunk in chunks:
    src = (chunk.get("metadata") or {}).get("source", "unknown")
    by_source.setdefault(src, []).append(chunk)
```

Step 1: group chunks back by their origin file. The splitter has already broken the file apart, but parent extraction needs the full file body.

```python
full_text = "\n".join(c.get("content", "") for c in file_chunks)
parents = extract_function_level_parents(full_text, language)

for name, body, start, end in parents:
    parent_id = f"parent::{source}::{name}::{start}-{end}"
    self.parent_store.add_parent(
        parent_id=parent_id,
        content=body,
        metadata={"source": source, "language": language,
                  "parent_name": name, "start_line": start,
                  "end_line": end, "scope": "function_or_class"},
    )
```

Each function/class becomes its own parent with a **stable, deterministic ID**. Re-running ingestion produces identical IDs (assuming source unchanged), which means incremental indexing in `context_aware_pipeline.py` Just Works.

```python
file_parent_id = f"parent::{source}::__module__"
self.parent_store.add_parent(parent_id=file_parent_id, content=full_text, ...)
```

The **file-level fallback** parent exists so that module-top-level statements (imports, module docstring, top-level constants) still have a parent to attach to.

```python
probe = content.strip()[:80]
if probe:
    for pid, body, _s, _e in parent_intervals:
        if probe and probe in body:
            enclosing_parent_id = pid
            break
```

For each child chunk, we use the first 80 non-whitespace characters as a fingerprint to find the enclosing parent by substring containment. This avoids re-running the line-number arithmetic — chunks are already opaque text, so substring is the safest match.

**Why this is better than file-scoped parents.** Before this refactor, the entire file was returned as "context". For a 1500-line file, that blew the LLM's context window and diluted relevance. Function-scoped parents return ~50-200 lines — the unit a developer would actually read.

---

### Phase 2 — Retrieval

#### 2.3 `services/retrieval/retriever_engine.py`

**Purpose.** Take a user query and return the top-5 most relevant chunks with full parent context. Three sub-engines collaborate: `QueryExpander`, `HybridRetriever`, `RerankingEngine`.

**The `_ChromaCollectionRetriever` adapter.**

```python
class _ChromaCollectionRetriever(BaseRetriever):
    collection: Any
    embeddings: Any
    k: int = 20
    metadata_filter: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(self, query, *, run_manager):
        query_embedding = self.embeddings.embed_query(query)
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": self.k,
            "include": ["documents", "metadatas", "distances"],
        }
        if self.metadata_filter:
            query_kwargs["where"] = self.metadata_filter   # ← Chroma filter applied here
        results = self.collection.query(**query_kwargs)
```

This wrapper exists because `EnsembleRetriever` requires both branches to be `BaseRetriever` instances. Wrapping the raw `chromadb.Collection` ourselves lets us **inject a `where=` filter dynamically** — LangChain's built-in `Chroma` retriever caches the filter at construction time, which is too rigid for a router-driven system.

**`HybridRetriever.__init__` — Ensemble assembly.**

```python
self.vector_retriever = _ChromaCollectionRetriever(
    collection=chroma_collection,
    embeddings=self.embeddings,
    k=candidate_k,
)

self.bm25_retriever = BM25Retriever.from_documents(documents_for_bm25)
self.bm25_retriever.k = candidate_k

self.ensemble = EnsembleRetriever(
    retrievers=[self.vector_retriever, self.bm25_retriever],
    weights=[vector_weight, bm25_weight],   # default 0.6 / 0.4
)
```

`EnsembleRetriever` performs **weighted Reciprocal Rank Fusion (RRF)**: each retriever produces a ranked list, the score for a doc is `Σ weight_i / (k + rank_i)`, and the fused list is sorted by total score. This is the *canonical* hybrid approach — we deliberately reuse LangChain's implementation rather than rolling our own.

**`HybridRetriever._retrieve_impl` — Dynamic weight tuning.**

```python
if self.enable_dynamic_weights and self.bm25_retriever is not None:
    intent_analysis = self.intent_detector.detect_intent(query, verbose=False)
    v_w, b_w = intent_analysis.vector_weight, intent_analysis.bm25_weight
    if abs(v_w - self.vector_weight) > 0.05 or abs(b_w - self.bm25_weight) > 0.05:
        ensemble = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[v_w, b_w],
        )
```

The `0.05` deadband prevents pointless rebuilds for trivial weight drift. The `IntentDetector` returns higher BM25 weights for exact-match queries (e.g. `def authenticate`) and higher vector weights for conceptual queries (`how does authentication work`). Rebuilding the `EnsembleRetriever` is cheap (no model load, just a wrapper).

**Routing-driven metadata filter.**

```python
previous_filter = getattr(self.vector_retriever, "metadata_filter", None)
if metadata_filter:
    self.vector_retriever.metadata_filter = metadata_filter

# ... ensemble.invoke(query) runs here ...

# Post-filter BM25 hits — BM25Retriever has no native metadata filter
if metadata_filter:
    def _matches(meta):
        for k, v in metadata_filter.items():
            if meta.get(k) != v: return False
        return True
    docs = [d for d in docs if _matches(d.metadata or {})]

# Restore prior filter so the retriever stays stateless across calls
self.vector_retriever.metadata_filter = previous_filter
```

This is one of the trickiest patches in the codebase. Because `EnsembleRetriever` wraps both retrievers, we cannot pass `where=` through it. Instead:

1. **Mutate** the vector retriever's filter for this call only.
2. Run the ensemble.
3. **Post-filter** BM25 results (BM25 has no native metadata-filter API).
4. **Restore** the original filter — critical for thread safety; without it, the next call could leak the wrong filter.

The mutation/restore is safe because Python dict assignment is atomic at the GIL level, and concurrent FastAPI handlers each hit different `RetrieverEngine` instance trees (per pipeline factory).

**`RerankingEngine.rerank` — BGE Cross-Encoder.**

```python
class RerankingEngine:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.cross_encoder = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc.get("content", "")] for doc in documents]
        scores = self.cross_encoder.predict(
            pairs, convert_to_numpy=True, show_progress_bar=False
        )
        scored = list(zip(documents, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
```

**Why a cross-encoder beats bi-encoder retrieval alone.** A bi-encoder (embedding model) encodes query and document independently and compares vectors. It scales but loses fine-grained interaction. The cross-encoder feeds `(query, document)` jointly to BERT-like layers, producing a much more accurate relevance score — at the cost of running ~20 forward passes (one per candidate). This is exactly the "20→5" workflow: cheap recall first (hybrid), expensive precision second (rerank).

`max_length=512` truncates oversized chunks at tokenization time so we never exceed BGE-v2-m3's window. `show_progress_bar=False` suppresses tqdm noise in production logs.

**`RetrieverEngine.retrieve` — End-to-end orchestration.**

```python
previous_dynamic = getattr(self.hybrid_retriever, "enable_dynamic_weights", False)
self.hybrid_retriever.enable_dynamic_weights = bool(
    use_dynamic_weights and HAS_INTENT_DETECTOR and self.hybrid_retriever.bm25_retriever is not None
)
```

Step 1: honor the caller-provided dynamic-weights flag, but degrade gracefully (no intent detector? no BM25 corpus? → silently disable). The previous-state save is matched by a `finally:` block that restores it.

```python
expanded_queries = self.query_expander.expand_query(query, max_variations=3)
for expanded_q in expanded_queries:
    results = self.hybrid_retriever.retrieve(
        expanded_q, top_k=candidates_k, metadata_filter=metadata_filter,
    )
    candidates.extend(results)
```

Step 2: run the **same hybrid retrieval for each query variant**. The variants come from `QueryExpander` and capture keyword-only / concept / technical reformulations. Each variant typically returns a slightly different top-20, and the union increases recall.

```python
candidates = self.hybrid_retriever._deduplicate_results(candidates)
candidates.sort(key=lambda x: x["score"], reverse=True)
candidates = candidates[:candidates_k]
```

Step 3: dedup by content (keeping the highest-scoring duplicate) and trim back to `candidates_k=20` for reranking. Without dedup, three variants matching the same chunk would push other relevant content out of the rerank window.

```python
top_chunks, rerank_scores = self.reranking_engine.rerank(
    query=query, documents=candidates, top_k=top_k,    # query=ORIGINAL, not expanded
)
```

Step 4: **rerank against the original query** (not the variants). The variants helped recall; precision should be measured against the user's actual intent.

```python
for chunk in top_chunks:
    parent_id = chunk.get("metadata", {}).get("parent_id", "")
    parent_context = None
    if parent_id and parent_id in self.parent_store:
        parent_context = self.parent_store[parent_id]
        parent_contexts[chunk_id] = parent_context
    final_chunks.append({
        "chunk_id": chunk_id, "content": chunk.get("content", ""),
        "metadata": chunk.get("metadata", {}),
        "parent_id": parent_id, "parent_context": parent_context,
        "score": chunk.get("score", 0.0),
    })
```

Step 5: attach the function-level parent. `AgentBrain` later prefers `parent_context` over `content` when formatting the prompt — the LLM sees the whole function rather than just the matched fragment.

---

### Phase 3 — Agentic Brain

#### 2.4 `services/agents/agent_brain.py`

**Purpose.** Orchestrate the full request lifecycle: history → routing → retrieval → few-shot → prompt → LLM → memory. Single source of truth: `_run_core_pipeline`.

**`_run_core_pipeline` — The unified pipeline.**

```python
async def _run_core_pipeline(self, request, span=None) -> Dict[str, Any]:
    history = await self._get_conversation_history(request.session_id)
```

Step 1: pull conversation history from `PostgresChatMessageHistory`. Returns `None` if no prior messages — the router treats this as a stateless query.

```python
    intent_analysis = None
    try:
        if hasattr(self.retriever, "retrieve_with_intent"):
            _, intent_analysis = self.retriever.retrieve_with_intent(
                query=request.query, top_k=self.config.retrieve_k,
            )
    except Exception as e:
        logger.warning(f"Failed to get intent analysis from retriever: {e}")
```

Step 2: best-effort intent extraction. We **discard the retrieval results** here (`_,`) — this call only exists to populate `intent_analysis`, which is fed to the router. The actual retrieval happens later with the routing filter applied. This trades one extra retriever call for routing-aware results; given the cost is dominated by reranking (which only runs on the second call), the overhead is small.

```python
    routing_result = self.router.route(
        query=request.query,
        intent_analysis=intent_analysis,
        conversation_context=history,
    )
```

Step 3: router fuses intent + history into a `RoutingResult` containing a `RoutingDecision` enum, sources to query, recommended tools, and confidence.

```python
    sources, retrieval_metadata = await self._retrieve_context(
        query=request.query, routing_decision=routing_result,
    )
```

Step 4: retrieval, with the routing decision converted into a Chroma `where=` filter inside `_retrieve_context`.

```python
    examples = None
    if self.config.enable_few_shot and self.example_selector:
        examples = self.example_selector.select_examples(
            query=request.query, k=self.config.num_examples,
        )
```

Step 5: pick `num_examples=2` curated Q&A pairs by cosine similarity. The selector uses the **singleton embedder** injected at factory time, so the cosine path is real (not the TF-IDF fallback).

```python
    retrieved_context = self._format_context(sources)
    if not retrieval_metadata.get("retrieval_success"):
        retrieved_context += "\n\n[SYSTEM: Retrieval attempted with fallback. ...]"

    prompt = self.prompt_builder.build_prompt(
        user_query=request.query,
        retrieved_context=retrieved_context,
        num_examples=len(examples) if examples else 0,
    )
    return {"history": history, "routing_result": routing_result,
            "sources": sources, "retrieval_metadata": retrieval_metadata,
            "examples": examples, "retrieved_context": retrieved_context,
            "prompt": prompt}
```

Step 6: build the structured-output prompt. The builder injects `PydanticOutputParser.get_format_instructions()` so the LLM is told to emit JSON conforming to `AnswerSchema(answer, sources, confidence_score)`.

**Why the dict return.** Streaming and non-streaming both need every field. Returning a dict (rather than nine positional values) keeps the call sites readable and lets us add fields without breaking either caller.

**`_retrieve_context` — Routing enforcement.**

```python
metadata_filter = None
try:
    from app.services.agents.agentic_router import routing_decision_to_metadata_filter
    metadata_filter = routing_decision_to_metadata_filter(routing_decision)
    if metadata_filter:
        logger.info(f"Applying routing metadata_filter={metadata_filter} ...")
except Exception as e:
    logger.debug(f"Could not derive metadata filter from routing: {e}")
```

This is where routing finally **bites the data**. The translation table:
- `RoutingDecision.CODEBASE_ONLY` → `{"file_type": "code"}`
- `RoutingDecision.KT_ONLY` → `{"file_type": "kt_doc"}`
- everything else → `None` (unconstrained)

```python
retrieval = self.retriever.retrieve(
    query=query,
    top_k=self.config.retrieve_k,
    use_dynamic_weights=self.config.use_dynamic_weights,
    metadata_filter=metadata_filter,
)

if hasattr(retrieval, "chunks"):
    results = retrieval.chunks or []
    retrieval_metadata["retrieval_time_ms"] = getattr(retrieval, "total_time_ms", 0.0)
```

Two critical correctness fixes are visible here:
1. **`use_dynamic_weights` is a real kwarg** — previously this raised `TypeError`, was swallowed by the broad `except`, and the entire system silently fell back to memory-based retrieval. The signature now accepts it.
2. **`RetrievalResult.chunks` is unwrapped explicitly** — earlier code iterated the dataclass directly and called `.get("content")` on it, blowing up. The `hasattr` check also accepts legacy `List[Dict]` returns for backward compatibility.

```python
sources = [{
    "content": (r.get("parent_context") or r.get("content", "")),  # ← parent preferred
    "metadata": r.get("metadata", {}),
    "score": r.get("score", 0.0),
    "source": (r.get("metadata", {}) or {}).get("source", r.get("source", "unknown")),
    "chunk_id": r.get("chunk_id", ""),
    "parent_id": r.get("parent_id", ""),
} for r in results]
```

The `r.get("parent_context") or r.get("content", "")` is the **realization of PDR**: if a function-scoped parent exists, the LLM sees the whole function. Otherwise it falls back to the literal matched chunk.

**`process_query_streaming` — Streaming entry point.**

```python
ctx = await self._run_core_pipeline(request, span=None)
prompt = ctx["prompt"]

chunk_count = 0
full_response = ""
try:
    async for chunk in self._stream_response(prompt):
        full_response += chunk
        yield chunk
        chunk_count += 1
except Exception as stream_error:
    logger.error(f"Stream interrupted: {stream_error}")
    yield f"\n\n[Stream interrupted: {str(stream_error)}]"
    return
```

The streaming path is now ~30 lines because all the hard work is in `_run_core_pipeline`. The `try/except` around the stream itself catches mid-flight LLM errors (network drop, OOM) and yields a final user-readable error chunk instead of crashing the SSE response.

```python
if self.config.enable_memory and self.memory_manager and chunk_count > 0:
    await self.memory_manager.add_message(role="user", content=request.query, ...)
    if full_response:
        await self.memory_manager.add_message(role="assistant", content=full_response, ...)
```

Both halves of the conversation are persisted **after** the stream ends. We do this post-hoc so a client disconnect mid-stream doesn't leave a half-message in the database.

**`_mock_response` — Production safety.**

```python
def _mock_response(self, prompt):
    try:
        debug_enabled = bool(get_settings().debug)
    except Exception:
        debug_enabled = False
    if not debug_enabled:
        msg = "LLM client is not configured and DEBUG mode is disabled. ..."
        logger.error(msg)
        return msg, len(msg.split())
    from app.services.agents.mock_utils import get_mock_response
    return get_mock_response(prompt)
```

Production guard: in `DEBUG=False`, we refuse to ship a mock answer and return a clear error string instead. This prevents the "demo response in production" failure mode that bit us during the audit.

---

### Phase 4 — API & Cache

#### 2.5 `api/chat.py`

**Purpose.** Expose the `AgentBrain` over a public HTTP API with low-latency caching, multi-tenant safety, and SSE streaming.

**`SemanticCache` — pgvector + tenant scoping.**

```python
def _init_backend(self):
    from app.core.database import pg_connection, get_embed_dim
    dim = get_embed_dim()
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    query TEXT NOT NULL,
                    response JSONB NOT NULL,
                    embedding VECTOR({dim}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""CREATE INDEX IF NOT EXISTS semantic_cache_user_idx
                           ON semantic_cache (user_id);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
                           ON semantic_cache USING ivfflat (embedding vector_cosine_ops)
                           WITH (lists = 100);""")
```

Two indexes matter here:
- **B-tree on `user_id`** — accelerates the per-tenant prefilter.
- **IVFFlat on `embedding`** — approximate nearest-neighbor index. `lists=100` is the standard tradeoff: more lists = faster query, lower recall. For a cache (where we accept some misses) this is correct.

```python
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='semantic_cache' AND column_name='user_id') THEN
        ALTER TABLE semantic_cache ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
    END IF;
END$$;
```

Inline migration block: if a previous deployment ran the un-scoped schema, this adds the column without breaking startup. Existing rows get `user_id='anonymous'` and are essentially garbage-collected by the TTL.

**`SemanticCache.get` — the multi-tenant ANN search.**

```python
embedder = get_embedder()                                    # singleton, no reload
embedding = np.array(embedder.embed_query(query), dtype=np.float32)
with pg_connection(register_pgvector=True) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT query, response, 1 - (embedding <=> %s::vector) AS similarity
            FROM semantic_cache
            WHERE user_id = %s
              AND created_at > NOW() - (%s || ' seconds')::interval
            ORDER BY embedding <=> %s::vector
            LIMIT 1;
        """, (embedding, user_id, str(self.ttl_seconds), embedding))
        row = cur.fetchone()
```

Why the `WHERE user_id = %s` is **before** the `ORDER BY`: pgvector applies the WHERE clause first to select candidate rows, then runs the IVFFlat scan over only that subset. Cross-tenant data is unreachable by construction — the planner cannot return another user's row even if it would be a closer cosine match.

The cosine distance operator `<=>` returns `[0, 2]`; `1 - distance` gives a `[-1, 1]` similarity, and we threshold at `0.95` for a hit.

**SSE streaming endpoint.**

```python
@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    agent: AgentBrain = Depends(get_agent_brain),
) -> StreamingResponse:
    cache_result = semantic_cache.get(request.query, user_id=request.user_id, similarity_threshold=0.95)
    if cache_result:
        return _create_cache_stream_response(cache_result)
```

The cache check is the **first thing** the endpoint does. A hit short-circuits the entire AgentBrain pipeline. The `0.95` threshold is conservative — only near-paraphrases hit. Lowering it (e.g. to `0.85`) would increase hit rate but risk serving stale or incorrect answers.

```python
async def generate():
    full_response = ""
    try:
        async for token in agent.process_query_streaming(agent_request):
            full_response += token
            yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'
            await asyncio.sleep(0.01)

        semantic_cache.set(request.query, full_response, user_id=request.user_id)

        try:
            retrieval_for_eval = agent.retriever.retrieve(
                query=request.query, top_k=agent.config.retrieve_k,
                use_dynamic_weights=agent.config.use_dynamic_weights,
            )
            eval_sources = getattr(retrieval_for_eval, "chunks", None) or []
        except Exception as _eval_err:
            eval_sources = []

        _schedule_rag_evaluation(
            background_tasks=background_tasks,
            query=request.query, answer=full_response,
            sources=eval_sources, session_id=request.session_id,
            source_type="chat_stream",
        )

        metadata = {"session_id": ..., "cached": False, "tokens": len(full_response.split())}
        yield f'data: {json.dumps({"type": "done", "metadata": metadata})}\n\n'
```

Each yielded line is a complete SSE event: `data: <json>\n\n`. The 10 ms `asyncio.sleep` is a tiny breather to let the event loop service other connections — without it, a fast LLM stream can starve other handlers.

**Headers matter for SSE.**

```python
return StreamingResponse(
    generate(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",       # ← critical for nginx
    }
)
```

`X-Accel-Buffering: no` tells nginx (the most common reverse proxy) **not** to buffer the response. Without it, nginx accumulates the entire stream and delivers it as one chunk — the client sees nothing for several seconds, then a wall of text.

**`_schedule_rag_evaluation` — the BackgroundTasks hook.**

```python
def _run() -> None:
    try:
        from app.observability.rag_evaluator import RAGEvaluator, EvaluationSample
        contexts = [(s.get("content") or "") for s in (sources or []) if s.get("content")]
        if not contexts:
            logger.debug("Skipping RAGAS eval: no retrieved contexts")
            return
        evaluator = RAGEvaluator.get_instance()      # singleton; no Ollama reload
        sample = EvaluationSample(
            query=query, ground_truth=answer,        # self-consistency baseline
            retrieved_context=contexts, answer=answer,
            session_id=session_id, source=source_type,
        )
        result = evaluator.evaluate_sample(sample)
        if result is not None:
            evaluator.db.store_result(result)
    except Exception as e:
        logger.warning(f"Background RAGAS evaluation failed: {e}")

background_tasks.add_task(_run)
```

The hook is **fire-and-forget**: any exception inside is logged at WARNING and never propagates to the user. The user sees their answer; faithfulness scoring lands in `evaluation_results.db` ~5-30 s later. Using `ground_truth=answer` (self-consistency) is a known trick — without an oracle, RAGAS still measures whether the answer is grounded in the retrieved contexts.

---

### Phase 5 — Observability

#### 2.6 `observability/rag_evaluator.py`

**Purpose.** Score every served response for hallucination/recall/relevancy without slowing down user requests.

**Singleton pattern.**

```python
@classmethod
def get_instance(cls, db_path="evaluation_results.db",
                 ollama_model="mistral", ollama_base_url="http://localhost:11434"):
    if cls._instance is not None:
        return cls._instance
    if cls._instance_lock is None:
        import threading
        cls._instance_lock = threading.Lock()
    with cls._instance_lock:
        if cls._instance is None:
            cls._instance = cls(db_path=db_path, ...)
    return cls._instance
```

Double-checked locking. Without it, a burst of requests at startup would each construct an evaluator (and each load the Ollama LLM handle). The lock is acquired only on the slow path.

**`evaluate_sample` — RAGAS invocation.**

```python
dataset_dict = {
    "question": [sample.query],
    "answer": [sample.answer],
    "contexts": [sample.retrieved_context],
    "ground_truth": [sample.ground_truth],
}
dataset = Dataset.from_dict(dataset_dict)

result = evaluate(
    dataset,
    metrics=[faithfulness, context_recall, answer_relevancy],
    llm=self.evaluator_llm,                # local Ollama, no API cost
)
```

RAGAS expects a HuggingFace `Dataset`. The three metrics:
- **`faithfulness`** — splits the answer into atomic claims, asks the judge LLM whether each claim is supported by the retrieved contexts. Hallucination detector.
- **`context_recall`** — checks what fraction of the ground truth is present in the retrieved contexts. Retrieval quality.
- **`answer_relevancy`** — generates synthetic questions from the answer and measures their similarity to the original query. Off-topic detector.

```python
metrics = EvaluationMetrics(
    faithfulness=float(result["faithfulness"][0]) if "faithfulness" in result else 0.0,
    context_recall=float(result["context_recall"][0]) if "context_recall" in result else 0.0,
    answer_relevancy=float(result["answer_relevancy"][0]) if "answer_relevancy" in result else 0.0,
    ...
)
```

Each metric is in `[0, 1]`. The `aggregate_score` (computed in `__post_init__`) is the unweighted mean — a useful single-number health KPI for dashboards.

**Fallback heuristic.**

```python
def _fallback_evaluate(self, sample) -> EvaluationResult:
    faithfulness_score = self._text_overlap(answer_lower, context_text)
    context_recall_score = self._text_overlap(ground_truth_lower, context_text)
    answer_relevancy_score = self._text_overlap(query_lower, answer_lower)
```

If RAGAS or Ollama is unavailable (CI environment, broken Ollama install) we degrade to Jaccard text overlap. Not as accurate but never zero — the dashboard always has data.

---

## 3. Infrastructure & Security

### 3.1 `app/core/database.py` — Singleton primitives

**Connection pool.**

```python
_pool_lock = threading.Lock()
_pool = None

def get_pg_pool():
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        from psycopg_pool import ConnectionPool
        dsn = build_psycopg_dsn()
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=int(os.getenv("PG_POOL_MIN_SIZE", "2")),
            max_size=int(os.getenv("PG_POOL_MAX_SIZE", "10")),
            timeout=float(os.getenv("PG_POOL_TIMEOUT", "10")),
            kwargs={"autocommit": False},
        )
    return _pool
```

The lazy init pattern matters: importing `app.core.database` does NOT connect to Postgres. The pool is created the first time `pg_connection()` is used, which means tests can monkeypatch the DSN before the pool exists.

**Pool sizing intuition:**
- `min_size=2`: keep two warm connections so the cache GET on the first request after idle doesn't pay handshake cost.
- `max_size=10`: enough for ~10 concurrent SSE streams (plus their cache+memory side calls). Each stream uses pool connections briefly during memory writes, not for the duration of the stream.
- `timeout=10s`: how long a caller waits for a free connection before raising `PoolTimeout`. Surfacing this as a 503 to the client is preferable to silent queueing.

**`pg_connection` context manager.**

```python
@contextmanager
def pg_connection(register_pgvector: bool = False):
    pool = get_pg_pool()
    with pool.connection() as conn:
        if register_pgvector:
            from pgvector.psycopg import register_vector
            register_vector(conn)
        yield conn
```

Two guarantees from `psycopg_pool.ConnectionPool.connection()`:
1. The connection is checked out from the pool on `__enter__`.
2. The connection is returned to the pool on `__exit__`, **even on exception**.

`register_vector` adapts numpy arrays to pgvector's wire format. We register it lazily (per-checkout) because connections in the pool may be replaced/recycled, and the registration is per-connection state.

**Singleton embedder.**

```python
def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedder = HuggingFaceEmbeddings(model_name=_DEFAULT_EMBED_MODEL)
    return _embedder
```

Same double-checked locking. The model load is ~3-5 s on first call; every subsequent call is a dict lookup. Three components share this: `SemanticCache` (cache), `SemanticExampleSelector` (few-shot), and indirectly `IngestionService` if it ever needs ad-hoc embeddings.

### 3.2 `services/agents/langchain_memory_manager.py`

**Per-operation connection checkout.**

```python
@contextmanager
def _checkout(self) -> Iterator[Any]:
    from app.core.database import pg_connection
    with pg_connection() as conn:
        yield conn

def _history_with(self, conn, session_id):
    return PostgresChatMessageHistory(
        self.table_name, session_id, sync_connection=conn,
    )
```

The previous implementation cached one private `psycopg.Connection` and bound every `PostgresChatMessageHistory` to it forever. Two problems:
1. **Concurrency:** psycopg connections serialize statements per connection. Two simultaneous `add_message` calls would queue.
2. **Staleness:** if the connection ever closed (network blip, server restart), the cache returned a dead connection until the manager was rebuilt.

The new pattern constructs `PostgresChatMessageHistory` **per operation** with a freshly-borrowed connection. The history object exists only for the duration of the `with` block.

```python
async def add_message(self, session_id, user_id, role, content, metadata=None):
    msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    def _do_add() -> None:
        with self._checkout() as conn:
            history = self._history_with(conn, session_id)
            history.add_messages([msg])
    await asyncio.to_thread(_do_add)
```

`asyncio.to_thread` is required because `PostgresChatMessageHistory.add_messages` is sync (psycopg sync API). Without it, the whole event loop would block on every memory write.

```python
async def get_history(self, session_id, max_tokens=2000) -> Optional[str]:
    def _load_messages() -> list:
        with self._checkout() as conn:
            history = self._history_with(conn, session_id)
            return list(history.messages)
    messages = await asyncio.to_thread(_load_messages)
    # ... budget walk newest→oldest, prepending until char_budget exceeded ...
```

The budget walk reverses the messages, accumulates from the most-recent end, and stops when adding another message would exceed `max_tokens * 4` characters. This guarantees we always include the most recent context — critical for conversational coherence even when history is large.

---

## 4. Execution Logic — Request Lifecycle

The complete sequence when an Angular client `POST`s to `/api/v1/chat/stream`:

### Step 1 — FastAPI dispatch and dependency resolution

```
POST /api/v1/chat/stream
  ChatRequest validated by Pydantic
  Depends(get_db)              → SQLAlchemy session (legacy ORM access)
  Depends(get_agent_brain)     → Pipeline factory yields fully-wired AgentBrain
  BackgroundTasks injected for the post-response evaluation hook
```

The agent brain comes from `get_pipeline_factory_cached()`, which is `@lru_cache`-decorated — so the entire object graph (retriever + reranker + embedder + memory manager + LLM client) is built **once at first request** and reused thereafter.

### Step 2 — Semantic cache lookup

```python
cache_result = semantic_cache.get(request.query, user_id=request.user_id, similarity_threshold=0.95)
```

Behind the scenes:
- `get_embedder()` returns the singleton embedder (no reload).
- `embedder.embed_query(query)` → 768-dim float32 numpy array (~5-15 ms warm).
- `with pg_connection(register_pgvector=True)` — pool checkout (~0.1 ms warm).
- SQL: `WHERE user_id = %s AND created_at > NOW() - interval ORDER BY embedding <=> %s LIMIT 1`.
- If `1 - distance ≥ 0.95`, return the cached response.

**On HIT:** `_create_cache_stream_response` re-streams the cached words with simulated tokenization. The user sees identical UX to a fresh LLM call. The pipeline ends here — total latency well under 100 ms.

### Step 3 — `AgentBrain.process_query_streaming` (cache miss)

```python
ctx = await self._run_core_pipeline(request, span=None)
```

Inside `_run_core_pipeline`:

| Sub-step | Operation | Latency |
|---|---|---|
| 3a | `ChatMemoryManager.get_history(session_id)` — pool checkout, fetch & format messages | ~5-15 ms |
| 3b | `retriever.retrieve_with_intent(query)` — get intent only (results discarded) | ~50-100 ms |
| 3c | `AgenticRouter.route(query, intent, history)` — pure CPU classification | <1 ms |
| 3d | `routing_decision_to_metadata_filter(routing_result)` → `{"file_type": "code"}` or `None` | <0.1 ms |
| 3e | `RetrieverEngine.retrieve(query, use_dynamic_weights=True, metadata_filter=where)` | ~150-400 ms |
|     | ├─ QueryExpander → 3 variants | <5 ms |
|     | ├─ HybridRetriever(×3 variants) → 60 candidates total | ~100 ms |
|     | ├─ Dedup → ~20 unique candidates | <5 ms |
|     | ├─ BGE Cross-Encoder rerank (20 pairs) | ~50-200 ms |
|     | └─ Attach function-level parents from `parent_store` | <1 ms |
| 3f | `SemanticExampleSelector.select_examples(query, k=2)` — cosine over curated Q&A | ~10-20 ms |
| 3g | `FewShotPromptBuilder.build_prompt(...)` with PydanticOutputParser fmt instructions | <1 ms |

Total pre-LLM latency: ~200-500 ms warm.

### Step 4 — LLM streaming

```python
async for chunk in self._stream_response(prompt):
    full_response += chunk
    yield f'data: {json.dumps({"type": "token", "content": chunk})}\n\n'
    await asyncio.sleep(0.01)
```

Each token from `llm_client.stream(prompt)` is immediately framed as an SSE event and pushed to the client. The Angular `EventSource` handler appends each `event.data` JSON's `content` field to the visible message.

### Step 5 — Post-stream side effects

After `async for` completes:

```python
semantic_cache.set(request.query, full_response, user_id=request.user_id)
```

The full response is embedded and inserted with the user's `user_id`. Future requests from this user (or a peer asking a similar question) will hit. `SET` runs on the same pool, ~5-15 ms warm.

```python
retrieval_for_eval = agent.retriever.retrieve(
    query=request.query,
    top_k=agent.config.retrieve_k,
    use_dynamic_weights=agent.config.use_dynamic_weights,
)
eval_sources = getattr(retrieval_for_eval, "chunks", None) or []

_schedule_rag_evaluation(
    background_tasks=background_tasks,
    query=request.query, answer=full_response,
    sources=eval_sources, session_id=request.session_id,
    source_type="chat_stream",
)
```

We re-run retrieval to capture the exact contexts (streaming yields tokens only — `sources` aren't surfaced through the AsyncIterator). This second retrieval hits the warm BGE model and the recently-queried ChromaDB collection — typically ~50-100 ms — and never blocks the user because it runs **before** the final SSE `done` event but its result is only used by the background task.

```python
yield f'data: {json.dumps({"type": "done", "metadata": metadata})}\n\n'
```

The final SSE event signals completion and provides response metadata (`tokens`, `cached`, `timestamp`).

### Step 6 — Background evaluation

After FastAPI sends the response, the `BackgroundTasks` runner executes `_run()`:

```python
contexts = [s.get("content") for s in sources if s.get("content")]
evaluator = RAGEvaluator.get_instance()                 # singleton
sample = EvaluationSample(query=..., answer=..., retrieved_context=contexts, ...)
result = evaluator.evaluate_sample(sample)              # Ollama judge: ~5-30 s
evaluator.db.store_result(result)                       # SQLite insert
```

This runs asynchronously to the user's request. By the time the dashboard refreshes, `evaluation_results.db` has a row with `faithfulness`, `context_recall`, `answer_relevancy`, and the aggregate score.

### Step 7 — Memory persistence

Inside `process_query_streaming`, after the stream loop:

```python
await self.memory_manager.add_message(
    session_id=request.session_id, user_id=request.user_id,
    role="user", content=request.query,
)
if full_response:
    await self.memory_manager.add_message(
        session_id=request.session_id, user_id=request.user_id,
        role="assistant", content=full_response,
    )
```

Two pool checkouts, each ~5-15 ms. The next `chat/stream` request from the same `session_id` will see this conversation on the next `get_history` call.

### Connection lifecycle summary (one request, cache miss)

| Pool checkout | Purpose | Returned to pool when |
|---|---|---|
| 1 | `ChatMemoryManager.get_history` (Step 3a) | After `with` block exits |
| 2 | `SemanticCache.set` (Step 5) | After insert commits |
| 3 | `ChatMemoryManager.add_message` (user msg, Step 7) | After insert |
| 4 | `ChatMemoryManager.add_message` (assistant msg, Step 7) | After insert |

Four separate, short-lived checkouts. With `max_size=10`, the pool can comfortably handle ten concurrent streams without contention.

---

## Appendix — File Map

```
backend/app/
├── core/
│   ├── config.py                     # Pydantic Settings (singleton via @lru_cache)
│   └── database.py                   # psycopg_pool + HuggingFaceEmbeddings singletons
├── api/
│   └── chat.py                       # /chat/stream + /chat + SemanticCache
├── services/
│   ├── pipeline_factory.py           # Wires the entire object graph (singleton)
│   ├── ingestion/
│   │   ├── multi_modal_loader.py     # PDF/MD/code → tagged Documents
│   │   ├── language_aware_splitter.py
│   │   ├── parent_document_retriever.py  # AST-based function-level parents
│   │   └── context_aware_pipeline.py     # End-to-end ingest orchestration
│   ├── retrieval/
│   │   ├── retriever_engine.py       # QueryExpander + HybridRetriever + Reranker
│   │   ├── query_intent_detector.py  # Intent → adaptive weights
│   │   └── retrieval_config.py
│   └── agents/
│       ├── agent_brain.py            # _run_core_pipeline orchestrator
│       ├── agentic_router.py         # RoutingDecision + metadata-filter helper
│       ├── semantic_example_selector.py  # Cosine ranking over curated Q&A
│       ├── few_shot_prompt.py        # FewShotPromptBuilder + PydanticOutputParser
│       ├── langchain_memory_manager.py   # Pooled PostgresChatMessageHistory
│       └── mock_utils.py             # DEBUG-only mock responses
└── observability/
    ├── otel_config.py                # Jaeger + Prometheus setup
    ├── quality_metrics.py            # RAGAS gauges + publish_ragas_scores() sink (post-L4 audit)
    └── rag_evaluator.py              # RAGAS singleton + SQLite storage; calls publish_ragas_scores()
```

## Appendix — Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_DSN` / `DATABASE_URL` | derived from `POSTGRES_*` | psycopg DSN for the pool |
| `PG_POOL_MIN_SIZE` | `2` | warm connections |
| `PG_POOL_MAX_SIZE` | `10` | concurrent connection ceiling |
| `PG_POOL_TIMEOUT` | `10` | seconds before `PoolTimeout` |
| `EMBED_MODEL` | `sentence-transformers/all-mpnet-base-v2` | singleton embedder |
| `EMBED_DIM` | `768` | must match the model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence |
| `LLM_PROVIDER` | `ollama` | `ollama` \| `groq` |
| `OLLAMA_URL` | `http://localhost:11434` | local LLM endpoint |
| `OTEL_ENABLED` | `false` | turn on Jaeger/Prometheus |
| `DEBUG` | `false` | enables mock-response fallback |

---

*Document generated 30 April 2026. Update whenever `_run_core_pipeline`, `pg_connection`, or `RoutingDecision` semantics change.*
