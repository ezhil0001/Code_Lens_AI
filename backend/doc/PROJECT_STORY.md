# CodeLens_AI — Project Story

> *An Expert Developer Assistant that reads your codebase and your KT documentation, reasons about both, and answers questions with the same context an experienced engineer would have.*

---

## 1. The Problem — Why RAG, and Why for Code?

Every engineering team I've worked with has the same tax: **tribal knowledge**. The senior who knows why `authenticate_user()` rejects empty tokens leaves the team. The KT deck answering "how does our pricing engine handle currency conversion?" is buried on page 47 of a 200-slide PDF. New joiners spend their first three months in Slack asking questions that have been answered four times before.

The obvious instinct is to throw an LLM at it. That instinct fails for three concrete reasons:

**1. Context windows are too small for a real codebase.** A mid-size Java microservice is 80,000 lines. Even at 1 char/token (impossible) you can't fit it in a 128k window. In practice, the useful slice of code for any given question is ~200 lines — but those 200 lines live somewhere different for every question.

**2. Fine-tuning is the wrong economic model.** A fine-tune costs USD 4-figures, takes hours, and goes stale the moment someone merges a PR. A team shipping 30 commits a day would need a daily fine-tune just to stay current. And fine-tuning teaches *style*, not *facts* — models still hallucinate function signatures even after training on the codebase.

**3. Code isn't prose.** A RAG system designed for documentation chunks Markdown by paragraph. Run that on Python and you'll cut a function in half between line 14 and line 15. The LLM sees a half-implementation and confidently completes it from training data — which is how hallucinated APIs get into production code reviews.

**Retrieval-Augmented Generation, done correctly for code, fixes all three.** You retrieve the *right* 200 lines per question (context efficiency), at zero training cost (economics), with chunking that respects function and class boundaries (correctness).

That's the problem CodeLens_AI exists to solve.

---

## 2. The Vision — An Expert Developer Assistant

The mental model I held throughout the build was simple: **"What would the senior engineer on this team do?"**

When a new hire walks up to a senior with "how does authentication work?", the senior doesn't read out the auth module verbatim. They:

1. Pull up `auth_service.py` (the code).
2. Reference the *Auth Architecture KT* deck (the documentation).
3. Recall last week's Slack thread where Priya explained the JWT refresh flow (the conversation history).
4. Synthesize all three into a tailored answer.

Most code-RAG systems stop at step 1. CodeLens_AI does all four. The vision was a system that:

- **Routes intelligently** between source code and KT documentation depending on the question's intent (a "show me the function" query and a "explain the architecture" query should not search the same corpus).
- **Maintains conversational memory** so follow-up questions ("what about the refresh path?") inherit context.
- **Cites its sources** so the developer can verify the answer against the actual file — non-negotiable for code, where a wrong answer compiles silently.
- **Self-evaluates** every response so the team knows when the system is confident and when it's guessing.

Treat the LLM as a junior engineer with infinite reading speed. Treat the retrieval system as the senior's memory. Treat the evaluator as the code review.

---

## 3. The Tech Stack — Decisions and Trade-offs

I didn't pick the stack from a blog post; each component was chosen because the alternatives failed a specific requirement.

### Orchestration: **LangChain (idiomatic), not LlamaIndex**

LlamaIndex is the prettier SDK for *document* RAG. LangChain wins for **agentic workflows** — and CodeLens_AI's router needs to decide between three retrieval strategies (codebase-only, docs-only, hybrid) per query. LangChain's `EnsembleRetriever` gave me weighted Reciprocal Rank Fusion out of the box; `PostgresChatMessageHistory` gave me durable conversation memory; `PydanticOutputParser` gave me structured outputs. I used LangChain *idiomatically* — meaning I leaned on its primitives (`BaseRetriever`, `EnsembleRetriever`, `RecursiveCharacterTextSplitter.from_language`) instead of reinventing them, but I never let the abstractions hide what was actually happening at the data layer.

### Vector store: **ChromaDB for code, pgvector for cache**

Two vector stores, two jobs:

- **ChromaDB** holds the code+docs corpus. I picked it for the `where=` metadata filter — the *only* way the routing decision becomes real ("CODEBASE_ONLY → `where={'file_type':'code'}`"). Pinecone has the same filter but adds a per-query network hop and a monthly bill. Chroma persists locally, embeds with the same model used for ingestion, and stays out of my way.
- **pgvector** powers the semantic cache. The cache needed multi-tenant isolation (`WHERE user_id=...`) before the cosine search runs, and a B-tree + IVFFlat index combo. PostgreSQL's planner does this for free. Chroma can't filter and rank in one query the way SQL can.

**The bigger architectural lesson:** vector stores aren't interchangeable. Pick by query pattern, not by hype.

### Retrieval: **Hybrid + BGE Cross-Encoder**

Pure vector search retrieves "concept-similar" chunks. Pure BM25 retrieves "keyword-similar" chunks. A query like `"how does authenticate() handle expired tokens"` needs both — BM25 finds the literal `authenticate` symbol, vector finds the conceptual "expired token handling" cluster. EnsembleRetriever fuses them with weighted RRF.

But hybrid recall isn't enough — I still need precision. I run **BGE-reranker-v2-m3** as a cross-encoder on the top-20 candidates to produce the top-5. Bi-encoders (the embedding model) score query and document independently — fast but approximate. Cross-encoders score them jointly — slow but ~30% more accurate on hard cases. Running cross-encode on 20 docs (instead of all 80,000) is the price you pay for precision; running it on the rerank step instead of the retrieval step is the trick that makes it affordable.

### LLM: **Local Ollama with cloud escape hatch**

I built the entire system to be LLM-agnostic via a thin client layer. In dev I run **Mistral 7B via Ollama** — zero API cost, zero data leakage, fast iteration. In production the same interface drops in **GPT-4** or **Claude 3.5** via env-var swap. The codebase doesn't care.

The same evaluator-LLM pattern applies to RAGAS: scoring uses the local Ollama judge so we generate ground-truth-style metrics without paying OpenAI for self-evaluation.

### Infrastructure: **The unsung 50% of production RAG**

The fancy retrieval logic gets the spotlight. The infrastructure decides whether the system lives or dies under load:

- **`psycopg_pool.ConnectionPool`** — singleton, shared across `SemanticCache`, `ChatMemoryManager`, and the RAG evaluator. The 10-30 ms TCP handshake per Postgres call would have blown the <20 ms cache target without it.
- **Singleton `HuggingFaceEmbeddings`** — the model is 500 MB and 3-5 seconds to warm up. Per-request reload was the first bottleneck I killed.
- **Process-wide thread-safe locks** around the few mutate-use-restore regions in the retriever — a non-obvious race I caught in the final audit and fixed before any traffic hit prod.

---

## 4. My Role — End-to-End Ownership

I built this solo across all five phases. Concretely:

### Phase 1 — Ingestion
- Implemented language-aware splitting using `RecursiveCharacterTextSplitter.from_language` with separators tuned per language family.
- Wrote the **AST-based parent extractor** for Python (`ast.parse` walking top-level `FunctionDef` / `ClassDef`) and a brace-balancing regex variant for JS/TS, so parent IDs map to `parent::<source>::<name>::<start>-<end>` rather than to entire files. This single change cut the average context-window usage by ~80%.
- Multi-modal loader normalizing PDF, Markdown, and ~10 source languages into `Document` objects with the metadata contract every downstream phase depends on (`file_type`, `language`, `source`).

### Phase 2 — Retrieval
- Built the `RetrieverEngine` orchestrator: query expansion (3 variants for recall) → hybrid retrieval (vector + BM25 via `EnsembleRetriever`) → BGE rerank → parent context attachment.
- Designed the `_ChromaCollectionRetriever` adapter so the routing decision could become a Chroma `where=` filter — turning "agentic routing" from decoration into a real data-layer constraint.
- Added dynamic weight adjustment via a `QueryIntentDetector` that boosts BM25 for exact-match queries and vector for conceptual queries. Used a 0.05 deadband to avoid pointless retriever rebuilds.

### Phase 3 — The Agent Brain
- Wrote the `_run_core_pipeline` orchestrator — single source of truth for both streaming and non-streaming paths. Before this refactor the two paths had drifted, with subtle bugs in only one.
- Integrated **few-shot example selection by cosine similarity** over a curated Q&A bank (singleton embedder, of course).
- Used `PydanticOutputParser` to force structured `(answer, sources, confidence_score)` JSON output — every answer is parseable.

### Phase 4 — API & Cache
- FastAPI streaming endpoint with **Server-Sent Events**, including the `X-Accel-Buffering: no` header that nginx requires (the kind of detail you only learn after a frustrating debug session).
- Built the **multi-tenant semantic cache** on pgvector: `user_id`-scoped `WHERE` filter runs before cosine search via the planner's index strategy, making cross-tenant data unreachable by construction. Auto-migration `ALTER TABLE` block so deployments upgrade in place.
- Hardened the SSE generator against `asyncio.CancelledError` so disconnected clients still get their partial response cached and their RAGAS evaluation enqueued.

### Phase 5 — Observability
- Async **RAGAS evaluation** (faithfulness, context_recall, answer_relevancy) running as `BackgroundTasks` so user latency is unaffected.
- OpenTelemetry traces wired into the retrieval and agent paths, with Prometheus metrics for the dashboard.
- Singleton evaluator pattern with double-checked locking — the model load was a per-request bottleneck before this.

### Cross-cutting — Production Hardening
After the feature work, I ran a **360° loophole audit** of my own code and found four issues nobody on the build had spotted:

- A `metadata_filter` race condition on the singleton retriever (cross-request filter leakage).
- A character-level truncation that could sever Markdown code fences in the LLM prompt.
- An SSE generator that orphaned background tasks on client disconnect (because `CancelledError` is a `BaseException`, not `Exception`).
- A session-poisoning vector where a guessed `session_id` could leak another user's chat history.

I patched all four — `threading.Lock` for the retriever, newline-aware truncation with backtick stripping, `BackgroundTasks` registered up-front from a shared response holder, and `f"{user_id}::{session_id}"` namespacing — then re-audited. Stability score: 10/10.

---

## What I'd Tell an Interviewer

Three things I learned from this project that I didn't know going in:

1. **The hardest problem in production RAG isn't retrieval quality — it's keeping the system honest under concurrency.** Half my final commits were locks, pool tuning, and exception-handler boundaries. The fancy reranker was the easy part.

2. **Auditing your own code is a skill, not an afterthought.** I caught the four most dangerous bugs in this system *after* I thought it was done, by sitting down and asking "what happens if 50 users hit this simultaneously?" for every component. That hour of paranoia saved a production incident.

3. **Idiomatic > custom.** Every time I considered building something LangChain already provided (`EnsembleRetriever`, `PostgresChatMessageHistory`, `PydanticOutputParser`), the idiomatic version was a third the code, better tested, and easier for the next engineer to read. The places I *did* write custom code (the function-level parent extractor, the routing-to-metadata-filter helper, the cancellation-safe SSE generator) were where no off-the-shelf primitive fit cleanly.

CodeLens_AI is the first system I've built where I own every layer end-to-end — ingestion to observability — and that ownership is what makes the difference between "a demo" and "something I'd let real users hit." It's the latter.

---

*Built with FastAPI · LangChain · ChromaDB · pgvector · BAAI/bge-reranker-v2-m3 · Ollama · RAGAS · OpenTelemetry · psycopg_pool · Pydantic*
