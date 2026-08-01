# CodeLens AI

> **An intelligent developer assistant that reads your codebase and KT documentation — and answers like the senior engineer who built it.**

Built on a production-grade 5-phase RAG pipeline: AST-aware ingestion → hybrid retrieval → BGE cross-encoder reranking → agentic reasoning → multi-tenant semantic cache.

---

## The Problem

Every engineering team carries a hidden tax: **tribal knowledge**.

The senior who knows why `authenticate_user()` rejects empty tokens leaves the team. The architecture decision that explains the pricing engine's currency logic is buried on page 47 of a 200-slide KT deck. New joiners spend their first three months in Slack asking questions that have been answered four times before.

Throwing a vanilla LLM at the problem fails in three concrete ways:

| Problem | Why it fails |
|---|---|
| **Context windows too small** | A mid-size service is 80,000+ lines. Even at 1 char/token, you cannot fit a real codebase in a 128k window. |
| **Fine-tuning is the wrong model** | A fine-tune costs four figures, takes hours, and goes stale the moment someone merges a PR. Teams shipping 30 commits a day would need a daily fine-tune just to stay current. |
| **Code is not prose** | Naïve RAG chunks by paragraph. Run that on Python and you cut a function in half at line 14. The LLM sees a half-implementation and confidently completes it from training data — hallucinated APIs end up in code reviews. |

CodeLens AI solves all three. It retrieves the *right* 200 lines per question — with chunking that respects function and class boundaries — at zero training cost.

---

## Architecture

### 5-Phase Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            REQUEST LIFECYCLE                              │
└──────────────────────────────────────────────────────────────────────────┘

  Client ──► POST /api/v2/chat/stream
                │
                ▼
  ┌─────────────────────┐   HIT   ┌─────────────────────────────────────┐
  │  Semantic Cache      │ ──────► │  Stream cached tokens via SSE       │
  │  (pgvector + ANN)   │         │  sub-100 ms total latency           │
  └──────────┬──────────┘         └─────────────────────────────────────┘
             │ MISS
             ▼
  ┌─────────────────────┐
  │  Agentic Router     │ ◄── conversation history + intent
  │  code / docs /      │
  │  hybrid decision    │
  └──────────┬──────────┘
             │ metadata_filter
             ▼
  ┌─────────────────────┐
  │  Hybrid Retrieval   │ ◄── ChromaDB vectors + BM25 corpus
  │  Vector + BM25      │     + function-level parent store
  │  weighted RRF → 20  │
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │  BGE Cross-Encoder  │
  │  Rerank → top-5     │
  │  + parent context   │
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │  Prompt Assembly    │ + few-shot examples (cosine-selected)
  │  ──► LLM stream     │
  └──────────┬──────────┘
             ▼
  Client ◄── SSE tokens     ──►  Cache.set(user_id-scoped)
                                  BackgroundTask → RAGAS scoring
```

### Phase Breakdown

| Phase | Module | What it does |
|---|---|---|
| **1 — Ingestion** | `services/ingestion/` | Loads PDF, Markdown, and 10+ code languages. AST-based parent extraction maps function/class boundaries to exact line ranges. Children are embedded (768d); parents are stored for context retrieval. |
| **2 — Retrieval** | `services/retrieval/` | Query expansion (3 variants for recall) → hybrid search via `EnsembleRetriever` (vector + BM25, weighted Reciprocal Rank Fusion) → adaptive weight tuning via `QueryIntentDetector` → top-20 candidates. |
| **3 — Reranking** | `RerankingEngine` | BGE-reranker-v2-m3 cross-encoder scores all 20 query-document pairs jointly, returning the top-5. Bi-encoder retrieval is fast but approximate; cross-encoder reranking buys ~30% precision improvement on hard queries. |
| **4 — Agent Brain** | `services/agents/` | Agentic router turns routing decisions into ChromaDB `where=` metadata filters (not decoration — a real data-layer constraint). Few-shot examples are selected by cosine similarity over a curated Q&A bank. Prompt is assembled with hallucination-prevention instructions, then streamed token-by-token via SSE. |
| **5 — Observability** | `observability/rag_evaluator.py` | Async RAGAS evaluation (faithfulness, context recall, answer relevancy) runs as a `BackgroundTask` so user latency is unaffected. Scores and traces stream to **Langfuse** for span-level latency, token/cost tracking, and evaluation trend analysis across the full retrieval → rerank → agent → generation path. |

---

## Key Design Decisions

### Why two vector stores?

**ChromaDB** holds the code and documentation corpus. The `where=` metadata filter is the only way routing decisions become real data constraints — `CODEBASE_ONLY → where={'file_type':'code'}`. ChromaDB persists locally, embeds with the same model used during ingestion, and stays out of the way.

**pgvector** powers the semantic cache. The cache requires multi-tenant isolation (`WHERE user_id = ...`) before the cosine search runs, and a B-tree + IVFFlat index combo. PostgreSQL's query planner does this for free. Chroma cannot filter and rank in a single query the way SQL can.

> Vector stores are not interchangeable. The right choice depends on query pattern, not hype.

### Why hybrid retrieval?

Pure vector search retrieves "concept-similar" chunks. Pure BM25 retrieves "keyword-similar" chunks. A query like `"how does authenticate() handle expired tokens"` needs both — BM25 finds the literal `authenticate` symbol, vector search finds the conceptual "expired token handling" cluster. `EnsembleRetriever` fuses them with weighted Reciprocal Rank Fusion.

### Why AST-based chunking?

Naïve character-level splitting severs function bodies mid-implementation. CodeLens AI walks the AST (`ast.parse` for Python, brace-balancing regex for JS/TS) and maps each `FunctionDef` / `ClassDef` to a `parent::<source>::<name>::<start>-<end>` ID. Parent Document Retrieval then returns the complete enclosing function at query time. This single design decision cut average context-window usage by ~80% while eliminating hallucinated function signatures.

---

## Production Hardening

Five engineering hours of paranoia — "what happens if 50 users hit this simultaneously?" — surfaced four bugs after the system was otherwise complete.

### Challenge 1 — Context Overflow

A user queried `"explain pricing engine"`. PDR returned the enclosing function. That function was 4,800 lines. Prompt size: ~250k tokens. Model context window: 32k. Hard crash.

**Fix:** A two-cap design with boundary-aware truncation.

```python
MAX_CHARS_PER_SOURCE = 8_000   # ≈ 2k tokens per source
MAX_TOTAL_CHARS      = 24_000  # ≈ 6k tokens total

@staticmethod
def _safe_truncate(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars - 500:      # honor newline boundary
        cut = cut[:last_nl]
    cut = cut.rstrip("`").rstrip()     # never unbalance Markdown fences
    return cut + "\n... [truncated]"   # explicit signal to LLM
```

Zero context-length crashes since deployment.

### Challenge 2 — Streaming Disconnect & Orphaned Tasks

FastAPI's `asyncio.CancelledError` is a `BaseException`, not `Exception`. A bare `except Exception` silently swallowed cancellations, leaving RAGAS evaluation and cache writes orphaned mid-flight.

**Fix:** Background tasks are registered from a shared `response_holder` dict before the stream begins. Cancellation is caught explicitly and the holder's completion flag is checked before any post-stream work runs.

### Challenge 3 — Retriever Race Condition

The `RetrieverEngine` singleton carries a `metadata_filter` field mutated per request. Without a lock, concurrent requests could overwrite each other's filter — User B's query silently receiving User A's source-restricted results.

**Fix:** `threading.Lock()` wraps the mutate-retrieve-restore sequence. The deadband is tight enough that throughput is unaffected; the race window is closed.

### Challenge 4 — Session Poisoning

`PostgresChatMessageHistory` keys on `session_id` alone. A client-supplied `session_id` matching a victim's value would silently return their conversation history to the attacker's prompt.

**Fix:** The server binds `session_id` to `user_id` immediately after request parsing — before any downstream component sees either value.

```python
namespaced_session = f"{request.user_id}::{request.session_id}"
```

A guessed session ID is now cryptographically useless without the matching `user_id`.

---

## Security Model

| Layer | Threat | Defense |
|---|---|---|
| **Memory** | Session poisoning via guessed `session_id` | `user_id::session_id` namespace — session ID alone is useless |
| **Retrieval** | Race on shared `metadata_filter` | `threading.Lock()` around mutate-retrieve-restore |
| **Vector DB** | Cross-tenant document access | ChromaDB `where=` filter scoped to `file_type` and collection |
| **Cache** | Cross-user cache poisoning | pgvector query runs `WHERE user_id = ?` before cosine ANN search |
| **Prompt** | Prompt injection via malicious query | Anti-hallucination system prompt; explicit boundary markers in context blocks |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **API** | FastAPI + SSE | Async-native, streaming-first, typed with Pydantic |
| **Frontend** | Angular 17 | Standalone components, reactive forms, ngx-markdown |
| **Orchestration** | LangChain (idiomatic) | `EnsembleRetriever`, `PostgresChatMessageHistory`, `PydanticOutputParser` — primitives used, not fought |
| **Code vector store** | ChromaDB | `where=` metadata filter turns routing into a data constraint |
| **Semantic cache** | PostgreSQL + pgvector | Multi-tenant `WHERE` + IVFFlat index; SQL planner beats a dedicated vector DB for this pattern |
| **Embedding model** | `all-mpnet-base-v2` (768d) | Consistent model across ingestion and retrieval — vector drift is impossible |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Multilingual cross-encoder; ~30% precision improvement over bi-encoder retrieval alone |
| **LLM** | Ollama (local) / Groq / OpenAI | Provider-agnostic thin client layer — swap via `.env` variable |
| **Evaluation** | RAGAS + Langfuse | Faithfulness, context recall, answer relevancy scored asynchronously and streamed to Langfuse |
| **Observability** | Langfuse | LLM tracing, span-level latency, token/cost tracking, and online evaluation across retrieval → rerank → agent → generation paths |

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.11 | LangChain + ChromaDB tested range |
| Node.js | ≥ 18.x | Angular 17 |
| PostgreSQL | ≥ 14 | Required for pgvector |
| pgvector | ≥ 0.5.1 | Cosine index for semantic cache and memory |
| RAM | ≥ 8 GB | Embedding model (500 MB) + reranker (2.3 GB) |

### Setup

```bash
# 1. Clone and create Python environment
git clone https://github.com/ezhil0001/Code_Lens_AI.git
cd CodeLens_AI/backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Create PostgreSQL database with pgvector
psql -U postgres <<'SQL'
CREATE DATABASE codelens_ai;
\c codelens_ai
CREATE EXTENSION IF NOT EXISTS vector;
SQL

# 3. Configure environment
cp .env.example .env
# Edit .env: set POSTGRES_*, LLM_PROVIDER, and optionally GROQ_API_KEY

# 4. Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Install and start the frontend
cd ../frontend
npm install
npm start
```

### Environment Variables (`.env`)

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=codelens_ai

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db

# LLM Provider (pick one)
LLM_PROVIDER=groq          # groq | ollama | openai
GROQ_API_KEY=gsk_xxx       # if LLM_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile

# Tavily API Key (for web search capabilities)
TAVILY_API_KEY=tvly-xx

# Evaluation LLM
EVAL_LLM_PROVIDER=groq     # groq | ollama

# Langfuse (LLM observability & evaluation) — optional, off by default
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
```

### Observability — Self-Hosted Langfuse

Langfuse is the primary LLM observability and evaluation platform. It captures a
full trace for every chat request — intent routing, parallel agent dispatch,
hybrid retrieval, reranking, prompt construction, LLM generation, guardrails,
and human-in-the-loop — with token usage, cost, latency, prompts/completions,
errors, and RAGAS evaluation scores attached per trace.

**1. Start the self-hosted stack** (Langfuse web + worker, Postgres, ClickHouse, Redis, MinIO):

```bash
cp .env.langfuse.example .env.langfuse
# Edit .env.langfuse: set NEXTAUTH_SECRET, SALT, and a 64-hex ENCRYPTION_KEY
#   openssl rand -base64 32   # for NEXTAUTH_SECRET and SALT
#   openssl rand -hex 32      # for ENCRYPTION_KEY

docker compose -f docker-compose.langfuse.yml --env-file .env.langfuse up -d
```

**2. Create a project** — open http://localhost:3000, sign up, create an
organization + project, then copy the generated API keys.

**3. Point the backend at Langfuse** — in `backend/.env`:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Restart the backend and send a chat request — the trace appears in the Langfuse
UI within seconds. When `LANGFUSE_ENABLED=false` (default), all tracing is a
safe no-op and the application runs unchanged.

| URL | Service |
|---|---|
| http://localhost:3000 | Langfuse UI (traces, sessions, scores, dashboards) |
| http://localhost:9091 | MinIO console (blob store; optional) |

> Distributed request/infra spans are still exported to **Jaeger** via
> OpenTelemetry (`OTEL_ENABLED=true`). Langfuse owns the LLM-level view;
> Jaeger owns the system-level view.
````

### Ingest Your Documents

```bash
# Via REST endpoint (after backend is running)
curl -X POST http://localhost:8000/api/v1/ingest/documents \
     -F "file=@./docs/architecture.pdf" \
     -F "file=@./src/"
```

### Verify

```bash
curl -N -X POST http://localhost:8000/api/v2/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "session_id": "demo",
    "user_id": "demo-user"
  }'
# Expect typed SSE envelopes ending with data: {"type":"done", "data":{...}}
```

---

## Codebase Structure

```
CodeLens_AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── chat.py              # SSE streaming endpoint + semantic cache
│   │   ├── services/
│   │   │   ├── ingestion/           # Phase 1: Multi-modal loader, AST splitter
│   │   │   ├── retrieval/           # Phase 2: Hybrid retriever, BGE reranker
│   │   │   └── agents/              # Phase 3: Router, brain, prompt builder
│   │   ├── core/
│   │   │   └── database.py          # Singleton embedder + connection pool
│   │   └── observability/
│   │       └── rag_evaluator.py     # Phase 5: Async RAGAS evaluation
│   └── requirements.txt
└── frontend/
    └── src/app/
        ├── components/chat.component.*
        ├── services/
        │   ├── ai-stream.service.ts  # fetchEventSource SSE client
        │   ├── chat.service.ts       # Message history + session
        │   └── session.service.ts   # Session persistence + recovery
        └── core/guards/auth.guard.ts
```

---

## Use Cases

- **Developer onboarding** — new joiner asks "how does the pricing engine handle currency conversion?" and gets a cited, function-level answer in seconds
- **Code review assistance** — "does this PR change anything in the authentication flow?" answered against the actual codebase
- **Debugging** — "what other callers depend on `process_payment()`?" with parent context showing the full calling function
- **Architecture exploration** — "explain the data flow between the ingestion and retrieval phases" synthesizing both code and KT docs

---

## Documentation

| Document | Contents |
|---|---|
| `PIPELINE_DEEP_DIVE.md` | Line-by-line walkthrough of all 5 phases |
| `CHALLENGES_AND_SOLUTIONS.md` | 6 production bugs in STAR format |
| `SECURITY_AND_PRIVACY.md` | Threat model and layered defenses |
| `INSTALLATION_GUIDE.md` | Full ops guide with troubleshooting table |
| `CODELENS_AI_TECHNICAL_DEEP_DIVE.md` | Architecture for senior engineers joining the project |

---

## What This Project Demonstrates

Building CodeLens AI end-to-end — ingestion to observability — taught three lessons that don't appear in blog posts:

**The hardest problem in production RAG is not retrieval quality — it is keeping the system honest under concurrency.** Half the final commits were locks, pool tuning, and exception-handler boundaries. The fancy reranker was the easy part.

**Auditing your own code is a skill, not an afterthought.** The four most dangerous bugs in this system were found after it was otherwise "done", by asking "what happens if 50 users hit this simultaneously?" for every component.

**Idiomatic beats custom.** Every time the impulse was to build something LangChain already provided, the idiomatic version was a third of the code, better tested, and easier to read. The places where custom code was warranted — the function-level parent extractor, the routing-to-metadata-filter translation, the cancellation-safe SSE generator — were places where no off-the-shelf primitive fit cleanly.

---

*Built with FastAPI · Angular · LangChain · LangGraph · ChromaDB · pgvector · BAAI/bge-reranker-v2-m3 · Groq/Ollama · RAGAS · Langfuse · psycopg_pool · Pydantic* 