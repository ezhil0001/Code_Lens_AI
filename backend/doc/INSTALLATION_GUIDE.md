# 📦 CodeLens_AI — Installation & Operations Guide

A complete, copy-paste-friendly setup for engineers who have never seen
this codebase before. Reading this end-to-end takes ~10 minutes; setup
takes ~20 minutes on a clean machine.

---

## 1. Pre-requisites

| Component        | Required version | Why                                              |
|------------------|------------------|--------------------------------------------------|
| **Python**       | 3.10 — 3.11      | LangChain v1.x + ChromaDB tested on 3.10/3.11    |
| **Node.js**      | ≥ 18.x           | Angular 17 frontend                              |
| **PostgreSQL**   | ≥ 14             | Required for `pgvector` extension                |
| **pgvector**     | ≥ 0.5.1          | Cosine vector index (semantic cache, memory)     |
| **Ollama** *(optional)* | latest    | Local LLM (Llama 3.1) — alternative is Groq/OpenAI |
| **Disk**         | ≥ 4 GB           | HF model weights (`bge-reranker-v2-m3` ≈ 2.3 GB) |
| **RAM**          | ≥ 8 GB           | Embedding + reranker models                      |

### 1.1. Install PostgreSQL + pgvector

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew install pgvector
brew services start postgresql@16
```

**Ubuntu/Debian:**
```bash
sudo apt install -y postgresql-16 postgresql-16-pgvector
sudo systemctl start postgresql
```

**Verify pgvector is available:**
```bash
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" -c "\dx"
# Expect to see "vector" in the extensions list.
```

### 1.2. Required environment variables

Create `backend/.env` (or export in your shell):

```bash
# --- PostgreSQL --------------------------------------------------------
# EITHER use a single DSN:
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/codelens_ai
# OR use individual fields (DSN takes precedence if set):
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=codelens_ai

# --- ChromaDB ----------------------------------------------------------
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_DEFAULT_COLLECTION=codelens_ingestion

# --- LLM (pick ONE provider) -------------------------------------------
LLM_PROVIDER=ollama                  # ollama | groq | openai
OLLAMA_URL=http://localhost:11434
# GROQ_API_KEY=gsk_xxx
# OPENAI_API_KEY=sk_xxx

# --- Optional: HuggingFace -------------------------------------------
# HF_HOME=~/.cache/huggingface
# HUGGINGFACE_HUB_TOKEN=hf_xxx        # only needed for gated models

# --- Observability (optional) -----------------------------------------
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## 2. Installation Steps

### 2.1. Clone & create Python env

```bash
git clone <your-repo-url> CodeLens_AI
cd CodeLens_AI/backend

python3.11 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install --upgrade pip wheel
```

### 2.2. Install Python dependencies

```bash
pip install -r requirements.txt

# Critical extras the new pipeline relies on (some may already be in requirements.txt):
pip install \
    langchain langchain-classic langchain-community \
    langchain-chroma langchain-huggingface langchain-postgres langchain-core \
    chromadb sentence-transformers rank-bm25 \
    pgvector "psycopg[binary]" \
    fastapi uvicorn[standard] python-multipart pydantic \
    pypdf
```

### 2.3. Initialise the PostgreSQL database

```bash
# 1. Create the application database
psql -U postgres <<'SQL'
CREATE DATABASE codelens_ai;
\c codelens_ai
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

The application creates the rest of its tables on first run:

| Table                  | Created by                                       | Purpose                                  |
|------------------------|--------------------------------------------------|------------------------------------------|
| `chat_message_history` | `PostgresChatMessageHistory.create_tables()`     | LangChain conversation memory            |
| `semantic_cache`       | `SemanticCache._init_backend()` (chat.py)        | pgvector query→response cache (768-D)    |
| Application tables     | `prisma migrate` or `Base.metadata.create_all()` | Users, sessions, etc.                    |

If you want to materialise everything up-front, here is the canonical schema for the two LangChain-managed tables (executed automatically on startup but reproduced here for reference):

```sql
-- Semantic cache (pgvector)
CREATE TABLE IF NOT EXISTS semantic_cache (
    id          BIGSERIAL PRIMARY KEY,
    query       TEXT NOT NULL,
    response    JSONB NOT NULL,
    embedding   VECTOR(768) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
    ON semantic_cache USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Chat message history (auto-managed schema by langchain_postgres,
-- shape may evolve across versions — never hand-edit)
```

### 2.4. (Optional) Apply Prisma migrations

If your project uses Prisma for the user/session tables:

```bash
cd backend/prisma
prisma migrate deploy        # production
# or
prisma migrate dev           # local dev with seed
```

### 2.5. Frontend install

```bash
cd ../frontend
npm install
```

---

## 3. Execution

### 3.1. Start the LLM (Ollama path)

```bash
# In a dedicated terminal:
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve
```

(Skip if you set `LLM_PROVIDER=groq` or `openai`.)

### 3.2. Start the FastAPI backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup you should see:
```
🏗️  Initializing RAG Pipeline Factory...
  ├─ Initializing Phase 2: Retrieval Engine...
  │  ✅ Retriever initialized with dynamic weights enabled
  ├─ Initializing Phase 3: Agent Brain Components...
  │  ✅ Chat Memory Manager (LangChain Postgres) initialized
  ✅ Semantic cache (pgvector) ready — threshold=0.95, dim=768
  ✅ EnsembleRetriever ready (vector=0.6, bm25=0.4)
  ✅ Loaded BGE Reranker: BAAI/bge-reranker-v2-m3
✅ RAG Pipeline Factory initialized successfully
```

### 3.3. Run the ingestion pipeline

You can ingest a folder (PDFs + code) once at the start, or trigger via API later.

**Option A — One-shot CLI script:**
```bash
cd backend
python main_ingestion.py --source ./docs --collection codelens_ingestion
```

**Option B — REST endpoint (after backend is up):**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./docs/handbook.pdf"
```

What happens during ingestion (verified by logs):
1. `MultiModalLoader.load_kt_documents()` parses PDFs (`pypdf`) and code.
2. `LanguageAwareSplitter` produces parent + child chunks (1500 / 400 chars).
3. `HuggingFaceEmbeddings("sentence-transformers/all-mpnet-base-v2")` embeds children.
4. `chromadb.PersistentClient("./chroma_db")` writes 768-D vectors to a `documents_*` collection.
5. Parent contents are stored in chunk metadata for PDR aggregation later.

### 3.4. Start the frontend

```bash
cd frontend
npm start                    # Angular dev server on :4200
```

### 3.5. Smoke-test the chat endpoint

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work in this codebase?",
    "session_id": "demo-session",
    "user_id": "demo-user",
    "stream": true
  }'
```

You should see SSE chunks streaming, ending with a `data: {"type":"done", "metadata":{...}}` envelope.

---

## 4. Operational Cheatsheet

| Action                              | Command                                                          |
|-------------------------------------|------------------------------------------------------------------|
| View cache size                     | `curl localhost:8000/api/v1/chat/cache/status`                  |
| Clear semantic cache                | `curl -X POST localhost:8000/api/v1/chat/cache/clear`           |
| Reset Chroma store                  | `rm -rf backend/chroma_db && restart backend`                   |
| Tail backend logs                   | `tail -f backend/logs/app.log`                                  |
| Re-download reranker                | `rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3` |
| Validate pgvector wired             | `psql -d codelens_ai -c "\dt semantic_cache"`                   |

---

## 5. Troubleshooting

| Symptom                                                             | Cause / Fix                                                                                       |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: langchain_chroma`                             | Run `pip install langchain-chroma` *inside the same venv* you launch uvicorn from.                |
| `EnsembleRetriever import` failing                                  | Install `langchain-classic` (for langchain ≥ 1.0) — fallback path is automatic.                   |
| `extension "vector" is not available`                                | Install pgvector (`brew install pgvector` / `apt install postgresql-16-pgvector`) and retry.      |
| Cache always misses                                                 | Confirm `semantic_cache._available is True` in startup logs. Otherwise check Postgres connection. |
| First request takes 30 s+                                            | Cold-start of HF models. Pre-warm by hitting `/api/v1/health` after server boot.                  |
| Reranker logs `MiniLM`                                               | You are running an older build. Re-pull and confirm `RerankingEngine` defaults to `BAAI/bge-reranker-v2-m3`. |
| `psycopg.errors.UndefinedFile`                                      | pgvector binary not installed for your Postgres version.                                          |
| LLM emits prose around the JSON                                     | Already handled by `AgentBrain._parse_structured_output`; check `metadata.structured_output_ok`.  |

---

## 6. Where to look next

- **`TECHNICAL_DEEP_DIVE.md`** — every line of the core 5 phases explained.
- **`README.md`** — high-level project pitch and screenshots.
