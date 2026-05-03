# RAG Interview Prep — Q&A

> *Curated for AI Engineer roles (2 YOE). Each answer is interview-ready: 60-90 seconds spoken, with one technical depth-bomb in case the interviewer drills in.*

---

## Section 1 — Evaluation: RAGAS & Friends

### Q1.1 — "How do you actually measure if a RAG system is good?"

**Short answer.** You decompose "good" into three independently-measurable axes and use RAGAS:

| Metric | Question it answers | What it catches |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? | Hallucinations |
| **Context Recall** | Does the retrieved context contain everything needed to answer the ground truth? | Retrieval gaps |
| **Answer Relevancy** | Does the answer actually address the user's question? | Off-topic answers |

Together these isolate the failure mode: low faithfulness ⇒ generation problem (LLM hallucinating). Low context recall ⇒ retrieval problem (chunks missing). Low answer relevancy ⇒ prompt problem (LLM answering a different question).

**Depth-bomb.** RAGAS uses an LLM-as-judge under the hood. Faithfulness works by decomposing the answer into atomic claims and asking the judge whether each claim is entailed by the contexts. That means **your RAGAS score quality is upper-bounded by your judge model.** I run the judge on a local Ollama Mistral instance — good enough for relative trends; for absolute numbers in a regulated environment you'd want GPT-4 as judge or human eval.

---

### Q1.2 — "What's the problem with using the model's own answer as ground truth?"

It's called **self-consistency baseline** — and it's a *useful* signal but not a *reliable* one. If the LLM hallucinates confidently, faithfulness scores can be high while the answer is wrong. The fix is a curated test set of human-graded `(query, ground_truth_answer, expected_contexts)` triples. CodeLens_AI uses self-consistency in production (free, always-on) and a 50-query gold set in CI for regression detection.

---

### Q1.3 — "When does RAGAS give you misleading numbers?"

Four traps:

1. **Short answers fool faithfulness.** A one-sentence response has fewer claims to check; faithfulness defaults to 1.0. Always pair with answer length distribution.
2. **Verbose contexts inflate context recall.** If you stuff 32k tokens of context, recall trivially looks high. Pair with `context_precision` (how much of the context is actually used).
3. **Paraphrase blindness.** The judge LLM may miss that "user authentication" and "auth flow" are equivalent. Trends are reliable; absolute thresholds are not.
4. **Judge-LLM bias.** A judge trained by OpenAI scores OpenAI-style outputs higher. Rotate judges or use multiple.

---

## Section 2 — Fine-tuning vs. RAG

### Q2.1 — "When would you fine-tune instead of RAG?"

The decision is about **what kind of knowledge** you need the model to have:

| Use Fine-tuning when… | Use RAG when… |
|---|---|
| Behavior, format, tone is the goal (e.g. "always respond in JSON", "respond in legal tone") | Facts, documents, codebase content is the goal |
| Knowledge is **stable** and small (e.g. classification labels) | Knowledge is **dynamic** (changes with each commit/PR) |
| Latency sensitivity (no retrieval round-trip) | Auditability required (must cite sources) |
| Cost amortizes over millions of inferences | Cost matters at training time |

**The honest framing:** they solve different problems. Fine-tuning teaches *style*; RAG provides *facts*. A real product often needs both — fine-tune for output structure, RAG for the dynamic knowledge.

---

### Q2.2 — "Why didn't you fine-tune CodeLens_AI on the codebase?"

Three reasons, in order of weight:

1. **Staleness.** The team ships 30 commits a day. A fine-tune is hours and dollars. RAG is a re-index — cheap and incremental.
2. **Citations.** Developers need to verify the answer against the actual file. Fine-tuned models can't cite; RAG returns `(content, source, line_range)`.
3. **Hallucination tax.** Fine-tuning *teaches* the model patterns from your code but doesn't prevent it from inventing API signatures that look plausible. RAG anchors generation to retrieved evidence.

I'd revisit fine-tuning for **output formatting** (currently handled by `PydanticOutputParser`) if structured-output reliability became a bottleneck.

---

### Q2.3 — "What about LoRA / PEFT / instruction tuning?"

Parameter-efficient fine-tuning (LoRA, QLoRA) changes the economics — you can adapt a 7B model on consumer hardware in hours, not days. That makes "fine-tune for *style* + RAG for *facts*" an actually-viable pattern. The decision tree becomes:

- **Tone, format, refusal behavior** → LoRA on instruction-tuning data.
- **Knowledge** → still RAG, always RAG.

For CodeLens_AI today: not worth the operational complexity. For a product version with a custom voice/persona: yes.

---

## Section 3 — Vector DBs: Scaling & Indexing

### Q3.1 — "What are the practical scaling limits of a vector DB?"

Three independent dimensions:

1. **Cardinality** — how many vectors. ChromaDB starts to struggle past ~10M; Pinecone, Weaviate, Qdrant, Milvus handle 100M-1B with sharding.
2. **Dimensionality** — most embedding models are 384-1536 dim. Going to 4096-dim (e.g. some OpenAI models) inflates memory ~10×.
3. **Filter complexity** — vector search + metadata filter is the silent killer. Pre-filtering (filter then vector search) is fast on B-tree-indexed columns but slow on high-cardinality string fields.

**The number that matters in practice:** queries-per-second-per-GB. Once your index doesn't fit in RAM, latency cliff-drops. Plan for fitting in memory.

---

### Q3.2 — "HNSW vs IVFFlat — when do you pick which?"

| | HNSW (Hierarchical Navigable Small World) | IVFFlat (Inverted File with Flat) |
|---|---|---|
| **Idea** | Multi-layer graph — start coarse, descend through nearest neighbors | Cluster vectors into `n_lists`, search the closest `n_probes` clusters |
| **Build time** | Slow (graph construction) | Fast (k-means clustering) |
| **Query time** | Faster, more consistent latency | Slower, but predictable |
| **Recall** | Higher recall at same latency | Tunable via `n_probes` |
| **Memory** | High (graph + vectors) | Low (centroids + vector pointers) |
| **Updates** | Insertions are cheap; deletes degrade graph | Insertions are cheap; periodic re-cluster needed |
| **Best for** | Read-heavy, low-latency, in-memory | Memory-constrained, batch-update-friendly |

**Rule of thumb:** HNSW for serving, IVFFlat for caches and budget deployments.

CodeLens_AI uses **IVFFlat for the semantic cache** because: cache rows are small per tenant, latency is dominated by the LLM stream anyway, and IVFFlat supports `pgvector` natively. **For the main code corpus (ChromaDB), HNSW is the default and we leave it alone.**

---

### Q3.3 — "What's `n_lists` and how do you tune it?"

For IVFFlat: `n_lists ≈ sqrt(total_vectors)` is the rule of thumb. CodeLens_AI uses `lists = 100` for the cache (designed for ~10k rows per tenant — `sqrt(10000) = 100`).

`n_probes` is the query-time tunable: how many clusters to search. Higher = better recall, slower. Default 1; bump to `lists / 10` for high-recall use cases.

**Trap:** if `n_lists` is too large for your data (e.g. 1000 lists with 100 vectors total), each cluster is tiny and HNSW's brute-force fallback kicks in — you lose the index entirely. Always revisit `lists` after data growth.

---

### Q3.4 — "How do you add metadata filtering without killing performance?"

Three approaches, increasing in sophistication:

1. **Pre-filter** — apply metadata filter first, then vector search the result set. Fast on selective B-tree filters (`user_id = X`). Slow on high-cardinality string filters.
2. **Post-filter** — vector search first, drop non-matching results. Risky: if the filter is selective, you may need k=1000 to get 5 matches.
3. **Hybrid filtered HNSW** — modern indexes (Milvus, Qdrant, Weaviate) build filtered subgraphs. Best-of-both-worlds.

CodeLens_AI uses **pre-filter on `user_id` + `file_type`** because both are low-cardinality and both have B-tree indexes. PostgreSQL's planner picks the cheap filter first by design — verified via `EXPLAIN ANALYZE`.

---

## Section 4 — Long Context & "Lost in the Middle"

### Q4.1 — "What is the 'Lost in the Middle' phenomenon?"

Stanford's 2023 paper showed that LLMs systematically **underweight information placed in the middle of a long context**. Performance is U-shaped: high recall on content at the beginning, high recall on content at the end, sharp dip for everything in between. Even Claude 100k and GPT-4 32k exhibit this.

**Why it matters for RAG:** if you naively dump 20 retrieved chunks in retrieved-rank order, the most relevant chunks (rank 1-3) end up at the start, ranks 18-20 at the end, and everything in between is *less likely to be used* even when relevant.

---

### Q4.2 — "How do you mitigate Lost-in-the-Middle?"

Four techniques, ranked by effectiveness:

1. **Reranker + small top-K** — the cheapest fix. Use a cross-encoder to get to top-5, not top-20. Five chunks all fit in the "edge attention" zones of any modern model. **CodeLens_AI uses this.**
2. **Reorder by relevance, then by U-shape** — place top-1 at the start, top-2 at the end, top-3 second-from-start, etc. Manual recipe, ~5% recall lift in published benchmarks.
3. **Summarize middle chunks** — a separate LLM call summarizes ranks 5-15, packs them dense in the middle. Adds latency.
4. **Iterative retrieval** — multiple rounds of "what else would you need to answer this?" Best quality, worst latency. Worth it for long-form research, not for chat.

**The pragmatic answer:** smaller top-K is usually better than fancier reordering. CodeLens_AI's BGE rerank to top-5 + 24k char total cap means *every chunk lives in the model's strong attention zone*.

---

### Q4.3 — "When would you actually use a 1M-token context model?"

Rarely for RAG. The honest use cases are:

- **Long-form document QA** where chunking destroys argument structure (legal contracts, scientific papers).
- **Multi-document synthesis** where cross-document references can't be retrieved chunk-wise.
- **Few-shot with hundreds of examples** when the task is too niche for embedding-based selection.

For typical RAG workloads (chat over a knowledge base), **a 32k model with good retrieval beats a 1M model with bad retrieval**. The 1M context tax — latency, cost, lost-in-the-middle — usually outweighs the benefit of being lazy about retrieval.

---

## Section 5 — Optimization: Chunking & Embeddings

### Q5.1 — "How do you pick the chunk size?"

There's no universal answer; the right size depends on **the unit of meaning** in your domain:

| Domain | Natural unit | Chunk size |
|---|---|---|
| Source code | Function / class | 200-1500 tokens (variable) |
| Legal contracts | Section / clause | 500-1000 tokens |
| Conversational logs | Message / turn | 50-300 tokens |
| Wikipedia-style prose | Paragraph | 300-500 tokens |
| Scientific papers | Subsection | 800-1500 tokens |

**The empirical recipe:**

1. Start at **400 tokens** with **80-token overlap**.
2. Run RAGAS context_recall on a 50-query test set.
3. Halve and double, take whichever direction lifts recall.
4. Stop when recall plateaus.

**Critical insight:** chunk size is coupled to embedding model context. Use 768-token chunks with a 512-token embedding model and you're truncating silently. CodeLens_AI uses `all-mpnet-base-v2` (514 tokens max) with chunks ≤ 400 tokens.

---

### Q5.2 — "Why language-aware splitting for code?"

Default char splitter cuts at the 400th character regardless of structure. On code, that means severing function bodies, breaking string literals, splitting class definitions. The LLM then sees `def authenti` and `cate_user(token):` as adjacent unrelated tokens and confidently invents a function called `cate_user`.

LangChain's `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` ships separators tuned per language: `["\nclass ", "\ndef ", "\n\tdef ", ...]`. Splits prefer structural boundaries.

**The full solution in CodeLens_AI:** language-aware splitting *plus* AST-based parent extraction. Children are 400 tokens for retrieval precision; parents are entire functions for LLM context. This is **Parent Document Retrieval (PDR)** — best of both worlds.

---

### Q5.3 — "How do you pick an embedding model?"

The MTEB (Massive Text Embedding Benchmark) leaderboard is the starting point, but not the endpoint. The decision tree:

1. **Domain match** — code-trained (CodeBERT, GTE-Code) for code; scientific-trained (BGE-Large, E5) for papers; multi-lingual for non-English.
2. **Dimensionality** — 384-dim (`all-MiniLM-L6-v2`) is fastest and good enough for most tasks. 768-dim (`all-mpnet-base-v2`) is the standard. 1024+ for nuanced retrieval.
3. **Cost / latency** — local CPU works for ≤768-dim. GPU-only past 1024-dim. API-based (OpenAI `text-embedding-3-large`) is fastest to ship but locks you in.
4. **Context window** — must accommodate your chunk size + safety margin.

CodeLens_AI uses `sentence-transformers/all-mpnet-base-v2` (768-dim, 514-token window, runs on CPU) — the workhorse default. Worth re-evaluating if we hit retrieval-precision ceilings.

---

### Q5.4 — "What's the biggest mistake in chunk-size optimization?"

**Optimizing the chunker without measuring downstream impact.**

I've seen teams obsess over chunk-boundary quality (ASTs, semantic splitting, hierarchical chunks) while their reranker/prompt was the actual bottleneck. The right loop is:

```
1. Build a 50-query gold test set.
2. Measure end-to-end answer quality (RAGAS or human eval).
3. Change ONE thing.
4. Re-measure.
5. Keep the change only if measurable lift.
```

Chunking is rarely the highest-leverage variable. Reranker on/off, prompt structure, and retrieval top-K are usually higher-leverage. Measure, don't intuit.

---

## Bonus — Curveball Questions

### Q-Bonus.1 — "Your RAG hallucinates. What do you do?"

Diagnose, don't fix blindly. The decomposition:

1. **Is the answer faithful to the retrieved context?** If yes, retrieval is the problem (wrong chunks). If no, the LLM is hallucinating despite correct context.
2. **For retrieval failures:** check context_recall. Low? Improve retrieval (hybrid, reranker, query expansion). High? It's a synthesis problem — LLM picked the wrong source.
3. **For LLM failures:** structured output enforcement (Pydantic), explicit "answer only from context" instruction, chain-of-thought + cite-as-you-go, or ultimately a different model.

Pin the cause before patching. RAG hallucination is a symptom; find the layer.

---

### Q-Bonus.2 — "How do you keep the index fresh?"

Two patterns:

1. **Periodic full re-index** — simple, good for small corpora (<100k docs).
2. **Incremental updates** — hash each chunk's content, only re-embed changed chunks. CodeLens_AI's `parent_id` scheme (`parent::source::name::start-end`) is deterministic — re-running ingestion produces identical IDs for unchanged code, so vector DB upserts are cheap.

For real-time use cases (chat over Slack messages, news feeds): event-driven incremental update on each new document. ChromaDB / pgvector both support this.

---

### Q-Bonus.3 — "Why not just give the LLM the entire codebase via long context?"

Three reasons it fails today:

1. **Cost.** A 100k-token query at GPT-4-class pricing is 10-100× a typical RAG query.
2. **Latency.** First-token latency on a 100k context is multi-second; on a 4k retrieved context, it's sub-second.
3. **Quality.** Lost-in-the-middle (Q4.1). The LLM does worse on 100k of code than on 5 well-retrieved functions.

The fourth reason — and this is the one nobody talks about — is **debuggability**. When RAG gets the wrong answer, you can read the retrieved chunks and see why. When a long-context model gets the wrong answer, the answer is "the attention pattern was suboptimal" — uninterpretable.

---

## Closing — How I'd Frame This in an Interview

If you ask me "what's your favorite part of building RAG systems?" — it's that every component has a clean falsifiable hypothesis:

- *"Retrieval is the bottleneck"* → measurable via context_recall.
- *"Generation is the bottleneck"* → measurable via faithfulness.
- *"Lost-in-the-middle is hurting us"* → measurable by reordering and re-evaluating.

That's rare in ML systems. Most ML problems are "the model is just kinda bad and we don't know why." RAG is **mechanically debuggable**. That's why it's the right tool for production AI today, even though the architecture looks unglamorous compared to a fine-tuned monolith.

That framing usually makes interviewers smile — and it's true.

---

*Companion documents:*
- *`PROJECT_STORY.md` — narrative*
- *`PIPELINE_DEEP_DIVE.md` — architecture*
- *`SECURITY_AND_PRIVACY.md` — hardening*
- *`CHALLENGES_AND_SOLUTIONS.md` — STAR-format war stories*
