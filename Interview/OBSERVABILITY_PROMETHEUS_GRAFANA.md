# Observability — Prometheus & Grafana Architecture

> **Why this doc exists.** Most RAG tutorials default to LangSmith for tracing because it's the easy path. CodeLens_AI is **self-hosted on Prometheus + Grafana** because data sovereignty, cost, and unified system+AI observability are non-negotiable for a code-search product handling private repositories. This doc explains the choice, the implementation, and how I'd defend it in an interview.

---

## Table of Contents
1. [Why Prometheus/Grafana over LangSmith](#1--why-prometheusgrafana-over-langsmith)
2. [Key Metrics — The Golden Signals for RAG](#2--key-metrics--the-golden-signals-for-rag)
3. [Implementation Logic](#3--implementation-logic)
4. [Production-Level Interview Questions](#4--production-level-interview-questions)
5. [Reference: Files in This Repo](#5--reference-files-in-this-repo)

---

## 1. 🚀 Why Prometheus/Grafana over LangSmith?

### 1.1 Data Sovereignty
LangSmith ships every prompt, every retrieved chunk, and every LLM completion to a **third-party SaaS**. For CodeLens_AI, every "trace" contains:
- Proprietary source code from user repositories
- Internal API names, secrets-shaped strings, business logic
- User identifiers tied to private organizational data

Routing that through an external service is a **compliance and IP risk** — failing SOC2 boundaries, breaking GDPR data-residency commitments, and creating an attack surface I don't control. Prometheus + Grafana **stay inside the VPC**: scraping `/metrics` over a private network, persisting to local TSDB. Zero egress of trace content.

> **The principle.** Telemetry should never carry data the application itself wouldn't expose to a public endpoint.

### 1.2 Cost Optimization
LangSmith pricing is **per-trace** (~\$0.0005–\$0.005 per trace at scale). At 1M queries/month that's \$500–\$5000 monthly — *just for observability*. Prometheus + Grafana are **open-source**: cost is a single VM (or sidecar pod) and disk for retention. The cost curve is **flat in query volume**.

| Approach | 100k traces/mo | 10M traces/mo | 100M traces/mo |
|---|---|---|---|
| **LangSmith (per-trace)** | ~\$50–\$500 | ~\$5k–\$50k | ~\$50k–\$500k |
| **Prometheus + Grafana (self-hosted)** | ~\$30 (1 VM) | ~\$80 (bigger VM) | ~\$300 (sharded) |

You don't even hit break-even — Prometheus is cheaper at every scale once you factor in the operational cost of paid SaaS.

### 1.3 Unified Monitoring (AI + System Together)
LangSmith shows you LLM traces. It does **not** show you:
- CPU/RAM on the embedding worker
- Postgres connection pool saturation
- Disk pressure on the ChromaDB volume
- Network latency to Ollama

When p95 latency spikes, **the cause is rarely the LLM in isolation**. It's "GPU was preempted" or "pgvector hit a slow query plan" or "the embedding pod was OOM-killed". Prometheus scrapes both AI metrics (`rag_retrieval_latency_ms`) and system metrics (`container_memory_usage_bytes`, `pg_stat_activity_count`) into the **same time series database**. Grafana renders both on **one dashboard**. One pane of glass. One set of correlations.

> **The architectural win.** A latency spike on the AI dashboard *immediately* aligns with the CPU spike on the system dashboard — same X-axis, same window, no context-switching.

---

## 2. 📊 Key Metrics — The Golden Signals for RAG

Google's SRE book defines four **Golden Signals**: *latency, traffic, errors, saturation*. For RAG, they specialize:

### 2.1 RAG Latency (Decomposed Per-Stage)

The single most important metric. **End-to-end latency alone is useless** — you can't fix what you can't isolate. CodeLens_AI exports a histogram per pipeline stage:

```
rag_retrieval_duration_seconds   # Vector + BM25 retrieval (in main.py)
rag_reranking_latency_ms         # Cross-encoder rerank (OTEL)
rag_generation_latency_ms        # LLM token generation (OTEL — agent_brain.py)
http_request_ttfb_seconds        # User-perceived end-to-end (post-L3)
http_request_duration_seconds    # Total request lifetime (incl. SSE body)
```

> **Why two end-to-end metrics?** TTFB is what users *feel* — the time until the first SSE chunk arrives. `http_request_duration_seconds` is the full lifetime including all token streaming, which scales with answer length and isn't a useful UX signal. SLOs target TTFB.

Histograms (not gauges) — so we get **p50, p95, p99** via PromQL `histogram_quantile`. The p99 is the user-visible tail; p50 is the typical experience.

### 2.2 Token Usage
LLM cost scales with tokens. We track:

```
rag_input_tokens_total{user_id, session_id, model}    # Counter
rag_output_tokens_total{user_id, session_id, model}   # Counter
```

`user_id` allows per-tenant cost attribution; `model` allows cost-per-model comparison when we A/B test Mistral vs GPT-4o.

⚠️ **Cardinality warning** — see Q4.2 below. We do **not** label by `query_text`.

### 2.3 Success / Error Rates

```
rag_queries_total{status}                      # status: success|error
rag_errors_total{error_type, component}        # error_type: timeout|retrieval|generation|llm_5xx
http_requests_total{method, path, status_code} # Standard FastAPI metrics
```

The classic Golden Signal — Prometheus alert fires when error ratio > 5% for 5 minutes (see `alert-rules.yml`).

### 2.4 Semantic Cache Hit Rate

```
rag_cache_hits_total{tenant}
rag_cache_misses_total{tenant}
rag_cache_size{tenant}                  # Gauge
rag_cache_lookup_latency_ms             # Histogram
```

Hit rate is computed in PromQL:
```promql
sum(rate(rag_cache_hits_total[5m])) /
(sum(rate(rag_cache_hits_total[5m])) + sum(rate(rag_cache_misses_total[5m])))
```

Hit rate < 30% triggers a warning — usually means similarity threshold is too tight or cache TTL is too aggressive.

### 2.5 Vector DB Latency

```
chroma_query_latency_ms{collection, n_results}    # Histogram
chroma_collection_size{collection}                # Gauge
pgvector_query_duration_seconds                   # Histogram (for semantic cache pgvector)
```

A bimodal latency distribution on `chroma_query_latency_ms` is the canonical signal of HNSW degradation — half the queries hit the in-RAM graph, half hit cold pages on disk. Triggers index re-warm or HNSW rebuild.

### 2.6 RAG Quality Metrics (RAGAS as Gauges)

This is the one that surprises people. We export evaluation scores as **Prometheus gauges**:

```
rag_faithfulness_score{model, retriever_strategy}    # 0.0 – 1.0
rag_context_recall_score{model, retriever_strategy}
rag_answer_relevancy_score{model, retriever_strategy}
```

A background task runs RAGAS on a sampled fraction (1–5%) of live queries and pushes the rolling average. Now hallucinations become an **alertable metric** — see `alert-rules.yml`'s `LowFaithfullnessScore`.

---

## 3. 🛠️ Implementation Logic

> **Status note.** The code samples below match the **current production state** in `backend/app/main.py` and `backend/app/observability/quality_metrics.py` after the L1–L5 hardening patches landed (see `OBSERVABILITY_AUDIT.md` § Resolution Log).

### 3.1 Exporting Custom Metrics with `prometheus_client`

Each metric is declared **once at module scope** in `backend/app/main.py`. Bounded labels only; RAG-realistic buckets:

```python
# backend/app/main.py (excerpt)
from prometheus_client import Counter, Histogram, Gauge

# AUDIT FIX L1: `endpoint` carries the route TEMPLATE, never raw paths.
# AUDIT FIX L3: `outcome` distinguishes success / client_disconnect / server_error.
request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code", "outcome"],
)

# Total request duration — for SSE, this includes streaming body.
request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency (total — includes SSE body streaming)",
    ["method", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# NEW: Time-to-first-byte. The metric users actually feel for SSE.
request_ttfb = Histogram(
    "http_request_ttfb_seconds",
    "Time to first byte (user-perceived latency, SSE-aware)",
    ["method", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# NEW: Counter for SSE client disconnects — keep separate from 5xx.
http_streams_cancelled = Counter(
    "http_streams_cancelled_total",
    "HTTP streams cancelled by client disconnect (asyncio.CancelledError)",
    ["endpoint"],
)

# RAG-stage latencies with tuned buckets — retrieval is fast, generation is slow.
rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval duration (vector + BM25 fusion)",
    ["retriever_type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)
```

**Design choice.** Labels are kept bounded — `method`, `endpoint` (route template), `status_code` (incl. `499`), `outcome` (`success` / `client_disconnect` / `server_error`), `retriever_type`. Never `user_id` / `session_id` / `query_text` / `repo_id` (those would be cardinality bombs — see `OBSERVABILITY_AUDIT.md` §2.1).

### 3.2 FastAPI Middleware — Route-Template + CancelledError-Aware

The production middleware is route-template-aware and distinguishes client disconnects from server errors:

```python
# backend/app/main.py (production middleware)
import asyncio as _asyncio
from time import perf_counter
from starlette.middleware.base import BaseHTTPMiddleware


def _resolve_route_template(request) -> str:
    """Return the matched route template — critical for label cardinality."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return "unmatched"


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        method = request.method
        route_template = _resolve_route_template(request)

        active_connections.inc()
        start = perf_counter()
        ttfb_seconds = None
        status_code = "200"
        outcome = "success"

        try:
            response = await call_next(request)
            ttfb_seconds = perf_counter() - start          # response headers ready
            status_code = str(response.status_code)
            return response
        except _asyncio.CancelledError:
            # SSE client disconnect — record as 499, NOT 500.
            status_code = "499"
            outcome = "client_disconnect"
            http_streams_cancelled.labels(endpoint=route_template).inc()
            raise
        except Exception:
            status_code = "500"
            outcome = "server_error"
            raise
        finally:
            duration = perf_counter() - start
            request_count.labels(
                method=method, endpoint=route_template,
                status_code=status_code, outcome=outcome,
            ).inc()
            request_duration.labels(method=method, endpoint=route_template).observe(duration)
            if ttfb_seconds is not None:
                request_ttfb.labels(method=method, endpoint=route_template).observe(ttfb_seconds)
            active_connections.dec()
```

The `/metrics` endpoint at `backend/app/main.py:/metrics` returns `generate_latest(REGISTRY)` — Prometheus scrapes it every 5 seconds (per `prometheus.yml`).

### 3.3 OTEL Histogram Pattern — Module-Scope Instruments

Per the L2 audit fix, OTEL instruments in `agent_brain.py` are declared **once at import time**, not per request. Labels (OTEL attributes) carry `model` for cost slicing:

```python
# backend/app/services/agents/agent_brain.py (excerpt)
if HAS_OTEL and meter is not None:
    QUERY_LATENCY_MS = meter.create_histogram(
        name="agent_brain.query_latency_ms",
        description="Query processing latency (end-to-end)",
        unit="ms",
    )
    TOKENS_GENERATED = meter.create_histogram(
        name="agent_brain.tokens_generated",
        description="Tokens generated by LLM (sliced by model and kind)",
        unit="1",
    )

# Per-request usage (no per-request allocation):
QUERY_LATENCY_MS.record(latency_ms, attributes={"model": model_name})
TOKENS_GENERATED.record(tokens, attributes={"model": model_name, "kind": "output"})
```

### 3.4 RAGAS Quality Metrics — `quality_metrics.py`

The L4 audit fix introduced a dedicated module for the quality-side gauges. Single sink → guaranteed labels → stale-data detection:

```python
# backend/app/observability/quality_metrics.py
from prometheus_client import Counter, Gauge

RAG_FAITHFULNESS    = Gauge("rag_faithfulness_score",    ..., ["model", "retriever_strategy"])
RAG_CONTEXT_RECALL  = Gauge("rag_context_recall_score",  ..., ["model", "retriever_strategy"])
RAG_ANSWER_RELEVANCY= Gauge("rag_answer_relevancy_score",..., ["model", "retriever_strategy"])
RAG_CONTEXT_PRECISION = Gauge("rag_context_precision_score", ..., ["model", "retriever_strategy"])

# Stale-data guard: alert rules JOIN against rate(...) on this counter
# so they don't fire while the evaluator is silent.
RAG_QUALITY_SAMPLES = Counter("rag_quality_samples_total", ...,
                              ["model", "retriever_strategy"])


def publish_ragas_scores(scores, *, model, retriever_strategy):
    """Single sink for RAGAS evaluator output. NaN- and None-safe.

    Called from rag_evaluator.py background task after each evaluation.
    """
    # ... defensive parsing, clamping to [0, 1], increments samples counter ...
```

Wiring is one call from the evaluator background task:

```python
from app.observability.quality_metrics import publish_ragas_scores
publish_ragas_scores(scores_dict, model=model_name, retriever_strategy=strategy)
```

### 3.5 Grafana Dashboards via PromQL

The dashboard JSON lives at `grafana/dashboards/rag-pipeline-dashboard.json` and is provisioned automatically (see `grafana/provisioning/`). Key panels (queries updated for the new metric names):

```promql
# 1. End-to-end p95 — TTFB (what users feel)
histogram_quantile(0.95, sum by (le) (rate(http_request_ttfb_seconds_bucket[5m])))

# 2. Where is the time going? Per-stage means.
avg(rate(rag_retrieval_duration_seconds_sum[5m]))
  / avg(rate(rag_retrieval_duration_seconds_count[5m]))

# 3. Real error rate — exclude client disconnects.
sum(rate(http_requests_total{outcome="server_error"}[5m]))
  / sum(rate(http_requests_total[5m]))

# 4. SSE disconnect rate — separate UX problems from infra problems.
sum(rate(http_streams_cancelled_total[5m]))
  / sum(rate(http_requests_total{endpoint="/api/v1/chat/stream"}[5m]))

# 5. RAGAS faithfulness rolling avg, gated on freshness.
avg_over_time(rag_faithfulness_score[1h])
  and on() (sum(rate(rag_quality_samples_total[1h])) > 0)
```

The dashboard has **four rows**: SLO summary → per-stage latency → cache & retrieval health → quality (RAGAS). Read top-to-bottom = "is the system healthy → where → why → what users are seeing".

---

## 4. 🙋 Production-Level Interview Questions

### Q4.1 — How do you monitor hallucinations using Prometheus?
Hallucination is a quality signal, not a binary error. The pattern:

1. Run **RAGAS** on a **sampled** percentage (1–5%) of live queries — full RAGAS on every request is too expensive.
2. Sampling lives in a `BackgroundTasks` hook so it never blocks the user.
3. Push the score to a **Prometheus Gauge** keyed by `(model, retriever_strategy)`:
   ```python
   RAG_FAITHFULNESS.labels(model="mistral-7b", retriever_strategy="hybrid").set(score)
   ```
4. Grafana shows the **rolling average** (`avg_over_time(rag_faithfulness_score[1h])`).
5. Alert fires when faithfulness drops below 0.7 for 30 minutes — this is the equivalent of a "hallucination alarm".

The depth-bomb: **RAGAS is itself an LLM-judged metric, so the gauge is noisy**. We use `avg_over_time` with a 1-hour window to smooth, and **only alert on sustained drops** (not single-sample dips). Otherwise you page on judge variance, not real regressions.

### Q4.2 — How do you handle high-cardinality in Prometheus labels (e.g. session_ids)?
**The hard rule: never label by an unbounded value.** `session_id`, `user_id`, `query_text`, `request_id` would each create one new time series per occurrence. Prometheus storage is per-series; 1M unique sessions = 1M series = OOM.

**The fix is a hierarchy:**

| Level | Where | Example |
|---|---|---|
| **Bounded labels** | Prometheus | `tenant_tier="pro"`, `model="mistral-7b"`, `retriever_strategy="hybrid"` |
| **Sampled high-card** | Logs (Loki / ELK) | Per-request `session_id`, `user_id`, query text |
| **Per-trace high-card** | Traces (Jaeger / Tempo) | Full request tree with all IDs |

When debugging, the operator: (1) Spots a latency spike on Grafana's Prometheus panel, (2) Pivots to the **same time window** in Loki/Jaeger using `{tenant_tier="pro"}` as the bridge, (3) Drills to the offending `session_id`. **Prometheus is for trends, traces are for individual requests.**

We also use **exemplars** — Prometheus 2.x feature that attaches a sampled trace ID to a histogram bucket. Click a p99 outlier in Grafana, jump straight to the Jaeger trace.

### Q4.3 — If latency spikes, how do you use Grafana to find if it's a Vector DB issue or an LLM issue?
The dashboard is **architected to answer this in 30 seconds**:

1. **Open the latency-decomposition panel.** It stacks `rag_retrieval_latency_ms`, `rag_reranking_latency_ms`, `rag_generation_latency_ms` on one chart.
2. **The spiking band identifies the culprit.**
   - Retrieval band thickens → vector DB issue. Check `chroma_query_latency_ms` distribution; check if it's bimodal (HNSW page misses) or uniformly slow (CPU-bound).
   - Reranking band thickens → cross-encoder GPU saturated or batched too aggressively.
   - Generation band thickens → LLM latency. Check `ollama_token_latency_ms` and `ollama_queue_depth`.
3. **Cross-reference system metrics on the same time window.** A Postgres connection-pool saturation (`pg_stat_activity_count` near `max_connections`) explains a retrieval spike *causally*, not just correlatively.
4. **Click an exemplar** in the spiking panel to jump to the Jaeger trace of a representative slow request.

The discipline: never stop at "latency is high" — always answer **which stage**, **why**, and **what system metric correlates**.

### Q4.4 — Explain your alerting strategy: when should a developer get a PagerDuty/Slack notification?
Alerts are tiered. Most teams over-alert; the cost is **alert fatigue** which is worse than no alerts.

| Severity | Channel | Example | Response time |
|---|---|---|---|
| **Critical (PagerDuty)** | Wake someone up | `PostgresDown`, `OllamaDown`, error rate > 5% for 5m | < 15 min |
| **Warning (Slack #ops)** | Look at it tomorrow | `LowCacheHitRate`, `HighRetrievalLatency` for 10m, `LowFaithfullnessScore` | < 24 hr |
| **Info (Dashboard only)** | Just visible | `CacheSizeAlert`, gradual trend regressions | N/A |

**The principles I apply:**

1. **Alert on symptoms, not causes.** Page on "users see errors" (high `rag_errors_total` rate), not on "CPU > 80%". CPU might be fine to ignore at night.
2. **Multi-window, multi-burn-rate** for SLOs. "Error rate > 5% for 5m" *and* "error rate > 1% for 1h" — catches both fast burns and slow drifts.
3. **`for: 5m` minimum** on warnings to swallow flapping.
4. **Runbook URL in every alert annotation.** If the on-call has to think, you've already failed.
5. **Quarterly alert audit** — if an alert never fired or always fired, delete or fix it.

The actual rules live in `alert-rules.yml` — `HighErrorRate` (critical), `HighRetrievalLatency` (warning), `LowFaithfullnessScore` (warning), and infra-down alerts are critical.

### Q4.5 — How do you track drift in embedding model performance over time?
Three approaches stacked. Drift is sneaky — your model didn't change, but your **users' query distribution did**.

1. **Live recall via canary queries.** A small set (50–100) of pinned `(query, expected_chunk_id)` pairs runs every hour against production. Push `embedding_canary_recall_at_5` as a gauge. Sudden drops = something changed (re-index, model swap, data corruption).

2. **RAGAS context_recall on sampled traffic.** The same sampling that catches faithfulness drift catches retrieval-recall drift. Monitor `avg_over_time(rag_context_recall_score[24h])` for slow downward trends.

3. **Embedding distribution shift.** At ingest time, log the **mean and variance per dimension** of recently-embedded chunks. Compare a rolling 7-day window to a baseline. A `KL divergence` shift > threshold → investigate. Common cause: a new document type entered the corpus that the embedder represents differently.

**The depth-bomb:** the most common embedding "drift" isn't model drift — it's **data drift**. The model is the same; users started asking different questions or new documents got ingested. Distinguish the two with: *"did the canary recall drop while the live recall dropped?"* — if both, model issue. If only live, data issue.

### Q4.6 — How do you measure cost per query in Grafana?
Cost = `tokens × price`. We export raw token counters and compute cost in PromQL using a **constant metric** for price:

```promql
sum by (model) (rate(rag_output_tokens_total[1h]))
  * on (model) group_left
sum by (model) (llm_price_per_token_dollars)
```

`llm_price_per_token_dollars` is a static gauge updated whenever vendor pricing changes. Per-tenant attribution comes from `tenant_tier` labels:

```promql
sum by (tenant_tier) (rate(rag_output_tokens_total[24h]) * 86400)  # Daily token burn
```

This **catches cost runaways before the bill arrives** — typical cause is a power user looping a buggy script. We alert when a tier crosses a daily token budget.

### Q4.7 — How do you handle `/metrics` endpoint security?
Three layers:

1. **Network isolation.** `/metrics` binds only to the internal network interface, not the public ingress. Prometheus scrapes via service-mesh DNS.
2. **Auth.** If exposed beyond the cluster, basic auth via reverse-proxy with credentials in Prometheus's `scrape_configs` `basic_auth`.
3. **No PII in label values.** Even internal observers shouldn't see user emails as label values. Tenant tier yes; tenant identity no.

Bonus: don't expose `/metrics` from production-only endpoints. Have a dedicated metrics port (e.g. `:9090` separate from `:8000`) so authn rules diverge cleanly.

### Q4.8 — What's your retention strategy?
**Hot vs cold.** Prometheus local TSDB stores 15 days hot — fast queries, dashboards, alerting. Beyond that we ship to **long-term storage** (Thanos / Cortex / VictoriaMetrics) at downsampled resolution:

| Window | Resolution | Use |
|---|---|---|
| 0–15 days | 15s scrape | Live dashboards, alerts |
| 15–90 days | 5-min downsampled | Trend analysis, weekly review |
| 90 days – 2 years | 1-hour downsampled | Capacity planning, SLO review |

The cost discipline: **never store full resolution beyond the alerting window**. Nobody investigates a p99 spike from 6 months ago at 15-second resolution.

### Q4.9 — How do you instrument streaming (SSE) responses?
Streaming complicates "request duration" — when does the request end? Three relevant metrics:

```
rag_stream_ttfb_ms          # Time-to-first-byte (= LLM first-token)
rag_stream_duration_ms      # Total stream lifetime
rag_stream_cancelled_total  # Counter of client disconnects
```

**TTFB is the metric users feel.** Total duration is dominated by token count, not infrastructure quality. We alert on TTFB, monitor total duration as a trend.

We also track **cancellations** — see Q4.13. CodeLens_AI's SSE generator catches `asyncio.CancelledError` and increments this counter so we can distinguish "users navigated away" from "actual errors".

### Q4.10 — How do you compare Prometheus to OpenTelemetry?
They're **complementary, not competitive**.

- **Prometheus** is a metrics store + alerter — a finished system. Pull-based scraping, TSDB, PromQL, AlertManager.
- **OpenTelemetry** is a **vendor-neutral instrumentation standard** — defines APIs and SDKs for metrics, traces, logs. The OTEL collector can *export* metrics to Prometheus, traces to Jaeger, logs to Loki.

**My architecture uses both.** The application code uses the OTEL SDK to emit metrics and traces. The OTEL collector receives them and:
- Forwards metrics to Prometheus (or exposes a `/metrics` endpoint Prometheus scrapes).
- Forwards traces to Jaeger.
- Forwards logs to Loki.

Result: if I ever switch storage backends, the application code doesn't change. **OTEL is the abstraction; Prometheus is one (excellent) backend.**

### Q4.11 — When would you use a histogram vs a summary in Prometheus?
Both compute quantiles, but the trade-offs differ:

| | **Histogram** | **Summary** |
|---|---|---|
| **Quantile compute** | Server-side via `histogram_quantile()` | Client-side, pre-computed |
| **Aggregatable across instances?** | ✅ Yes — sum buckets, then quantile | ❌ No — quantiles don't aggregate |
| **Bucket cost** | Need to choose buckets up front | No buckets needed |
| **Use when** | Distributed services, need cluster-wide percentiles | Single instance, exact quantiles |

**Always histograms for RAG.** Multiple FastAPI replicas, multiple workers — only histograms aggregate correctly. Bucket choice matters: pick buckets covering your **actual latency range** (50ms to 10s for RAG); too few buckets = imprecise quantiles, too many = storage cost.

### Q4.12 — How do you alert on SLOs (Service Level Objectives)?
SLO example: **"99% of queries complete in under 3 seconds, measured over a 30-day window"**.

**Multi-burn-rate alerts** are the right pattern:

```yaml
# Fast burn — we'd exhaust the monthly error budget in 1 hour
- alert: SLOBurnRateFast
  expr: |
    (
      sum(rate(rag_e2e_latency_ms_bucket{le="3000"}[5m]))
      / sum(rate(rag_e2e_latency_ms_count[5m]))
    ) < 0.985
  for: 5m
  labels: { severity: critical }

# Slow burn — exhausting budget over the whole month
- alert: SLOBurnRateSlow
  expr: |
    (... same expr over 1h window ...) < 0.995
  for: 1h
  labels: { severity: warning }
```

Two alerts. Fast burn pages on real outages; slow burn warns on creeping degradation. **This is from Google's SRE Workbook chapter 5** and it's the production standard.

### Q4.13 — How do you debug a Grafana panel that shows "no data"?
A diagnostic checklist:

1. **Is Prometheus scraping the target?** Check `up{job="codelens-backend"}` in PromQL.
2. **Is the metric being emitted?** Curl `localhost:8000/metrics` directly, grep for the metric name. If absent, the application code path that emits it never executed.
3. **Are the labels matching?** PromQL `rag_retrieval_latency_ms_bucket` (no labels) shows everything. If your panel filters by `tenant_tier="pro"`, maybe no pro-tier traffic in the window.
4. **Is the time window correct?** Grafana's relative range can lie if NTP drifts. Check absolute timestamps.
5. **Is the rate window > scrape interval?** `rate(metric[1m])` with a 5s scrape interval needs at least 2–3 samples; bump to `[2m]` or longer.

90% of "no data" is one of `up == 0`, application not exporting, or label mismatch.

### Q4.14 — How do you instrument vector-DB-specific metrics that the DB doesn't expose?
Wrap the client. ChromaDB doesn't natively export Prometheus metrics, so the application's vector-store wrapper does:

```python
async def query(self, query_embedding, k):
    with observe_stage(CHROMA_QUERY_LATENCY, collection=self.name, k=str(k)):
        results = await self._client.query(query_embedding, n_results=k)
    CHROMA_RESULTS_RETURNED.labels(collection=self.name).observe(len(results))
    return results
```

This gives you **what the application observes**, which is what matters — not what the DB internally measures. Network round-trip, serialization, connection-pool wait — all included.

### Q4.15 — How do you avoid "alert fatigue" in a small on-call rotation?
Five disciplines:

1. **Cap the number of paging alerts.** Five at most. If you have ten, you have zero.
2. **Every paging alert must be actionable** — link a runbook with concrete steps.
3. **Quarterly alert review** — for each alert: did it fire this quarter? Did the response actually fix anything? If no to either, delete or downgrade.
4. **Group related alerts** in AlertManager — one notification when "the database is on fire" instead of fifteen.
5. **Distinguish symptom from cause.** Page on user impact (errors, SLO burn). Don't page on causes (CPU usage) — those go to dashboards.

The mantra from Google SRE: **"Every page should be a novel, valuable problem."** Repeated pages mean automation is missing.

### Q4.16 — How do you trace a single slow request across embedding → retrieval → rerank → LLM?
This is **distributed tracing** territory, not Prometheus. Pattern:

1. Every request starts with a **trace ID** (e.g. via OTEL middleware).
2. Each pipeline stage opens a **span** with stage-specific attributes (`stage="retrieval"`, `n_chunks=20`).
3. Spans are sent to **Jaeger / Tempo** via OTLP.
4. **Prometheus exemplars** link individual histogram observations back to a specific trace ID. Click a slow point in Grafana → land in Jaeger on that exact request.

The integration is: Prometheus answers *"are we slow in aggregate?"*; Jaeger answers *"why was this one request slow?"* The two together close the loop.

### Q4.17 — What metrics do you specifically NOT collect, and why?
Negative space matters as much as positive:

1. **Raw query text or chunk content.** Privacy + cardinality. Use logs (with redaction) instead.
2. **`user_id` as a Prometheus label.** Cardinality. Track via logs.
3. **Per-prompt token counts as labels.** Use a histogram of token counts; don't label-explode.
4. **Anything Prometheus can compute from existing metrics.** Don't store both `cache_hit_rate` and the underlying counters — derive in PromQL.
5. **Duplicate metrics from middleware AND manual code.** Pick one location to instrument each thing.

The principle: **every metric has a maintenance cost**. Each one needs a dashboard panel, an alert (or a justification for not having one), retention disk, and mental overhead at incident time.

---

## 5. 📁 Reference: Files in This Repo

| File | Purpose |
|---|---|
| `prometheus.yml` | Scrape configs — points Prometheus at FastAPI `/metrics`, Jaeger, optional pg_exporter |
| `alert-rules.yml` | Alert rules — RAG latency (`histogram_quantile`-based, post-L5), cache hit rate, error rate, infra down, RAGAS faithfulness with stale-data guard |
| `grafana/dashboards/rag-pipeline-dashboard.json` | The canonical RAG dashboard JSON |
| `grafana/provisioning/dashboards/` | Auto-provisioning config so Grafana finds the dashboard on boot |
| `grafana/provisioning/datasources/` | Wires Prometheus as the default Grafana datasource |
| `docker-compose.monitoring.yml` | Spins up Prometheus + Grafana + Jaeger as a stack |
| `backend/app/main.py` | Prometheus metric declarations + `PrometheusMiddleware` (route-template + 499-aware) + `/metrics` endpoint |
| `backend/app/services/agents/agent_brain.py` | OTEL `QUERY_LATENCY_MS` / `TOKENS_GENERATED` declared at module scope (post-L2) |
| `backend/app/observability/quality_metrics.py` | RAGAS gauges + `publish_ragas_scores()` sink (post-L4) |
| `backend/app/observability/otel_config.py` | OTEL collector / exporter setup |
| `backend/app/observability/rag_evaluator.py` | RAGAS background evaluator — calls `publish_ragas_scores()` |
| `OBSERVABILITY_AUDIT.md` | Code-level audit + Resolution Log for L1–L5 |

---

## Closing — How I Frame This in an Interview

> *"LangSmith is a great tool when your priorities are speed-to-ship and you're comfortable putting prompts and retrieved content on a third party's infrastructure. CodeLens_AI handles private code, so I built on Prometheus + Grafana — the same stack that monitors the rest of the platform. The win isn't observability per se; it's **unified observability** — one Grafana pane shows AI quality, infrastructure health, and cost burn on the same X-axis. When something breaks, the answer is two clicks away, not in three different SaaS UIs."*

That's the **architectural** version of the answer. The **technical** version is in this document.

---

*Companion documents:*
- *`PROJECT_STORY.md` — narrative*
- *`PIPELINE_DEEP_DIVE.md` — architecture*
- *`SECURITY_AND_PRIVACY.md` — hardening*
- *`CHALLENGES_AND_SOLUTIONS.md` — STAR-format war stories*
- *`RAG_INTERVIEW_PREP_Q&A.md` — first-pass Q&A primer*
- *`ADVANCED_RAG_ENGINEERING_MASTER_QA.md` — 110-question master drill*
