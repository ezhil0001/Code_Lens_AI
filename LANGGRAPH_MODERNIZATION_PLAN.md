# CodeLens AI — LangGraph Multi-Agent Modernization Plan

> **Objective:** Transform the current single-brain RAG pipeline into a production-grade  
> LangGraph-based multi-agent platform with a Supervisor orchestrator, long-term/short-term  
> memory, checkpointing, time-travel debugging, Human-in-the-Loop workflows, guardrails,  
> streaming execution, and full runtime observability.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Target Architecture](#2-target-architecture)
3. [Complete Feature Roadmap](#3-complete-feature-roadmap)
4. [Phased Implementation Schedule](#4-phased-implementation-schedule)
5. [Detailed Implementation Specifications](#5-detailed-implementation-specifications)
   - [Phase A — LangGraph Foundation](#phase-a--langgraph-foundation)
   - [Phase B — Multi-Agent Supervisor System](#phase-b--multi-agent-supervisor-system)
   - [Phase C — Memory Architecture](#phase-c--memory-architecture)
   - [Phase D — Checkpointing & Time-Travel](#phase-d--checkpointing--time-travel)
   - [Phase E — Human-in-the-Loop](#phase-e--human-in-the-loop)
   - [Phase F — Middleware & Guardrails](#phase-f--middleware--guardrails)
   - [Phase G — Streaming & API Layer](#phase-g--streaming--api-layer)
   - [Phase H — Runtime Observability](#phase-h--runtime-observability)
   - [Phase I — Frontend Modernization](#phase-i--frontend-modernization)
   - [Phase J — Production Hardening](#phase-j--production-hardening)
6. [Git Commit Convention](#6-git-commit-convention)
7. [Git Commit Log by Phase](#7-git-commit-log-by-phase)
8. [Testing Strategy & Startup Validation](#8-testing-strategy--startup-validation)
9. [Dependency Additions](#9-dependency-additions)
10. [Migration Risk Register](#10-migration-risk-register)

---

## 1. Current State Assessment

### What Exists Today

| Component | Location | Status |
|---|---|---|
| Single-brain orchestrator | `services/agents/agent_brain.py` | Monolithic — 1,157 lines |
| Routing (keyword classifier) | `services/agents/agentic_router.py` | Rule-based, no state |
| Hybrid retrieval | `services/retrieval/retriever_engine.py` | Thread-safe singleton |
| BGE reranker | Inside retriever_engine.py | Embedded, not composable |
| Semantic cache | `api/chat.py::SemanticCache` | pgvector, per-user scoped |
| Conversation memory | `PostgresChatMessageHistory` | LangChain primitive only |
| Evaluation | `observability/rag_evaluator.py` | Async background task |
| Observability | `observability/otel_config.py` | OTEL + Prometheus |
| Auth / rate limiting | `auth/`, `middleware/` | Enterprise-grade, keep as-is |

### Identified Gaps

- **No true graph execution** — all routing is `if/elif` inside a single function.
- **No persistent state across turns** — memory is append-only chat history; no graph checkpoint.
- **No agent specialization** — one brain handles code queries, doc queries, debugging, and architecture equally poorly.
- **No time-travel** — impossible to replay a query at an earlier graph state.
- **No HIL** — agent decisions cannot pause for human review.
- **No input/output guardrails** — prompt injection and toxic content reach the LLM unchecked.
- **No cross-session long-term memory** — each session starts cold.
- **Streaming is per-token SSE** — no structured event envelope (token / tool_call / checkpoint / interrupt).

---

## 2. Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLIENT  (Angular + SSE)                               │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │  POST /api/v2/chat/stream
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GUARDRAIL MIDDLEWARE LAYER                               │
│   InputGuardrail → PII Scrubber → PromptInjectionDetector → ContentFilter   │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH SUPERVISOR GRAPH                                │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                    SupervisorAgent (StateGraph)                       │  │
│   │                                                                       │  │
│   │   ┌────────────┐   route_decision    ┌───────────────────────────┐  │  │
│   │   │  __start__ │ ─────────────────►  │  intent_classifier node   │  │  │
│   │   └────────────┘                     └──────────┬────────────────┘  │  │
│   │                                                  │                    │  │
│   │              ┌───────────────┬──────────────────┬─────────────┐     │  │
│   │              ▼               ▼                  ▼             ▼     │  │
│   │    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │  │
│   │    │  CodeAgent  │  │  DocAgent    │  │  DebugAgent  │  │  Arch  │ │  │
│   │    │  SubGraph   │  │  SubGraph    │  │  SubGraph    │  │ Agent  │ │  │
│   │    └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └───┬────┘ │  │
│   │           └────────────────┴──────────────────┴──────────────┘      │  │
│   │                                    │                                  │  │
│   │                                    ▼                                  │  │
│   │                         ┌──────────────────┐                          │  │
│   │                         │ synthesizer node │                          │  │
│   │                         └────────┬─────────┘                          │  │
│   │                                  │                                    │  │
│   │                    ┌─────────────┴──────────────┐                    │  │
│   │                    │  HIL interrupt (optional)  │                    │  │
│   │                    └─────────────┬──────────────┘                    │  │
│   │                                  │                                    │  │
│   │                         ┌────────▼────────┐                          │  │
│   │                         │  response_node  │                          │  │
│   │                         └─────────────────┘                          │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   Checkpointer: PostgresSaver  │  LTM Store: pgvector  │  STM: StateGraph  │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OUTPUT GUARDRAIL + STREAMING LAYER                        │
│   OutputGuardrail → CitationVerifier → SSE EventEmitter                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Inventory

| Agent | Responsibilities | Key Tools |
|---|---|---|
| **SupervisorAgent** | Intent classification, agent dispatch, state coordination, synthesis | LLM router, memory read |
| **CodeAgent** | Code search, function-level context, implementation explanation | `code_search`, `symbol_lookup`, `ast_analyze` |
| **DocAgent** | KT document retrieval, architecture explanation, onboarding answers | `doc_search`, `section_lookup` |
| **DebugAgent** | Error diagnosis, stack trace analysis, root-cause reasoning | `code_search`, `error_pattern_lookup`, `dependency_graph` |
| **ArchitectureAgent** | System design questions, cross-component data-flow, ADR lookup | `doc_search`, `code_search` (hybrid), `diagram_generator` |
| **WebSearchAgent** | Current documentation, CVE lookups, external library references | `tavily_search` |
| **EvaluatorAgent** | RAGAS scoring, quality gate enforcement | `ragas_evaluate` |

---

## 3. Complete Feature Roadmap

### 3.1 Core LangGraph Migration

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-01 | Define `AgentState` TypedDict with all graph state fields | P0 | Low |
| F-02 | Replace `agent_brain.py` monolith with `StateGraph` definition | P0 | High |
| F-03 | Port `AgenticRouter` to LangGraph `intent_classifier` node | P0 | Medium |
| F-04 | Build `SupervisorAgent` as the root `StateGraph` | P0 | High |
| F-05 | Build `CodeAgent` as a compiled sub-graph | P0 | High |
| F-06 | Build `DocAgent` as a compiled sub-graph | P0 | Medium |
| F-07 | Build `DebugAgent` as a compiled sub-graph | P1 | Medium |
| F-08 | Build `ArchitectureAgent` as a compiled sub-graph | P1 | Medium |
| F-09 | Build `WebSearchAgent` (Tavily) as a compiled sub-graph | P1 | Low |
| F-10 | Wire all sub-graphs into Supervisor via `add_node` + conditional edges | P0 | High |
| F-11 | Implement `synthesizer` node for multi-agent result fusion | P0 | Medium |

### 3.2 Memory Architecture

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-12 | Short-term memory: `AgentState` carries conversation window | P0 | Low |
| F-13 | Long-term memory: pgvector store for cross-session user facts | P1 | High |
| F-14 | Long-term memory: entity extractor (user preferences, code facts) | P1 | High |
| F-15 | Memory retrieval node: inject relevant LTM facts into each turn | P1 | Medium |
| F-16 | Memory write node: persist salient facts post-turn | P1 | Medium |
| F-17 | Memory namespace isolation: per-user, per-org partitioning | P0 | Medium |
| F-18 | Episodic memory: store conversation episodes with semantic index | P2 | High |

### 3.3 Checkpointing & Time-Travel

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-19 | Integrate `PostgresSaver` as LangGraph checkpointer | P0 | Medium |
| F-20 | Thread-level checkpoint on every node completion | P0 | Low |
| F-21 | `GET /api/v2/sessions/{session_id}/checkpoints` endpoint | P1 | Medium |
| F-22 | Time-travel: `GET /api/v2/sessions/{session_id}/replay/{checkpoint_id}` | P1 | High |
| F-23 | Time-travel: branch from historical checkpoint (new thread) | P2 | High |
| F-24 | Checkpoint diff viewer — state delta between any two checkpoints | P2 | Medium |

### 3.4 Human-in-the-Loop

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-25 | Interrupt node at synthesizer for high-confidence-required queries | P1 | Medium |
| F-26 | `POST /api/v2/sessions/{session_id}/resume` to inject human feedback | P1 | Medium |
| F-27 | HIL confidence threshold configuration (per user/org) | P2 | Low |
| F-28 | Approval workflow: agent proposes action, human approves/rejects | P2 | High |
| F-29 | Audit log: every HIL interrupt and resolution persisted | P1 | Low |

### 3.5 Middleware & Guardrails

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-30 | Input guardrail: PII detection and scrubbing | P1 | Medium |
| F-31 | Input guardrail: prompt injection detection | P0 | Medium |
| F-32 | Input guardrail: toxic/harmful content filter | P1 | Medium |
| F-33 | Input guardrail: query length / token budget enforcement | P0 | Low |
| F-34 | Output guardrail: citation verifier (answer grounded in context) | P1 | High |
| F-35 | Output guardrail: code safety scanner (no `rm -rf`, `eval`, etc.) | P1 | Medium |
| F-36 | Output guardrail: PII leak prevention in responses | P1 | Medium |
| F-37 | LangGraph middleware: pre-node and post-node hook decorators | P1 | Medium |
| F-38 | Retry middleware: automatic node retry with exponential backoff | P1 | Medium |

### 3.6 Streaming Execution

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-39 | LangGraph `graph.astream_events()` integration | P0 | Medium |
| F-40 | Structured SSE envelope: `{type, data, metadata, checkpoint_id}` | P0 | Low |
| F-41 | Stream token events: `{type: "token", content: "...", agent: "CodeAgent"}` | P0 | Low |
| F-42 | Stream tool_call events: `{type: "tool_call", tool: "code_search", input: ...}` | P1 | Low |
| F-43 | Stream checkpoint events: `{type: "checkpoint", id: "...", state_summary: ...}` | P1 | Low |
| F-44 | Stream interrupt events: `{type: "interrupt", reason: "...", awaiting_input: true}` | P1 | Low |
| F-45 | Stream agent_switch events: `{type: "agent_switch", from: "Supervisor", to: "CodeAgent"}` | P1 | Low |
| F-46 | Reconnect support: stream from specific checkpoint on reconnect | P2 | Medium |

### 3.7 Runtime Observability

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-47 | Per-node LangGraph OTEL spans: `langgraph.node.{name}.duration` | P0 | Medium |
| F-48 | Per-agent token usage histogram: `agent.tokens.{agent_name}` | P1 | Low |
| F-49 | Graph topology metrics: edges traversed, nodes visited per turn | P1 | Medium |
| F-50 | Interrupt rate metric: `supervisor.hil_interrupt_rate` | P1 | Low |
| F-51 | Memory hit/miss metrics: `ltm.cache_hit_rate`, `stm.context_length` | P1 | Low |
| F-52 | LangSmith tracing integration (optional cloud) | P2 | Low |
| F-53 | Grafana dashboard: Agent Activity dashboard panel | P1 | Medium |
| F-54 | Alert: `AgentDeadlock` — graph stuck in same node for > 30s | P1 | Low |
| F-55 | Alert: `HILBacklog` — more than 10 pending HIL interrupts | P2 | Low |

### 3.8 Advanced Agent Capabilities

| # | Feature | Priority | Complexity |
|---|---|---|---|
| F-56 | Tool use with schema validation (Pydantic tool args) | P0 | Medium |
| F-57 | Parallel tool dispatch in CodeAgent sub-graph | P1 | Medium |
| F-58 | Dynamic tool registration at runtime (plugin pattern) | P2 | High |
| F-59 | Agent self-reflection node: evaluate own answer quality | P2 | High |
| F-60 | Adaptive retrieval: agent adjusts `top_k` based on query complexity | P1 | Medium |
| F-61 | Agentic code generation: DebugAgent proposes fix diffs | P2 | High |
| F-62 | Multi-turn planning: supervisor decomposes complex queries into sub-tasks | P2 | High |

---

## 4. Phased Implementation Schedule

```
Phase A  ─── LangGraph Foundation          │ Week 1-2   │ F-01 to F-04, F-39-40
Phase B  ─── Multi-Agent System            │ Week 2-4   │ F-05 to F-11, F-56-57
Phase C  ─── Memory Architecture           │ Week 4-5   │ F-12 to F-18
Phase D  ─── Checkpointing & Time-Travel   │ Week 5-6   │ F-19 to F-24
Phase E  ─── Human-in-the-Loop             │ Week 6-7   │ F-25 to F-29
Phase F  ─── Middleware & Guardrails        │ Week 7-8   │ F-30 to F-38
Phase G  ─── Streaming & API Layer         │ Week 8-9   │ F-41 to F-46
Phase H  ─── Runtime Observability         │ Week 9-10  │ F-47 to F-55
Phase I  ─── Frontend Modernization        │ Week 10-11 │ Angular agent-event UI
Phase J  ─── Production Hardening          │ Week 11-12 │ F-58 to F-62, load tests
```

### Milestone Gates

| Milestone | Definition of Done |
|---|---|
| **M1** (end of Phase B) | Full multi-agent graph runs end-to-end; all existing tests pass; CI green |
| **M2** (end of Phase D) | Every graph turn is checkpointed; time-travel replay works in staging |
| **M3** (end of Phase F) | All P0/P1 guardrails active; red-team prompt injection tests pass |
| **M4** (end of Phase H) | Grafana "Agent Activity" dashboard live; all P0/P1 alerts firing correctly |
| **M5** (end of Phase J) | Load test: 50 concurrent users, p95 < 2s TTFB, zero CancelledError leaks |

---

## 5. Detailed Implementation Specifications

---

### Phase A — LangGraph Foundation

---

#### A.1 — Define `AgentState` (F-01)

**File:** `backend/app/graph/state.py`

**What to build:**
Create the central `TypedDict` that flows through every node in the LangGraph graph. This is the single source of truth for all information that agents read and write. Treat it like a Redux store — immutable per-node, reducers produce the next state.

**Fields to include:**

```python
from typing import TypedDict, Annotated, Sequence, Optional, Any
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # ── Core conversation ──────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # operator.add means: each node appends, never overwrites

    # ── Request context ────────────────────────────────────────────────
    user_id: str
    session_id: str           # namespaced: "{user_id}::{raw_session_id}"
    org_id: Optional[str]
    query: str                # original user query (immutable after set)
    query_embedding: Optional[list[float]]  # cached — computed once

    # ── Routing & intent ──────────────────────────────────────────────
    intent: Optional[str]     # "CODE_LOOKUP" | "DEBUG" | "ARCHITECTURE" | ...
    routing_decision: Optional[str]  # "CodeAgent" | "DocAgent" | ...
    routing_confidence: float
    metadata_filter: Optional[dict]  # ChromaDB where= clause

    # ── Retrieval ──────────────────────────────────────────────────────
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    parent_contexts: dict[str, str]

    # ── Agent outputs ──────────────────────────────────────────────────
    agent_responses: dict[str, str]  # agent_name -> partial answer
    active_agent: Optional[str]
    tool_calls: list[dict]
    tool_results: list[dict]

    # ── Memory ────────────────────────────────────────────────────────
    short_term_window: list[dict]    # last N turns from SQLAlchemy
    long_term_facts: list[str]       # retrieved LTM facts for this query
    memory_write_queue: list[dict]   # facts to persist post-turn

    # ── Guardrails ────────────────────────────────────────────────────
    guardrail_passed: bool
    guardrail_violations: list[str]
    pii_scrubbed_query: Optional[str]

    # ── HIL ───────────────────────────────────────────────────────────
    hil_required: bool
    hil_reason: Optional[str]
    hil_human_input: Optional[str]
    hil_approved: Optional[bool]

    # ── Final response ─────────────────────────────────────────────────
    final_response: Optional[str]
    sources: list[dict]
    cache_hit: bool
    evaluation_queued: bool

    # ── Observability ─────────────────────────────────────────────────
    span_id: Optional[str]
    checkpoint_id: Optional[str]
    nodes_visited: list[str]
    total_latency_ms: float
```

**Key design rules:**
- Use `Annotated[Sequence[BaseMessage], operator.add]` for messages — this is the LangGraph reducer pattern. Every node that appends a message works correctly even with parallel execution.
- All other fields use last-write-wins (default LangGraph behavior) unless you need a custom reducer.
- Never mutate the state object — always return a dict with only the fields you are changing.
- `query` is set by the entry node and never modified — downstream nodes read it.

---

#### A.2 — Build the Supervisor Graph Shell (F-02, F-04)

**File:** `backend/app/graph/supervisor_graph.py`

**What to build:**
The root `StateGraph` that wires all agents together. Use LangGraph's `StateGraph(AgentState)` pattern. This replaces `agent_brain.py`'s monolithic `process_query()` method.

**Node sequence:**
```
__start__
    │
    ▼
input_guardrail_node       ← validates & scrubs input
    │
    ▼
cache_check_node           ← semantic cache lookup (existing SemanticCache)
    │  HIT ──────────────────────────────────────────────► response_node
    │  MISS
    ▼
memory_read_node           ← load STM window + LTM facts
    │
    ▼
intent_classifier_node     ← classify intent, set routing_decision
    │
    ├── "CodeAgent"  ──────► code_agent_node  (compiled sub-graph)
    ├── "DocAgent"   ──────► doc_agent_node
    ├── "DebugAgent" ──────► debug_agent_node
    ├── "ArchAgent"  ──────► arch_agent_node
    └── "WebAgent"   ──────► web_agent_node
            │
            ▼
    synthesizer_node       ← merge multi-agent results (if HYBRID)
            │
            ▼
    hil_check_node         ← interrupt if hil_required == True
            │  (interrupt)
            │  RESUME ──────► (human input injected into state)
            │
            ▼
    output_guardrail_node  ← citation check, code safety, PII
            │
            ▼
    response_node          ← assemble final SSE + cache write + memory write
            │
            ▼
        __end__
```

**Implementation pattern for each node:**

```python
# Every node is a pure async function:
async def intent_classifier_node(state: AgentState, config: RunnableConfig) -> dict:
    """Classifies query intent and sets routing_decision."""
    # 1. Read from state (never mutate state directly)
    query = state["query"]
    
    # 2. Do work
    intent, confidence, metadata_filter = await classify_intent(query)
    
    # 3. Return ONLY the fields that changed
    return {
        "intent": intent,
        "routing_decision": intent_to_agent(intent),
        "routing_confidence": confidence,
        "metadata_filter": metadata_filter,
        "nodes_visited": state["nodes_visited"] + ["intent_classifier"],
    }
```

**Conditional routing:**

```python
def route_to_agent(state: AgentState) -> str:
    """Return the name of the next node based on routing_decision."""
    if state.get("cache_hit"):
        return "response_node"
    decision = state.get("routing_decision", "CodeAgent")
    return decision  # must match a node name in the graph

graph.add_conditional_edges("intent_classifier_node", route_to_agent)
```

**Graph compilation:**

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

def build_supervisor_graph(checkpointer: PostgresSaver) -> CompiledGraph:
    builder = StateGraph(AgentState)
    
    # Add all nodes
    builder.add_node("input_guardrail_node", input_guardrail_node)
    builder.add_node("cache_check_node", cache_check_node)
    builder.add_node("memory_read_node", memory_read_node)
    builder.add_node("intent_classifier_node", intent_classifier_node)
    builder.add_node("CodeAgent", code_agent_subgraph)
    builder.add_node("DocAgent", doc_agent_subgraph)
    builder.add_node("DebugAgent", debug_agent_subgraph)
    builder.add_node("ArchAgent", arch_agent_subgraph)
    builder.add_node("WebAgent", web_agent_subgraph)
    builder.add_node("synthesizer_node", synthesizer_node)
    builder.add_node("hil_check_node", hil_check_node)
    builder.add_node("output_guardrail_node", output_guardrail_node)
    builder.add_node("response_node", response_node)
    
    # Add edges
    builder.set_entry_point("input_guardrail_node")
    builder.add_edge("input_guardrail_node", "cache_check_node")
    builder.add_conditional_edges("cache_check_node", route_cache)
    builder.add_edge("memory_read_node", "intent_classifier_node")
    builder.add_conditional_edges("intent_classifier_node", route_to_agent)
    for agent in ["CodeAgent", "DocAgent", "DebugAgent", "ArchAgent", "WebAgent"]:
        builder.add_edge(agent, "synthesizer_node")
    builder.add_edge("synthesizer_node", "hil_check_node")
    builder.add_conditional_edges("hil_check_node", route_hil)
    builder.add_edge("output_guardrail_node", "response_node")
    builder.add_edge("response_node", END)
    
    return builder.compile(checkpointer=checkpointer, interrupt_before=["hil_check_node"])
```

---

#### A.3 — LangGraph Streaming Integration (F-39, F-40)

**File:** `backend/app/graph/streaming.py`

**What to build:**
Replace the current raw `asyncio.CancelledError`-guarded SSE generator with a proper LangGraph event stream consumer that maps graph events to structured SSE envelopes.

**SSE Envelope Schema:**

```python
@dataclass
class SSEEvent:
    type: str          # "token" | "tool_call" | "tool_result" |
                       # "agent_switch" | "checkpoint" | "interrupt" |
                       # "done" | "error"
    data: Any          # payload varies by type
    agent: str         # which agent emitted this
    checkpoint_id: str # current checkpoint after this event
    ts: float          # unix timestamp (ms precision)
```

**Streaming consumer:**

```python
async def stream_graph_events(
    graph: CompiledGraph,
    initial_state: AgentState,
    config: RunnableConfig,
) -> AsyncIterator[str]:
    """
    Consumes graph.astream_events() and yields formatted SSE strings.
    
    astream_events() emits:
      - on_chat_model_stream   → token events
      - on_tool_start          → tool_call events
      - on_tool_end            → tool_result events
      - on_chain_start         → agent_switch events (node entry)
      - on_chain_end           → checkpoint events (node exit)
    """
    async for event in graph.astream_events(initial_state, config, version="v2"):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield format_sse(SSEEvent(
                    type="token",
                    data={"content": token},
                    agent=event.get("name", "unknown"),
                    checkpoint_id=event.get("run_id", ""),
                    ts=time.time() * 1000,
                ))

        elif kind == "on_tool_start":
            yield format_sse(SSEEvent(
                type="tool_call",
                data={"tool": event["name"], "input": event["data"].get("input")},
                agent=event.get("tags", [""])[0],
                checkpoint_id=event.get("run_id", ""),
                ts=time.time() * 1000,
            ))

        elif kind == "on_chain_start" and "__interrupt__" in event.get("tags", []):
            yield format_sse(SSEEvent(
                type="interrupt",
                data={"reason": event["data"].get("input", {}).get("hil_reason")},
                agent="Supervisor",
                checkpoint_id=event.get("run_id", ""),
                ts=time.time() * 1000,
            ))

    yield format_sse(SSEEvent(type="done", data={}, agent="Supervisor",
                              checkpoint_id="", ts=time.time() * 1000))
```

---

### Phase B — Multi-Agent Supervisor System

---

#### B.1 — CodeAgent Sub-Graph (F-05)

**File:** `backend/app/graph/agents/code_agent.py`

**What to build:**
A compiled `StateGraph` that performs: query expansion → hybrid retrieval → reranking → context assembly → code-focused LLM generation. This is a complete encapsulation of the existing `RetrieverEngine` + code generation path.

**Node sequence inside CodeAgent:**
```
code_expand_query_node
        │
        ▼
code_retrieve_node        ← EnsembleRetriever (BM25 + ChromaDB with file_type=code)
        │
        ▼
code_rerank_node          ← BGE cross-encoder top-5
        │
        ▼
code_pdr_node             ← Parent Document Retrieval (full function bodies)
        │
        ▼
code_truncate_node        ← Safe truncation (MAX_CHARS_PER_SOURCE = 8000)
        │
        ▼
code_generate_node        ← LLM with code-specific few-shot prompt
        │
        ▼
    (returns to Supervisor synthesizer_node)
```

**Key implementation rules:**
- `code_retrieve_node` must call `RetrieverEngine.retrieve()` inside a `threading.Lock()` (existing fix, preserve it).
- `code_generate_node` reads `state["reranked_chunks"]` and `state["parent_contexts"]` — never reaches into the full corpus directly.
- The sub-graph reads `state["metadata_filter"]` set by the Supervisor's `intent_classifier_node` — it NEVER sets the filter itself.
- Output is written to `state["agent_responses"]["CodeAgent"]` and `state["sources"]`.
- All node names must be prefixed: `code_*` to avoid name collisions with other agents.

**Tool wiring (F-56):**
```python
from langchain_core.tools import tool
from pydantic import BaseModel

class CodeSearchInput(BaseModel):
    query: str
    top_k: int = 5
    file_pattern: Optional[str] = None  # e.g. "*.py"

@tool("code_search", args_schema=CodeSearchInput)
async def code_search_tool(query: str, top_k: int = 5, file_pattern: Optional[str] = None) -> dict:
    """Search the codebase for functions, classes, and implementation details."""
    ...
```
Every tool must have a Pydantic `args_schema` — this enables schema validation before the tool is called, preventing malformed inputs from reaching the retriever.

---

#### B.2 — DocAgent Sub-Graph (F-06)

**File:** `backend/app/graph/agents/doc_agent.py`

**What to build:**
Mirror of CodeAgent but for KT documentation. Key differences:
- `metadata_filter` is `{"file_type": "kt_doc"}`.
- Splitter uses prose-mode (`RecursiveCharacterTextSplitter` without language separator).
- LLM prompt uses documentation-focused few-shot examples (the `KT_DOCUMENTATION` template from `few_shot_prompt.py`).
- `doc_retrieve_node` weights the BM25 retriever higher (0.6 BM25 / 0.4 vector) for exact section lookup.

---

#### B.3 — DebugAgent Sub-Graph (F-07)

**File:** `backend/app/graph/agents/debug_agent.py`

**What to build:**
Specialized agent for error diagnosis queries (`"why is X failing"`, `"what causes this stack trace"`).

**Node sequence:**
```
debug_parse_error_node     ← extract error type, file, line from query
        │
        ▼
debug_retrieve_node        ← code_search filtered to error-adjacent code
        │
        ▼
debug_pattern_node         ← BM25 search of known error patterns
        │
        ▼
debug_dependency_node      ← find callers of the failing function
        │
        ▼
debug_generate_node        ← LLM generates root-cause analysis + fix suggestion
```

**Key design:** `debug_parse_error_node` uses a small extraction LLM call (or regex) to pull structured error info: `{error_type, file_path, line_number, stack_frames}`. This structured info is then passed as extra context to `debug_retrieve_node` to improve retrieval precision.

---

#### B.4 — Synthesizer Node (F-11)

**File:** `backend/app/graph/nodes/synthesizer.py`

**What to build:**
When the Supervisor routes to multiple agents (HYBRID decision), their outputs need to be merged coherently. The synthesizer is a dedicated LLM call that:

1. Reads `state["agent_responses"]` (a dict: `agent_name → partial_answer`).
2. Reads all `state["sources"]` accumulated from each agent.
3. Deduplicates sources by `source_id`.
4. Calls LLM with a synthesis prompt: *"You have received answers from N specialized agents. Produce a single, coherent, non-repetitive answer that cites each source."*
5. Writes the merged answer to `state["final_response"]`.

**When synthesizer is skipped:**
If only one agent ran (single routing decision), the synthesizer simply copies `state["agent_responses"][agent_name]` into `state["final_response"]` without an LLM call (zero token cost).

---

### Phase C — Memory Architecture

---

#### C.1 — Short-Term Memory (F-12, F-13)

**File:** `backend/app/graph/memory/short_term.py`

**What to build:**
The `memory_read_node` that loads the last N conversation turns from `PostgresChatMessageHistory` and injects them as `state["short_term_window"]`.

**Key design rules:**
- Session ID must be the namespaced value `"{user_id}::{session_id}"` (existing security fix, preserve it).
- Window size: configurable via `settings.stm_window_turns` (default 10 turns).
- The node converts `BaseMessage` list into a structured list of dicts for state storage.
- Token budget guard: if the combined window exceeds `settings.stm_max_tokens` (default 4096), the oldest turns are summarized using a lightweight LLM call before being included.

**Summarization fallback:**
```python
async def summarize_old_turns(turns: list[dict], llm) -> str:
    """Summarize old conversation turns into a compressed memory string."""
    prompt = (
        "Summarize the following conversation history in 3-5 sentences, "
        "preserving all technical facts, file names, and code symbols mentioned:\n\n"
        + format_turns(turns)
    )
    response = await llm.ainvoke(prompt)
    return response.content
```

---

#### C.2 — Long-Term Memory Store (F-13 to F-17)

**Files:**
- `backend/app/graph/memory/long_term_store.py`
- `backend/app/graph/memory/entity_extractor.py`

**What to build:**
A vector-backed memory store using the existing pgvector infrastructure. Each memory entry is:

```python
@dataclass
class MemoryEntry:
    user_id: str
    org_id: Optional[str]
    content: str           # "User is working on the pricing engine module"
    entity_type: str       # "user_fact" | "code_fact" | "preference"
    embedding: list[float] # 768d from all-mpnet-base-v2
    source_session: str    # session that produced this memory
    created_at: datetime
    last_accessed: datetime
    access_count: int
    relevance_score: float # set at retrieval time
```

**PostgreSQL schema (new table):**
```sql
CREATE TABLE IF NOT EXISTS agent_long_term_memory (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    org_id      TEXT,
    content     TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'user_fact',
    embedding   VECTOR(768) NOT NULL,
    source_session TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);
CREATE INDEX ltm_user_idx ON agent_long_term_memory (user_id);
CREATE INDEX ltm_embedding_idx ON agent_long_term_memory
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**Entity extractor:**
After each turn, run a small LLM call (or NER pipeline) to extract salient facts from the conversation:
```python
EXTRACT_PROMPT = """
Extract factual statements from this conversation that should be remembered
for future sessions. Focus on: user goals, code modules they are working on,
known bugs, architectural decisions discussed.
Output as JSON list: [{"content": "...", "entity_type": "code_fact|user_fact|preference"}]
"""
```

**Memory retrieval in `memory_read_node`:**
```python
async def retrieve_long_term_facts(user_id: str, query: str, top_k: int = 5) -> list[str]:
    embedding = await get_embedder().aembed_query(query)
    # pgvector cosine search scoped to user_id
    rows = await pg_pool.fetch(
        """SELECT content FROM agent_long_term_memory
           WHERE user_id = $1
           ORDER BY embedding <=> $2::vector
           LIMIT $3""",
        user_id, embedding, top_k
    )
    return [row["content"] for row in rows]
```

---

### Phase D — Checkpointing & Time-Travel

---

#### D.1 — PostgresSaver Integration (F-19, F-20)

**File:** `backend/app/graph/checkpointing/pg_checkpointer.py`

**What to build:**
Configure LangGraph's `PostgresSaver` (or `AsyncPostgresSaver`) to persist the full `AgentState` after every node completes. This is the foundation for time-travel and HIL resume.

**Implementation:**
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def get_checkpointer() -> AsyncPostgresSaver:
    """Returns a shared AsyncPostgresSaver backed by the existing pg_pool."""
    from app.core.database import get_pg_pool
    pool = get_pg_pool()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()  # creates checkpoints table if not exists
    return saver
```

**Thread configuration:**
LangGraph uses `config["configurable"]["thread_id"]` as the checkpoint key. Set:
```python
config = {
    "configurable": {
        "thread_id": f"{user_id}::{session_id}",  # same namespace as memory
        "checkpoint_ns": org_id or "default",
    }
}
```

**What gets stored:**
PostgresSaver serializes the entire `AgentState` as JSONB per node completion. Each row: `{thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at}`.

---

#### D.2 — Time-Travel API (F-21, F-22)

**File:** `backend/app/api/checkpoints.py`

**What to build:**

```
GET  /api/v2/sessions/{session_id}/checkpoints
     → list all checkpoints for a thread, with node_name + created_at + state_summary

GET  /api/v2/sessions/{session_id}/replay/{checkpoint_id}
     → re-execute the graph from the given checkpoint (new thread_id branch)

GET  /api/v2/sessions/{session_id}/state/{checkpoint_id}
     → return the full AgentState at that checkpoint (for debugging)

POST /api/v2/sessions/{session_id}/branch
     body: {from_checkpoint_id, new_query?}
     → fork the conversation from a historical state (new thread)
```

**List checkpoints implementation:**
```python
@router.get("/sessions/{session_id}/checkpoints")
async def list_checkpoints(session_id: str, current_user=Depends(get_current_user)):
    thread_id = f"{current_user.id}::{session_id}"
    checkpointer = await get_checkpointer()
    
    history = []
    async for checkpoint_tuple in checkpointer.alist(
        {"configurable": {"thread_id": thread_id}}
    ):
        history.append({
            "checkpoint_id": checkpoint_tuple.config["configurable"]["checkpoint_id"],
            "parent_id": checkpoint_tuple.parent_config,
            "created_at": checkpoint_tuple.checkpoint.get("ts"),
            "nodes_visited": checkpoint_tuple.metadata.get("nodes_visited", []),
            "query_preview": checkpoint_tuple.checkpoint.get("channel_values", {})
                             .get("query", "")[:100],
        })
    return {"checkpoints": history, "total": len(history)}
```

**Time-travel replay:**
```python
@router.get("/sessions/{session_id}/replay/{checkpoint_id}")
async def replay_from_checkpoint(session_id: str, checkpoint_id: str,
                                  current_user=Depends(get_current_user)):
    """Re-runs the graph from a historical checkpoint as a NEW branch thread."""
    thread_id = f"{current_user.id}::{session_id}"
    branch_thread_id = f"{thread_id}::branch::{checkpoint_id[:8]}"
    
    graph = get_supervisor_graph()
    config = {
        "configurable": {
            "thread_id": branch_thread_id,
            "checkpoint_id": checkpoint_id,  # start from here
        }
    }
    # Stream the replay back as SSE
    return StreamingResponse(
        stream_graph_events(graph, None, config),  # None = resume from checkpoint
        media_type="text/event-stream"
    )
```

---

### Phase E — Human-in-the-Loop

---

#### E.1 — HIL Interrupt Mechanism (F-25, F-26)

**File:** `backend/app/graph/nodes/hil_node.py`

**What to build:**
A node that evaluates whether the current query requires human review before the response is sent. If yes, it updates state with `hil_required=True` and the graph halts at the `interrupt_before=["hil_check_node"]` boundary configured during graph compilation.

**When to interrupt:**
- Routing confidence < 0.5 (agent is unsure which domain applies).
- Query contains keywords suggesting destructive actions: `"delete"`, `"drop table"`, `"remove all"`.
- DebugAgent proposes a code change (fix diff).
- Configurable per-org threshold.

**HIL node logic:**
```python
async def hil_check_node(state: AgentState) -> dict:
    confidence = state.get("routing_confidence", 1.0)
    response_preview = state.get("final_response", "")
    
    needs_hil = (
        confidence < settings.hil_confidence_threshold       # default 0.5
        or contains_destructive_intent(state["query"])
        or state.get("active_agent") == "DebugAgent" and "```diff" in response_preview
    )
    
    if needs_hil:
        reason = derive_hil_reason(state)
        # Returning hil_required=True causes the graph to pause at this node
        # because interrupt_before=["hil_check_node"] was set at compile time
        return {
            "hil_required": True,
            "hil_reason": reason,
            "nodes_visited": state["nodes_visited"] + ["hil_check_node:paused"],
        }
    
    return {
        "hil_required": False,
        "nodes_visited": state["nodes_visited"] + ["hil_check_node:passed"],
    }
```

**Resume endpoint:**
```python
@router.post("/sessions/{session_id}/resume")
async def resume_from_hil(
    session_id: str,
    body: HILResumeRequest,  # {human_input: str, approved: bool}
    current_user=Depends(get_current_user)
):
    """Inject human feedback and resume graph execution."""
    thread_id = f"{current_user.id}::{session_id}"
    graph = get_supervisor_graph()
    
    # Update state with human decision
    await graph.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {
            "hil_human_input": body.human_input,
            "hil_approved": body.approved,
            "hil_required": False,  # clear the interrupt
        },
        as_node="hil_check_node",
    )
    
    # Stream the resumed execution
    return StreamingResponse(
        stream_graph_events(graph, None, {"configurable": {"thread_id": thread_id}}),
        media_type="text/event-stream"
    )
```

---

### Phase F — Middleware & Guardrails

---

#### F.1 — Input Guardrail System (F-30 to F-33)

**File:** `backend/app/graph/guardrails/input_guardrail.py`

**What to build:**
A LangGraph node that runs before any retrieval or agent work. Implements a chain of checks. Each check is a `GuardrailCheck` protocol:

```python
class GuardrailCheck(Protocol):
    name: str
    severity: str  # "block" | "warn" | "scrub"
    async def check(self, query: str, state: AgentState) -> GuardrailResult: ...
```

**Checks to implement:**

**1. Prompt Injection Detector:**
```python
INJECTION_PATTERNS = [
    r"ignore (previous|all) instructions",
    r"you are now",
    r"disregard (your|the) (system|previous) prompt",
    r"act as (a |an )?(different|new|evil|unrestricted)",
    r"DAN|jailbreak|developer mode",
    r"<\|.*?\|>",           # token boundary injection
    r"\[INST\]|\[SYS\]",   # LLaMA instruction injection
]
```
Severity: **block** — return `guardrail_passed=False`, add violation, stop graph.

**2. PII Scrubber:**
Use `presidio-analyzer` + `presidio-anonymizer` to detect and replace:
- Email addresses → `[EMAIL]`
- Phone numbers → `[PHONE]`
- Credit card numbers → `[CC_NUMBER]`
- SSN patterns → `[SSN]`

Store the original (encrypted) in `state["_pii_original"]` for audit. Use `state["pii_scrubbed_query"]` downstream.

Severity: **scrub** — continue with sanitized query, log violation.

**3. Token Budget Check:**
```python
if len(query.split()) > settings.max_query_tokens:  # default 512
    return GuardrailResult(
        passed=False,
        violation="Query exceeds maximum token budget",
        severity="block"
    )
```

**4. Content Safety Filter:**
Use a lightweight classification model (e.g., `unitary/toxic-bert` via HuggingFace) or a rule-based list for toxic/harmful content.

**Guardrail node:**
```python
async def input_guardrail_node(state: AgentState) -> dict:
    violations = []
    query = state["query"]
    
    for check in INPUT_GUARDRAIL_CHAIN:
        result = await check.check(query, state)
        if not result.passed:
            violations.append({"check": check.name, "reason": result.violation})
            if check.severity == "block":
                return {
                    "guardrail_passed": False,
                    "guardrail_violations": violations,
                    "final_response": f"Request blocked: {result.violation}",
                }
            elif check.severity == "scrub":
                query = result.sanitized_query  # use scrubbed version
    
    return {
        "guardrail_passed": True,
        "guardrail_violations": violations,
        "pii_scrubbed_query": query,
        "query": query,  # downstream nodes see the scrubbed version
    }
```

---

#### F.2 — Output Guardrail System (F-34 to F-36)

**File:** `backend/app/graph/guardrails/output_guardrail.py`

**What to build:**
Runs after agents generate responses, before streaming to the client.

**Checks to implement:**

**1. Citation Verifier (Faithfulness Guard):**
Verify that factual claims in the response are grounded in `state["reranked_chunks"]`. Approach:
- Extract claim sentences from response (split on `.`).
- For each claim, compute cosine similarity against the retrieved chunks.
- If any claim has max similarity < 0.3 (no source supports it), flag it as potentially hallucinated.
- Append a `⚠️ Unverified claim` marker (warn severity) or strip it (block severity if configured).

**2. Code Safety Scanner:**
Scan generated code blocks for dangerous patterns:
```python
DANGEROUS_CODE_PATTERNS = [
    r"rm\s+-rf",
    r"os\.system\(",
    r"subprocess\.call\(",
    r"eval\(",
    r"exec\(",
    r"__import__\(",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM\s+\w+\s*;",  # no WHERE clause
]
```
Severity: **block** if found in code blocks (wrap in warning or refuse).

**3. PII Leak Prevention:**
Run the same Presidio scanner on the generated response — catch cases where the LLM echoed back PII from the retrieved chunks.

---

#### F.3 — LangGraph Middleware Hooks (F-37, F-38)

**File:** `backend/app/graph/middleware/node_middleware.py`

**What to build:**
A decorator pattern for adding cross-cutting concerns (logging, retry, timing) to any LangGraph node without modifying the node's core logic.

```python
def with_node_middleware(
    node_fn: NodeFn,
    *,
    node_name: str,
    enable_retry: bool = True,
    max_retries: int = 3,
    retry_on: tuple = (Exception,),
    timeout_seconds: float = 30.0,
    trace: bool = True,
) -> NodeFn:
    """Wraps a LangGraph node with retry, timeout, and OTEL tracing."""
    
    @functools.wraps(node_fn)
    async def wrapped(state: AgentState, config: RunnableConfig) -> dict:
        start = time.perf_counter()
        attempt = 0
        
        while attempt <= max_retries:
            try:
                # OTEL span
                with optional_span(f"langgraph.node.{node_name}") as span:
                    span.set_attribute("node.name", node_name)
                    span.set_attribute("session.id", state.get("session_id", ""))
                    
                    # Timeout wrapper
                    result = await asyncio.wait_for(
                        node_fn(state, config),
                        timeout=timeout_seconds
                    )
                    
                    elapsed = (time.perf_counter() - start) * 1000
                    NODE_LATENCY_HISTOGRAM.record(elapsed, {"node": node_name})
                    return result
                    
            except asyncio.TimeoutError:
                logger.error(f"[{node_name}] Timeout after {timeout_seconds}s")
                raise
            except retry_on as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = min(2 ** attempt, 30)
                logger.warning(f"[{node_name}] Retry {attempt}/{max_retries} after {wait}s: {e}")
                await asyncio.sleep(wait)
    
    return wrapped
```

**Usage at graph build time:**
```python
builder.add_node("code_retrieve_node",
    with_node_middleware(code_retrieve_node, node_name="code_retrieve",
                         max_retries=2, retry_on=(ChromaDBError,)))
```

---

### Phase G — Streaming & API Layer

---

#### G.1 — v2 Chat API (F-41 to F-46)

**File:** `backend/app/api/v2/chat.py`

**What to build:**
A new `APIRouter` at `/api/v2` that replaces `/api/v1/chat/stream`. Keeps v1 running for backward compatibility.

**Request schema:**
```python
class ChatV2Request(BaseModel):
    query: str = Field(..., max_length=2048)
    session_id: str
    user_id: str
    org_id: Optional[str] = None
    stream: bool = True
    hil_enabled: bool = False          # opt-in to HIL interrupts
    hil_confidence_threshold: float = 0.5
    agent_hint: Optional[str] = None   # "CodeAgent" | "DocAgent" — override routing
    resume_from_checkpoint: Optional[str] = None  # for time-travel
```

**Endpoint:**
```python
@router.post("/chat/stream")
async def chat_stream_v2(
    request: ChatV2Request,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    graph: CompiledGraph = Depends(get_supervisor_graph),
):
    # 1. Build initial state
    initial_state = build_initial_state(request, current_user)
    
    # 2. Build LangGraph config with thread_id (checkpoint key)
    config = {
        "configurable": {
            "thread_id": f"{current_user.id}::{request.session_id}",
        },
        "recursion_limit": 25,
    }
    
    # 3. If resuming from checkpoint, skip initial state
    if request.resume_from_checkpoint:
        initial_state = None
        config["configurable"]["checkpoint_id"] = request.resume_from_checkpoint
    
    # 4. Stream
    return StreamingResponse(
        stream_graph_events(graph, initial_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )
```

**Reconnect support (F-46):**
On client reconnect, the frontend sends the `Last-Event-ID` header (last received checkpoint_id). The endpoint resumes streaming from that checkpoint:
```python
last_checkpoint = request.headers.get("Last-Event-ID")
if last_checkpoint:
    config["configurable"]["checkpoint_id"] = last_checkpoint
```

---

### Phase H — Runtime Observability

---

#### H.1 — Per-Node OTEL Spans (F-47)

**File:** `backend/app/observability/langgraph_instrumentation.py`

**What to build:**
Automatic OTEL span injection for every LangGraph node, without modifying each node individually. Uses the `with_node_middleware` wrapper from Phase F, plus a LangGraph callback.

**Metrics to add to `quality_metrics.py`:**

```python
# Per-node latency histogram
NODE_LATENCY_MS = Histogram(
    "langgraph_node_latency_ms",
    "Execution latency per LangGraph node",
    labelnames=["node_name", "agent"],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
)

# Tokens per agent
AGENT_TOKENS = Histogram(
    "langgraph_agent_tokens_total",
    "LLM tokens consumed per agent per turn",
    labelnames=["agent_name", "token_type"],  # token_type: "input" | "output"
    buckets=[100, 500, 1000, 2000, 4000, 8000],
)

# Edges traversed per turn
GRAPH_EDGES_TRAVERSED = Histogram(
    "langgraph_edges_per_turn",
    "Number of graph edges traversed per query",
    buckets=[1, 2, 3, 5, 8, 13, 21],
)

# HIL interrupt rate
HIL_INTERRUPTS = Counter(
    "langgraph_hil_interrupts_total",
    "Total HIL interrupt events",
    labelnames=["reason"],
)

# LTM hit/miss
LTM_LOOKUPS = Counter(
    "langgraph_ltm_lookups_total",
    "Long-term memory lookup events",
    labelnames=["result"],  # "hit" | "miss"
)

# Guardrail events
GUARDRAIL_EVENTS = Counter(
    "langgraph_guardrail_events_total",
    "Guardrail check events",
    labelnames=["check_name", "action"],  # action: "passed" | "blocked" | "scrubbed"
)
```

---

#### H.2 — Grafana Agent Activity Dashboard (F-53)

**File:** `grafana/dashboards/agent-activity-dashboard.json`

**Panels to add:**

| Panel | PromQL | Visualization |
|---|---|---|
| Agent dispatch distribution | `sum by (routing_decision) (rate(langgraph_node_latency_ms_count{node_name="intent_classifier_node"}[5m]))` | Pie chart |
| Per-node p95 latency | `histogram_quantile(0.95, sum by (node_name, le) (rate(langgraph_node_latency_ms_bucket[5m])))` | Bar gauge |
| HIL interrupt rate | `rate(langgraph_hil_interrupts_total[5m])` | Time series |
| LTM hit rate | `rate(langgraph_ltm_lookups_total{result="hit"}[5m]) / rate(langgraph_ltm_lookups_total[5m])` | Stat |
| Guardrail blocks per minute | `rate(langgraph_guardrail_events_total{action="blocked"}[1m])` | Time series |
| Tokens per agent | `histogram_quantile(0.95, sum by (agent_name, le) (rate(langgraph_agent_tokens_total_bucket[5m])))` | Bar chart |
| Graph edges per turn | `histogram_quantile(0.50, sum by (le) (rate(langgraph_edges_per_turn_bucket[5m])))` | Stat |

**New alert rules to add to `alert-rules.yml`:**

```yaml
- alert: AgentDeadlock
  expr: |
    max by (thread_id) (
      time() - langgraph_node_last_entry_timestamp
    ) > 30
  for: 30s
  labels:
    severity: critical
    component: langgraph
  annotations:
    summary: "LangGraph agent appears deadlocked"
    description: "A graph thread has been in the same node for > 30s"

- alert: HILBacklog
  expr: sum(langgraph_hil_pending_count) > 10
  for: 5m
  labels:
    severity: warning
    component: hil
  annotations:
    summary: "HIL interrupt backlog growing"
    description: "More than 10 queries are waiting for human approval"

- alert: GuardrailBlockSurge
  expr: rate(langgraph_guardrail_events_total{action="blocked"}[5m]) > 0.1
  for: 2m
  labels:
    severity: warning
    component: guardrails
  annotations:
    summary: "Elevated guardrail block rate"
    description: "Possible prompt injection attack in progress"
```

---

### Phase I — Frontend Modernization

---

#### I.1 — Agent Event Stream Consumer

**File:** `frontend/src/app/services/agent-stream.service.ts`

**What to build:**
Replace the current raw SSE consumer in `chat.service.ts` with a structured event dispatcher that handles the new v2 SSE envelope types.

**Event handler map:**
```typescript
interface AgentStreamEvent {
  type: 'token' | 'tool_call' | 'tool_result' | 'agent_switch' |
        'checkpoint' | 'interrupt' | 'done' | 'error';
  data: any;
  agent: string;
  checkpoint_id: string;
  ts: number;
}

@Injectable({ providedIn: 'root' })
export class AgentStreamService {
  private eventHandlers = new Map<string, (event: AgentStreamEvent) => void>();

  onToken(handler: (token: string, agent: string) => void): void { ... }
  onToolCall(handler: (tool: string, input: any) => void): void { ... }
  onAgentSwitch(handler: (from: string, to: string) => void): void { ... }
  onInterrupt(handler: (reason: string, checkpointId: string) => void): void { ... }
  onDone(handler: (metadata: any) => void): void { ... }

  resumeFromHIL(sessionId: string, humanInput: string, approved: boolean): Observable<void> { ... }
  replayFromCheckpoint(sessionId: string, checkpointId: string): void { ... }
}
```

---

#### I.2 — Agent Activity Panel

**File:** `frontend/src/app/components/agent-activity.component.ts`

**What to build:**
A real-time panel (sidebar or collapsible drawer) that visualizes the agent graph traversal as the query executes:

```
🔵 Supervisor   → classifying intent...
  └── 🟢 CodeAgent  → retrieving (BM25 + vector)...
       ├── 🔧 code_search [query="authenticate_user"]
       └── ✅ 5 chunks retrieved, reranked to 3
  └── 🔄 Synthesizer  → merging answers...
✅ Response ready  [checkpoint: abc123]
```

**Key behaviors:**
- Each `agent_switch` event adds a new row with animated indicator.
- Each `tool_call` event adds a child row showing the tool name and input.
- Each `checkpoint` event adds a clickable badge (click → load time-travel view).
- An `interrupt` event shows a yellow banner: *"Waiting for your approval..."* with Approve/Reject buttons.
- On `done`, all indicators turn green; sources accordion auto-expands.

---

#### I.3 — Time-Travel UI

**File:** `frontend/src/app/components/checkpoint-timeline.component.ts`

**What to build:**
A timeline view (accessible from chat history) showing all checkpoints for a session. Each checkpoint is a node on the timeline. Clicking a checkpoint:
1. Shows the state summary at that point (query, intent, which agent ran, sources found).
2. Offers a "Replay from here" button → calls `/api/v2/sessions/{id}/replay/{checkpoint_id}`.
3. Offers a "Branch conversation" button → creates a new session from that state.

---

### Phase J — Production Hardening

---

#### J.1 — Parallel Tool Dispatch (F-57)

**File:** `backend/app/graph/agents/code_agent.py`

In CodeAgent, when multiple tools need to run (e.g., `code_search` + `symbol_lookup`), dispatch them in parallel using LangGraph's `Send` API or `asyncio.gather`:

```python
async def code_parallel_tools_node(state: AgentState) -> dict:
    """Dispatch multiple code tools in parallel."""
    tasks = [
        code_search_tool.ainvoke({"query": state["query"], "top_k": 10}),
        symbol_lookup_tool.ainvoke({"symbol": extract_symbol(state["query"])}),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    tool_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Tool {i} failed: {result}")
        else:
            tool_results.append(result)
    
    return {"tool_results": tool_results}
```

---

#### J.2 — Load Testing Configuration

**File:** `backend/scripts/load_test_langgraph.py`

Using `locust` or `k6`, define a load test that:
- Simulates 50 concurrent users.
- Mix: 60% CodeAgent queries, 25% DocAgent, 10% DebugAgent, 5% HIL triggers.
- Validates: p95 TTFB < 2s, p99 total < 10s, zero `CancelledError` leaks, all checkpoints persisted.
- Reports per-agent latency breakdown.

---

## 6. Git Commit Convention

All commits follow **Conventional Commits** spec with a project-specific scope taxonomy:

```
<type>(<scope>): <short imperative summary>

[optional body: what changed and why, not how]

[optional footer: BREAKING CHANGE, Closes #issue, Refs #feature]
```

**Type taxonomy:**

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `refactor` | Restructure existing code without behavior change |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `test` | Add or update tests |
| `docs` | Documentation only |
| `chore` | Build system, deps, tooling |
| `obs` | Observability (metrics, tracing, alerting) |
| `sec` | Security improvement |
| `infra` | Docker, CI, Kubernetes configs |

**Scope taxonomy:**

| Scope | Component |
|---|---|
| `graph` | LangGraph graph definition and compilation |
| `state` | AgentState definition |
| `supervisor` | SupervisorAgent |
| `code-agent` | CodeAgent sub-graph |
| `doc-agent` | DocAgent sub-graph |
| `debug-agent` | DebugAgent sub-graph |
| `arch-agent` | ArchitectureAgent sub-graph |
| `web-agent` | WebSearchAgent sub-graph |
| `memory` | Long-term and short-term memory |
| `checkpoint` | Checkpointing, PostgresSaver |
| `time-travel` | Time-travel replay API |
| `hil` | Human-in-the-Loop |
| `guardrails` | Input/output guardrail nodes |
| `streaming` | SSE streaming layer |
| `api` | FastAPI route handlers |
| `frontend` | Angular components and services |
| `obs` | Prometheus metrics, OTEL, Grafana |
| `infra` | Docker, deps |
| `migration` | DB schema migrations |

---

## 7. Git Commit Log by Phase

### Phase A — LangGraph Foundation

```
feat(state): define AgentState TypedDict with all graph channel fields

Introduces the central state schema for the LangGraph supervisor graph.
Covers: messages (reducer=operator.add), routing, retrieval, memory,
guardrails, HIL, streaming, and observability fields.
All downstream phases read from and write to this contract.

Refs: F-01
```

```
feat(graph): scaffold SupervisorGraph StateGraph with all node stubs

Adds backend/app/graph/supervisor_graph.py with full node and edge
declarations. All nodes are stubs returning empty dicts. Wires conditional
edges for cache_hit routing, agent dispatch, and HIL branching.
Compiles graph with PostgresSaver checkpointer (setup() deferred to startup).

Refs: F-02, F-04
```

```
feat(graph): implement intent_classifier_node replacing AgenticRouter

Ports the keyword-based RoutingDecision logic from agentic_router.py into
a pure LangGraph node. Adds routing_decision and metadata_filter to state.
Preserves the routing_decision_to_metadata_filter() translation.
Marks agentic_router.py as deprecated (removed in Phase B cleanup).

Refs: F-03
```

```
feat(streaming): add astream_events consumer with structured SSE envelope

Adds backend/app/graph/streaming.py. Consumes graph.astream_events(version="v2")
and maps on_chat_model_stream / on_tool_start / on_chain_start events to typed
SSEEvent dataclasses serialized as JSON-over-SSE.

Token events carry agent name. Interrupt events carry hil_reason.
Done event carries checkpoint_id for reconnect support.

Refs: F-39, F-40
```

```
chore(infra): add langgraph, langgraph-checkpoint-postgres to requirements

Pins: langgraph>=0.2.0, langgraph-checkpoint-postgres>=1.0.0,
presidio-analyzer>=2.2.0, presidio-anonymizer>=2.2.0.
Keeps all existing pins. Updates docker-compose.phase5.yml to expose
postgres checkpoint table via healthcheck.

Refs: Phase A deps
```

---

### Phase B — Multi-Agent System

```
feat(code-agent): implement CodeAgent sub-graph with hybrid retrieval

Adds backend/app/graph/agents/code_agent.py.
Nodes: code_expand_query → code_retrieve → code_rerank → code_pdr →
       code_truncate → code_generate.

Preserves threading.Lock() on RetrieverEngine metadata_filter mutation.
Wires code_search_tool with Pydantic args_schema (CodeSearchInput).
Sub-graph compiled and registered in supervisor_graph.py as node "CodeAgent".

Closes: F-05, F-56 (code_search tool schema)
```

```
feat(doc-agent): implement DocAgent sub-graph for KT document retrieval

Adds backend/app/graph/agents/doc_agent.py.
Uses file_type="kt_doc" metadata filter from Supervisor state.
BM25 weight=0.6, vector weight=0.4 for exact section retrieval.
Prompt uses KT_DOCUMENTATION few-shot template from few_shot_prompt.py.

Closes: F-06
```

```
feat(debug-agent): implement DebugAgent sub-graph with error parsing

Adds backend/app/graph/agents/debug_agent.py.
debug_parse_error_node extracts {error_type, file_path, line_number}
from query using regex + small LLM extraction call.
debug_dependency_node performs reverse-lookup of callers via BM25.
Output includes root-cause analysis and optional fix suggestion.

Closes: F-07
```

```
feat(arch-agent): implement ArchitectureAgent sub-graph

Adds backend/app/graph/agents/arch_agent.py.
Hybrid metadata_filter: OR of file_type=code and file_type=kt_doc.
Prompt instructs LLM to synthesize data-flow narrative from both sources.
Registered in Supervisor with routing decision ARCHITECTURE.

Closes: F-08
```

```
feat(web-agent): add WebSearchAgent using Tavily integration

Adds backend/app/graph/agents/web_agent.py.
Wraps existing TAVILY_API_KEY env var. Falls back gracefully if key absent.
Used for queries classified as EXTERNAL_REFERENCE or CVE_LOOKUP.
Results appended to state["tool_results"] for synthesizer.

Closes: F-09
```

```
feat(supervisor): wire all sub-graphs into Supervisor with conditional routing

Updates supervisor_graph.py to replace node stubs with compiled sub-graphs.
Adds HYBRID routing path: Supervisor dispatches to >1 agent in sequence
(parallel dispatch deferred to Phase J).
Registers all agent-to-synthesizer edges.

Closes: F-10
```

```
feat(graph): implement synthesizer_node for multi-agent result fusion

Adds backend/app/graph/nodes/synthesizer.py.
Single-agent path: copies agent_responses[agent] to final_response (no LLM call).
Multi-agent path: deduplicate sources by source_id, call LLM with synthesis prompt.
Writes final_response and deduplicated sources to state.

Closes: F-11
```

```
refactor(agents): deprecate agent_brain.py monolith, add compat shim

Marks AgentBrain.process_query() as deprecated.
Adds a thin shim that delegates to the new Supervisor graph for v1 API callers.
Existing /api/v1/chat/stream continues to work during migration window.
Adds MIGRATION_NOTE comment block at top of agent_brain.py.
```

---

### Phase C — Memory Architecture

```
feat(memory): implement short-term memory window with token budget guard

Adds backend/app/graph/memory/short_term.py.
memory_read_node loads last N turns from PostgresChatMessageHistory
using namespaced session_id (user_id::session_id security fix preserved).
If combined window > stm_max_tokens (4096), oldest turns are summarized
via lightweight LLM call before injection into state.

Closes: F-12
```

```
feat(migration): add agent_long_term_memory table with ivfflat index

Adds Prisma migration: 20240617_add_ltm_table.
Creates agent_long_term_memory with VECTOR(768) embedding column,
user_id B-tree index, and ivfflat cosine index (lists=100).
Table is multi-tenant: user_id + org_id scoped — matches semantic_cache pattern.

Closes: F-13 (schema)
```

```
feat(memory): implement LongTermStore with pgvector cosine retrieval

Adds backend/app/graph/memory/long_term_store.py.
retrieve() performs: embed query → pgvector <=> cosine search WHERE user_id=$1.
store() upserts MemoryEntry with embedding via shared pg_pool.
Namespace isolation: user_id + optional org_id — never cross-user leakage.

Closes: F-13, F-17
```

```
feat(memory): implement entity extractor for post-turn fact persistence

Adds backend/app/graph/memory/entity_extractor.py.
After each turn, extract_facts() calls LLM with EXTRACT_PROMPT to produce
[{content, entity_type}] list. memory_write_node batches writes via pg_pool.
Facts with entity_type="preference" get higher cosine threshold for retrieval.

Closes: F-14, F-15, F-16
```

---

### Phase D — Checkpointing & Time-Travel

```
feat(checkpoint): integrate AsyncPostgresSaver as LangGraph checkpointer

Adds backend/app/graph/checkpointing/pg_checkpointer.py.
get_checkpointer() returns singleton AsyncPostgresSaver backed by shared pg_pool.
setup() called at app lifespan startup (creates checkpoints table if absent).
Supervisor graph compiled with checkpointer=await get_checkpointer().

Closes: F-19, F-20
```

```
feat(time-travel): add checkpoint list and state inspection endpoints

Adds backend/app/api/checkpoints.py.
GET /api/v2/sessions/{session_id}/checkpoints — lists all checkpoint tuples
    for a thread, returning checkpoint_id, nodes_visited, query_preview.
GET /api/v2/sessions/{session_id}/state/{checkpoint_id} — returns full
    AgentState snapshot for developer inspection.
Endpoint requires JWT auth; thread_id namespaced to current_user.id.

Closes: F-21
```

```
feat(time-travel): implement graph replay from historical checkpoint

Adds GET /api/v2/sessions/{session_id}/replay/{checkpoint_id}.
Creates branch_thread_id = original_thread::branch::{checkpoint_id[:8]}.
Resumes graph execution from checkpoint state via astream_events.
Streams replay events as SSE with type="replay_token" for client distinction.

Closes: F-22
```

```
feat(time-travel): add conversation branch endpoint

Adds POST /api/v2/sessions/{session_id}/branch.
Body: {from_checkpoint_id, new_query?}.
Forks AgentState at checkpoint; optionally injects a new query before resuming.
Returns new session_id for the branch. Branch listed in checkpoint tree.

Closes: F-23
```

---

### Phase E — Human-in-the-Loop

```
feat(hil): implement hil_check_node with confidence and intent triggers

Adds backend/app/graph/nodes/hil_node.py.
Triggers HIL when routing_confidence < settings.hil_confidence_threshold (0.5)
OR query contains destructive-intent keywords OR DebugAgent proposes a diff.
Returns hil_required=True; graph halts at interrupt_before=["hil_check_node"].
All interrupts logged to audit_log table with reason + state summary.

Closes: F-25, F-29
```

```
feat(hil): add resume endpoint for human-in-the-loop decisions

Adds POST /api/v2/sessions/{session_id}/resume.
aupdate_state() injects {hil_human_input, hil_approved, hil_required=False}
at the hil_check_node position. Graph resumes from next node.
If hil_approved=False, graph short-circuits to response_node with
rejection message. Resumes stream as SSE.

Closes: F-26
```

```
obs(hil): emit HIL interrupt events in SSE stream and Prometheus

Updates stream_graph_events() to detect __interrupt__ tag on chain_start events.
Emits SSEEvent(type="interrupt", data={reason, awaiting_input=True}).
Increments langgraph_hil_interrupts_total counter with reason label.
Frontend receives event and renders approval UI without polling.

Closes: F-44 (interrupt events), F-50 (HIL metric)
```

---

### Phase F — Middleware & Guardrails

```
sec(guardrails): implement input guardrail node with injection detection

Adds backend/app/graph/guardrails/input_guardrail.py.
PromptInjectionDetector: 12 regex patterns covering ignore-instructions,
DAN, token-boundary, and LLaMA instruction injection.
TokenBudgetCheck: blocks queries > 512 tokens.
GuardrailChain: ordered checks, first block returns immediately.
All violations logged with session_id (no query_text — cardinality safety).

Closes: F-31, F-33
```

```
sec(guardrails): add PII scrubber using Presidio analyzer/anonymizer

Updates input_guardrail.py with PIIScrubber check.
Uses presidio_analyzer.AnalyzerEngine + presidio_anonymizer.AnonymizerEngine.
Detects: EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS.
Scrubbed query stored in state["pii_scrubbed_query"]; original AES-encrypted
in state["_pii_original"] for audit (never logged or streamed).

Closes: F-30
```

```
sec(guardrails): implement output guardrail with citation verifier

Adds backend/app/graph/guardrails/output_guardrail.py.
CitationVerifier: extracts claim sentences, computes cosine similarity
against reranked_chunks. Claims with max_sim < 0.3 receive ⚠️ annotation.
CodeSafetyScanner: 8 regex patterns for dangerous shell/SQL/Python patterns.
PII leak scan: re-runs Presidio on final_response before streaming.

Closes: F-34, F-35, F-36
```

```
feat(graph): add node middleware with retry and OTEL tracing decorator

Adds backend/app/graph/middleware/node_middleware.py.
with_node_middleware() wraps any async node fn with:
  - asyncio.wait_for timeout
  - exponential backoff retry (configurable exception types)
  - OTEL span creation (langgraph.node.{name})
  - NODE_LATENCY_HISTOGRAM recording
Applied to all retrieval and LLM nodes in CodeAgent, DocAgent, DebugAgent.

Closes: F-37, F-38
```

```
obs(guardrails): add Prometheus counters for all guardrail check results

Updates quality_metrics.py with GUARDRAIL_EVENTS Counter.
Labels: check_name (injection_detector, pii_scrubber, etc.) x action (passed/blocked/scrubbed).
Updates alert-rules.yml with GuardrailBlockSurge alert.
Updates Grafana RAG dashboard with "Guardrail Events" panel.

Closes: partial F-48, alert F-54
```

---

### Phase G — Streaming & API Layer

```
feat(api): add /api/v2/chat/stream endpoint with LangGraph astream_events

Adds backend/app/api/v2/chat.py with ChatV2Request schema.
Routes to Supervisor graph via stream_graph_events().
Supports resume_from_checkpoint param (Last-Event-ID reconnect).
Keeps /api/v1/chat/stream fully operational via compat shim.
Adds OpenAPI schema for all v2 SSE event types.

Closes: F-41, F-46
```

```
feat(streaming): add tool_call, tool_result, agent_switch SSE event types

Updates stream_graph_events() to emit:
  - tool_call events on on_tool_start (tool name + sanitized input)
  - tool_result events on on_tool_end (tool name + result summary, not full content)
  - agent_switch events on on_chain_start for named sub-graphs
SSE envelope includes checkpoint_id on every event for client state sync.

Closes: F-42, F-43, F-45
```

---

### Phase H — Runtime Observability

```
obs(metrics): declare per-node and per-agent Prometheus instruments

Updates backend/app/observability/quality_metrics.py with:
NODE_LATENCY_MS Histogram (labels: node_name, agent).
AGENT_TOKENS Histogram (labels: agent_name, token_type).
GRAPH_EDGES_TRAVERSED Histogram.
HIL_INTERRUPTS Counter (label: reason).
LTM_LOOKUPS Counter (label: result).
All declared at module scope — zero per-request instrument creation.

Closes: F-47, F-48, F-49, F-50, F-51
```

```
obs(grafana): add Agent Activity dashboard with 7 panels

Adds grafana/dashboards/agent-activity-dashboard.json.
Panels: agent dispatch pie, per-node p95 latency bar, HIL interrupt rate,
LTM hit rate stat, guardrail blocks time series, tokens per agent bar,
graph edges per turn stat.
Datasource: prometheus (matches existing provisioning config).

Closes: F-53
```

```
obs(alerts): add AgentDeadlock, HILBacklog, GuardrailBlockSurge alert rules

Updates alert-rules.yml under new group langgraph_alerts.
AgentDeadlock: node last_entry > 30s — severity critical.
HILBacklog: pending HIL count > 10 — severity warning.
GuardrailBlockSurge: block rate > 0.1/s for 2m — severity warning.
All alerts include runbook-style description fields.

Closes: F-54, F-55
```

---

### Phase I — Frontend Modernization

```
feat(frontend): add AgentStreamService for typed v2 SSE event dispatch

Adds frontend/src/app/services/agent-stream.service.ts.
Replaces raw fetchEventSource usage in chat.service.ts with typed dispatcher.
Handles: token, tool_call, agent_switch, interrupt, checkpoint, done, error.
Reconnect: sends Last-Event-ID header with last received checkpoint_id.
onInterrupt() emits observable for HIL approval UI.

Closes: I.1
```

```
feat(frontend): add AgentActivityPanel component for real-time graph visualization

Adds frontend/src/app/components/agent-activity.component.ts/.html/.scss.
Renders live agent traversal tree using ngFor over activity events array.
Animated spinner per in-progress node; green check on completion.
Tool call rows shown as children with tool name and input preview.
Checkpoint badges clickable — emits event to open time-travel sidebar.

Closes: I.2
```

```
feat(frontend): add HIL approval UI banner integrated with interrupt events

Updates chat.component.ts and chat.component.html.
On interrupt event: renders yellow approval banner with hil_reason text.
Approve button → calls AgentStreamService.resumeFromHIL(approved=true).
Reject button → calls AgentStreamService.resumeFromHIL(approved=false).
Banner dismisses on receipt of next token event (graph resumed).

Closes: E.2 (frontend side)
```

```
feat(frontend): add CheckpointTimeline component for time-travel UI

Adds frontend/src/app/components/checkpoint-timeline.component.ts/.html/.scss.
Fetches GET /api/v2/sessions/{id}/checkpoints on panel open.
Renders vertical timeline; each node shows checkpoint_id prefix,
nodes_visited list, and query_preview.
"Replay from here" button streams replay events into current chat view.
"Branch" button opens new chat tab with branch session_id.

Closes: I.3
```

---

### Phase J — Production Hardening

```
perf(code-agent): dispatch multiple tools in parallel using asyncio.gather

Updates code_agent.py code_parallel_tools_node.
code_search_tool and symbol_lookup_tool invoked concurrently.
Individual tool failures caught as exceptions; partial results used.
Reduces p95 CodeAgent latency by ~35% on symbol-heavy queries.

Closes: F-57
```

```
test(graph): add integration test suite for full Supervisor graph execution

Adds backend/tests/graph/test_supervisor_graph.py.
Tests: intent classification routing, CodeAgent full path, cache hit/miss,
HIL interrupt and resume, time-travel branch, guardrail block.
Uses in-memory MemorySaver checkpointer (no Postgres required in CI).
All tests run < 60s on GitHub Actions (no LLM calls — mocked).

Closes: M1 test gate
```

```
test(guardrails): add red-team prompt injection test suite

Adds backend/tests/security/test_prompt_injection.py.
100 injection samples from OWASP LLM Top-10 and community datasets.
All samples must be blocked by PromptInjectionDetector.
PII scrubber tested against 50 synthetic PII samples (email, phone, SSN).
CI runs on every PR targeting main.

Closes: M3 security gate
```

```
perf(graph): load test 50 concurrent users, validate p95 TTFB < 2s

Adds backend/scripts/load_test_langgraph.py (locust-based).
Mix: 60% CodeAgent, 25% DocAgent, 10% DebugAgent, 5% HIL.
Assertions: p95 TTFB < 2000ms, p99 total < 10000ms,
zero asyncio.CancelledError leaks, 100% checkpoints persisted.
Results exported to load_test_results.json and checked into CI artifacts.

Closes: M5 load test gate
```

```
docs: update README with LangGraph architecture diagram and v2 API reference

Updates README.md:
- Replaces Phase Breakdown table with new 10-phase LangGraph table.
- Adds Mermaid supervisor graph diagram.
- Adds v2 SSE event type reference table.
- Adds time-travel and HIL usage examples.
- Updates installation section with new pip deps.
- Keeps v1 API docs for migration window.
```

```
chore(infra): update docker-compose.phase5.yml for LangGraph services

Adds Redis service (required for optional HIL backlog queue).
Adds checkpoint table healthcheck to postgres service.
Adds LANGGRAPH_CHECKPOINT_DB env var to backend service.
Bumps backend service memory limit to 6GB (reranker + embedding + LTM).
```

---

## 8. Testing Strategy & Startup Validation

### Architecture

Every phase of the modernization has a **mandatory** test suite that runs automatically when the server starts. The system is structured as follows:

```
backend/app/tests/
├── __init__.py            ← package marker + description
├── base.py                ← PhaseTest, TestResult, PhaseReport, TestStatus types
├── runner.py              ← StartupTestRunner — discovers, runs, renders report
├── phase_a/
│   ├── __init__.py        ← PHASE_ID="A", PHASE_NAME="LangGraph Foundation"
│   └── tests.py           ← TESTS: list[PhaseTest]  (10 tests)
├── phase_b/
│   ├── __init__.py
│   └── tests.py           ← 12 tests
├── phase_c/
│   ├── __init__.py
│   └── tests.py           ← 10 tests
├── phase_d/
│   ├── __init__.py
│   └── tests.py           ← 9 tests
├── phase_e/
│   ├── __init__.py
│   └── tests.py           ← 9 tests
├── phase_f/
│   ├── __init__.py
│   └── tests.py           ← 14 tests
├── phase_g/
│   ├── __init__.py
│   └── tests.py           ← 9 tests
└── phase_h/
    ├── __init__.py
    └── tests.py           ← 12 tests
```

### How It Works

1. **Server startup** calls `await StartupTestRunner.run_all()` in `app/main.py`'s `lifespan()` after the RAG pipeline is pre-warmed.
2. `StartupTestRunner` uses `pkgutil.iter_modules` to auto-discover all `phase_*` packages in alphabetical order.
3. Each `phase_*/__init__.py` exports `PHASE_ID`, `PHASE_NAME`, and `TESTS`.
4. Every `PhaseTest` is an `async` function that returns a `TestResult` (`.passed()`, `.failed()`, `.skipped()`, `.error()`).
5. Results are rendered to the terminal in a structured, colour-coded table.

### Sample Log Output

```
════════════════════════════════════════════════════════════════════════
  CodeLens AI — Startup Phase Test Report
════════════════════════════════════════════════════════════════════════

Phase A  —  LangGraph Foundation  [PHASE PASSED]  8/10 passed  412ms
────────────────────────────────────────────────────────────────────────
  ✓  PASSED   A-001  AgentState TypedDict importable           (12ms)
  ✓  PASSED   A-002  messages field uses operator.add reducer  (3ms)
  ✗  FAILED   A-003  Supervisor graph module importable        (1ms)  [CRITICAL]
       ↳ Cannot import app.graph.supervisor_graph: No module named 'app.graph'
  ─  SKIPPED  A-004  Supervisor graph compiles (MemorySaver)   (0ms)
  ✓  PASSED   A-008  langgraph >= 0.2.0 installed              (5ms)
  ...

════════════════════════════════════════════════════════════════════════
  ❌  SOME PHASES FAILED — 62 passed  3 failed  5 skipped  (1842ms total)
════════════════════════════════════════════════════════════════════════
```

### Test Categories Per Phase

| Phase | Test Count | Critical Tests | What Is Validated |
|---|---|---|---|
| A | 10 | A-001, A-004, A-008 | State schema, graph compilation, LangGraph version |
| B | 12 | B-001, B-002, B-011 | All 5 agent sub-graphs, synthesizer, compat shim |
| C | 10 | C-001, C-003, C-006, C-010 | STM/LTM modules, namespace isolation, DB schema |
| D | 9  | D-001, D-003, D-004, D-008 | Checkpointer, thread ID format, API endpoints |
| E | 9  | E-001, E-002, E-003, E-009 | HIL triggers, destructive intent detection |
| F | 14 | F-001–F-004, F-007–F-009 | All guardrail checks, code safety, middleware retry |
| G | 9  | G-001, G-002, G-006, G-009 | v2 endpoints, SSE format, done event, v1 compat |
| H | 12 | H-002, H-008 | 6 Prometheus metrics, alert rules, Grafana dashboard |

### Success Criteria Per Phase

| Phase | Gate | Definition |
|---|---|---|
| A | M1-pre | No critical tests fail; langgraph importable |
| B | M1 | All 5 agents compile; Supervisor has all nodes; compat shim intact |
| C | M2-pre | STM session namespace correct; LTM user-scoped |
| D | M2 | Checkpointer attached; thread_id namespaced; all 4 API endpoints present |
| E | M3-pre | HIL triggers correctly on all 3 conditions |
| F | M3 | All injection patterns blocked; safe queries pass; code scanner works |
| G | M4-pre | v2 endpoint live; SSE always ends with done; v1 untouched |
| H | M4 | All 6 metrics declared; label names correct; all 3 alerts in YAML |

### Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `STARTUP_TESTS_ENABLED` | `true` | Set to `false` to skip all tests in production |
| `STARTUP_TESTS_FAIL_FAST` | `false` | Set to `true` to abort on first critical failure |

### Adding Tests for a New Phase

When implementing a new feature, create `backend/app/tests/phase_X/` with:

```python
# phase_x/__init__.py
PHASE_ID   = "X"
PHASE_NAME = "My New Phase"
from app.tests.phase_x.tests import TESTS

# phase_x/tests.py
from app.tests.base import PhaseTest, TestResult

async def _test_something() -> TestResult:
    try:
        # assert your feature works
        return TestResult.passed("feature works ✓")
    except AssertionError as e:
        return TestResult.failed(str(e))

TESTS = [
    PhaseTest(
        id="X-001",
        name="Something works",
        description="Validates that something works",
        run=_test_something,
        critical=True,
        tags=["my-feature"],
    )
]
```

The runner discovers it automatically on next server start — no registration needed.

---

## 9. Dependency Additions

Add to `backend/requirements.txt`:

```
# ==================== LangGraph ====================
langgraph>=0.2.0
langgraph-checkpoint-postgres>=1.0.0   # AsyncPostgresSaver

# ==================== Guardrails ====================
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
# Optional: faster NLP pipeline for Presidio
spacy>=3.7.0
# python -m spacy download en_core_web_lg

# ==================== Load Testing ====================
locust>=2.20.0  # dev dependency only

# ==================== LangSmith (optional) ====================
# langsmith>=0.1.0
```

Add to `frontend/package.json`:
```json
"@microsoft/fetch-event-source": "^2.0.1"
```

---

## 10. Migration Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `agent_brain.py` callers break when monolith removed | High | High | Keep compat shim through M2; remove at M3 |
| `RetrieverEngine` threading.Lock lost in sub-graph refactor | Medium | High | Explicit test in test_supervisor_graph.py for concurrent retrieval |
| PostgresSaver serialization of large AgentState (>1MB) | Medium | Medium | Cap `retrieved_chunks` at 20 items in state; store full results in transient cache |
| LangGraph version instability (API changes in 0.2.x) | Low | High | Pin exact version; add changelog monitoring |
| PII scrubber Presidio cold start (+3s) | High | Medium | Pre-warm Presidio engine at lifespan startup |
| HIL resume creating duplicate response_node execution | Medium | High | `as_node=` param on aupdate_state ensures correct graph position |
| pgvector IVFFlat index needs VACUUM after LTM bulk writes | Low | Medium | Add maintenance cron for `VACUUM ANALYZE agent_long_term_memory` |
| Frontend SSE reconnect sending wrong checkpoint_id | Medium | Medium | Validate checkpoint_id format server-side; 400 on invalid |

---

*This document is the single source of truth for the CodeLens AI → LangGraph modernization.*  
*Each phase is independently mergeable. Branch from `main` per phase: `feat/phase-a-langgraph-foundation`, etc.*
