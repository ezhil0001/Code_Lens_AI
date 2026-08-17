"""
SSE streaming layer — consumes graph.astream_events() and emits typed
Server-Sent Event envelopes to the browser.

Each SSE message carries a JSON envelope so the Angular client can handle
different event kinds without string-matching token content:

  token        — one LLM output chunk; content field holds the text fragment
  tool_call    — a tool was invoked; data includes tool name and sanitized input
  tool_result  — tool returned; data includes a 200-char result preview
  agent_switch — the supervisor handed off to a specialist sub-graph
  checkpoint   — a node finished and its state was persisted
  interrupt    — the graph paused for human review (HIL)
  done         — graph execution completed; client can finalize the message
  error        — fatal error; client shows the error message and stops

The `_streamed_run_ids` set inside stream_graph_events() prevents
on_chain_end from emitting final_response as a duplicate token event when
the LLM already streamed individual chunks.  Without this guard, long
responses would appear twice — once token-by-token and again as a bulk
emission when the chain closes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Optional

import logging

logger = logging.getLogger(__name__)


# ── SSE Event Types ───────────────────────────────────────────────────────────

SSE_EVENT_TYPES: frozenset[str] = frozenset({
    "token",
    "token_reset",
    "tool_call",
    "tool_result",
    "agent_switch",
    "checkpoint",
    "interrupt",
    "done",
    "error",
})

# Nodes whose LLM output IS the answer shown to the user. Every other LLM call
# in the graph is internal machinery — intent routing emits routing JSON and
# memory writing emits extracted facts — and must never reach the client.
ANSWER_NODES: frozenset[str] = frozenset({
    "code_generate_node",
    "doc_generate_node",
    "debug_generate_node",
    "arch_generate_node",
    "synthesizer_node",
})

# On a multi-agent query every agent streams its own full answer and THEN the
# synthesiser streams the merged answer. Appending both showed the user the
# same facts two or three times over. The synthesiser's output supersedes the
# drafts, so a token_reset is emitted the moment it produces its first token.
SUPERSEDING_NODE: str = "synthesizer_node"


# ── SSEEvent Dataclass ────────────────────────────────────────────────────────

@dataclass
class SSEEvent:
    """Structured SSE envelope sent over the wire as JSON-in-SSE."""

    type: str           # must be one of SSE_EVENT_TYPES
    data: Any           # payload — varies by type
    agent: str          # which agent/node emitted this event
    checkpoint_id: str  # LangGraph run_id (proxy for checkpoint id)
    ts: float           # unix timestamp in milliseconds


# ── Formatting ────────────────────────────────────────────────────────────────

def format_sse(event: SSEEvent) -> str:
    """
    Serialise an SSEEvent to the SSE wire format.

    Output:   "data: {json}\\n\\n"

    The double newline is mandatory per the EventSource spec — it signals
    the end of a single event to the client's EventSource parser.
    """
    payload = asdict(event)
    return f"data: {json.dumps(payload)}\n\n"


_MAX_SOURCES = 8
_MAX_SNIPPET_CHARS = 400


def _collect_sources(snapshot: Any) -> list:
    """Extract client-safe citation sources from the final graph state.

    The agents already sanitise ``file_path`` (see ``sanitise_source_path``), so
    no server-side absolute path is exposed. Snippets are truncated and the
    list is capped so the terminal event stays small. Never raises.
    """
    try:
        values = getattr(snapshot, "values", None) or {}
        raw = values.get("sources") or []
        out = []
        seen = set()
        for s in raw:
            if not isinstance(s, dict):
                continue
            path = str(s.get("file_path") or s.get("source") or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            out.append({
                "file_path": path,
                "score": round(float(s.get("score") or 0.0), 4),
                "snippet": str(s.get("content") or "")[:_MAX_SNIPPET_CHARS],
            })
            if len(out) >= _MAX_SOURCES:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("[streaming] source collection failed: %s", exc)
        return []


# ── Graph Event Stream Consumer ───────────────────────────────────────────────

async def stream_graph_events(
    graph: Any,
    initial_state: Optional[Any],
    config: dict,
) -> AsyncIterator[str]:
    """
    Consume ``graph.astream_events()`` and yield formatted SSE strings.

    Mapping from LangGraph event kinds to SSE event types:
      on_chat_model_stream  → token
      on_tool_start         → tool_call
      on_tool_end           → tool_result
      on_chain_start        → agent_switch  (for named sub-graph nodes)
                            → interrupt     (when __interrupt__ tag present)
      on_chain_end          → checkpoint    (node completion)

    Parameters
    ----------
    graph:
        A compiled LangGraph ``CompiledGraph`` (or compatible duck-type).
    initial_state:
        The starting AgentState dict, a ``Command(resume=...)`` when answering a
        HIL interrupt, or ``None`` to continue from the saved checkpoint
        (the graph reads state from the checkpointer in that case).
    config:
        LangGraph ``RunnableConfig`` dict, must contain
        ``config["configurable"]["thread_id"]``.

    Yields
    ------
    str
        SSE-formatted strings ready to be streamed directly to the client.
        On normal completion or a caught error, ends with a ``done`` event.
        On client disconnect (GeneratorExit) it terminates silently without
        emitting further events, per async-generator semantics.
    """
    try:
        # Resolve astream_events kwargs
        stream_kwargs: dict[str, Any] = {"version": "v2", "config": config}
        if initial_state is not None:
            stream_input = initial_state
        else:
            # Resuming from checkpoint — pass an empty command / None
            stream_input = None

        # Collect astream_events arguments based on whether we have initial state
        if stream_input is not None:
            event_stream = graph.astream_events(stream_input, **stream_kwargs)
        else:
            # Resume mode: no initial input — graph reads from checkpoint
            event_stream = graph.astream_events(None, **stream_kwargs)

        # Track run_ids of nodes that already streamed at least one LLM token.
        # Used to prevent on_chain_end from emitting a duplicate final_response.
        _streamed_run_ids: set[str] = set()
        # Simpler guard: set to True the moment any on_chat_model_stream fires.
        # If True, the response_node fallback is skipped (tokens already delivered).
        _any_tokens_streamed: bool = False
        # True once synthesiser output has superseded the per-agent drafts.
        _synthesis_started: bool = False

        async for event in event_stream:
            kind: str = event.get("event", "")
            run_id: str = str(event.get("run_id", ""))
            name: str = event.get("name", "unknown")
            tags: list = event.get("tags", [])

            # ── Token events ─────────────────────────────────────────────────
            if kind == "on_chat_model_stream":
                node = (event.get("metadata") or {}).get("langgraph_node", "")
                if node not in ANSWER_NODES:
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk is not None:
                    token = (
                        chunk.content
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    if token:
                        if node == SUPERSEDING_NODE and not _synthesis_started:
                            _synthesis_started = True
                            yield format_sse(SSEEvent(
                                type="token_reset",
                                data={"reason": "synthesis supersedes agent drafts"},
                                agent=name,
                                checkpoint_id=run_id,
                                ts=time.time() * 1000,
                            ))
                        _any_tokens_streamed = True
                        for parent_id in event.get("parent_ids", []):
                            _streamed_run_ids.add(parent_id)
                        _streamed_run_ids.add(run_id)
                        yield format_sse(SSEEvent(
                            type="token",
                            data={"content": token},
                            agent=name,
                            checkpoint_id=run_id,
                            ts=time.time() * 1000,
                        ))

            # ── Tool call started ─────────────────────────────────────────────
            elif kind == "on_tool_start":
                tool_input = event.get("data", {}).get("input")
                yield format_sse(SSEEvent(
                    type="tool_call",
                    data={"tool": name, "input": tool_input},
                    agent=tags[0] if tags else "unknown",
                    checkpoint_id=run_id,
                    ts=time.time() * 1000,
                ))

            # ── Tool call completed ───────────────────────────────────────────
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output")
                # Summarise large tool outputs to avoid bloating SSE stream
                summary = str(output)[:200] if output is not None else ""
                yield format_sse(SSEEvent(
                    type="tool_result",
                    data={"tool": name, "result_preview": summary},
                    agent=tags[0] if tags else "unknown",
                    checkpoint_id=run_id,
                    ts=time.time() * 1000,
                ))

            # ── Chain start — agent switch or interrupt ───────────────────────
            elif kind == "on_chain_start":
                if "__interrupt__" in tags:
                    hil_reason = (
                        event.get("data", {})
                        .get("input", {})
                        .get("hil_reason", "Human review required")
                    )
                    yield format_sse(SSEEvent(
                        type="interrupt",
                        data={"reason": hil_reason, "awaiting_input": True},
                        agent="Supervisor",
                        checkpoint_id=run_id,
                        ts=time.time() * 1000,
                    ))
                else:
                    # Named sub-graph entry → agent_switch event
                    if name and name not in ("LangGraph", "unknown", ""):
                        yield format_sse(SSEEvent(
                            type="agent_switch",
                            data={"to": name},
                            agent=name,
                            checkpoint_id=run_id,
                            ts=time.time() * 1000,
                        ))

            # ── Chain end — node checkpoint ───────────────────────────────────
            elif kind == "on_chain_end":
                # Emit final_response as a bulk token event when the LLM did
                # NOT stream individual chunks (Groq with streaming disabled,
                # or when on_chat_model_stream events are not propagated from
                # sub-graphs through astream_events).
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    final_response = output.get("final_response")
                    node_name = name.lower()
                    is_response_node = any(
                        kw in node_name
                        for kw in ("synthesizer", "response", "generate", "code_generate", "doc_generate")
                    )
                    # Only emit the fallback when no token events have been seen yet.
                    if (
                        is_response_node
                        and final_response
                        and isinstance(final_response, str)
                        and not _any_tokens_streamed
                    ):
                        _any_tokens_streamed = True   # mark so we don't emit twice
                        yield format_sse(SSEEvent(
                            type="token",
                            data={"content": final_response},
                            agent=name,
                            checkpoint_id=run_id,
                            ts=time.time() * 1000,
                        ))

                yield format_sse(SSEEvent(
                    type="checkpoint",
                    data={"node": name},
                    agent=name,
                    checkpoint_id=run_id,
                    ts=time.time() * 1000,
                ))

        # LangGraph's interrupt() does NOT surface through astream_events — no
        # event carries a "__interrupt__" tag, name or output key. The pause is
        # only observable on the checkpoint: snapshot.next names the paused node
        # and snapshot.tasks[].interrupts carries the payload. Without this the
        # graph paused correctly but the browser never learned about it, so the
        # HIL review UI could never appear.
        interrupted = False
        snapshot = None
        try:
            # State must be read from the ROOT namespace. The request config
            # carries checkpoint_ns=<org_id|"default">, but LangGraph treats
            # checkpoint_ns as a SUBGRAPH name, so aget_state() raised
            # "Subgraph default not found" and the interrupt was never seen.
            state_config = dict(config or {})
            configurable = dict(state_config.get("configurable") or {})
            configurable["checkpoint_ns"] = ""
            configurable.pop("checkpoint_id", None)
            state_config["configurable"] = configurable

            snapshot = await graph.aget_state(state_config)
            for task in (getattr(snapshot, "tasks", None) or []):
                for intr in (getattr(task, "interrupts", None) or []):
                    payload = getattr(intr, "value", None) or {}
                    if not isinstance(payload, dict):
                        payload = {"reason": str(payload)}
                    interrupted = True
                    yield format_sse(SSEEvent(
                        type="interrupt",
                        data={
                            "reason": payload.get("reason", "Human review required"),
                            "query": payload.get("query"),
                            "awaiting_input": True,
                            "node": getattr(task, "name", None),
                        },
                        agent=getattr(task, "name", "Supervisor"),
                        checkpoint_id=str(
                            (getattr(snapshot, "config", {}) or {})
                            .get("configurable", {})
                            .get("checkpoint_id", "")
                        ),
                        ts=time.time() * 1000,
                    ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[streaming] interrupt detection failed: %s", exc)

        # Normal completion → emit exactly one terminal ``done`` event.
        yield format_sse(SSEEvent(
            type="done",
            data={"interrupted": interrupted, "sources": _collect_sources(snapshot)},
            agent="Supervisor",
            checkpoint_id="",
            ts=time.time() * 1000,
        ))

    except GeneratorExit:
        # Client disconnected / browser refresh / network drop → the consumer
        # called ``aclose()`` and threw GeneratorExit into us at the suspended
        # ``yield``. Python async-generator semantics forbid yielding during
        # shutdown (doing so raises "async generator ignored GeneratorExit"),
        # so we MUST NOT emit any SSE event here. Terminate silently.
        raise

    except Exception as exc:  # noqa: BLE001
        logger.error("[STREAMING] Fatal error: %s", exc, exc_info=True)
        yield format_sse(SSEEvent(
            type="error",
            data={"message": str(exc)},
            agent="Supervisor",
            checkpoint_id="",
            ts=time.time() * 1000,
        ))
        # Still emit a terminal ``done`` so a connected client can finalize.
        yield format_sse(SSEEvent(
            type="done",
            data={},
            agent="Supervisor",
            checkpoint_id="",
            ts=time.time() * 1000,
        ))
