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
    "tool_call",
    "tool_result",
    "agent_switch",
    "checkpoint",
    "interrupt",
    "done",
    "error",
})


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


# ── Graph Event Stream Consumer ───────────────────────────────────────────────

async def stream_graph_events(
    graph: Any,
    initial_state: Optional[dict],
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
        The starting AgentState dict.  Pass ``None`` when resuming from
        a previously saved checkpoint (the graph reads state from the
        checkpointer in that case).
    config:
        LangGraph ``RunnableConfig`` dict, must contain
        ``config["configurable"]["thread_id"]``.

    Yields
    ------
    str
        SSE-formatted strings ready to be streamed directly to the client.
        Always ends with a ``done`` event.
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

        async for event in event_stream:
            kind: str = event.get("event", "")
            run_id: str = str(event.get("run_id", ""))
            name: str = event.get("name", "unknown")
            tags: list = event.get("tags", [])

            # ── Token events ─────────────────────────────────────────────────
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk is not None:
                    token = (
                        chunk.content
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    if token:
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

    except Exception as exc:  # noqa: BLE001
        logger.error("[STREAMING] Fatal error: %s", exc, exc_info=True)
        yield format_sse(SSEEvent(
            type="error",
            data={"message": str(exc)},
            agent="Supervisor",
            checkpoint_id="",
            ts=time.time() * 1000,
        ))

    finally:
        # Always emit the done event so the client knows the stream has ended
        yield format_sse(SSEEvent(
            type="done",
            data={},
            agent="Supervisor",
            checkpoint_id="",
            ts=time.time() * 1000,
        ))
