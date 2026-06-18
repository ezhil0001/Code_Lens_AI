"""
Phase G Test Cases — Streaming & API Layer
===========================================
  G-001  v2 chat router importable
  G-002  POST /api/v2/chat/stream endpoint registered
  G-003  ChatV2Request schema validates correctly
  G-004  ChatV2Request rejects query > 2048 chars
  G-005  SSE event types are an exhaustive enum/set
  G-006  stream_graph_events yields 'done' event
  G-007  format_sse produces valid SSE line format
  G-008  Last-Event-ID reconnect param wired in config
  G-009  v1 /api/v1/chat/stream still registered (backward compat)
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

from app.tests.base import PhaseTest, TestResult


def _try_import(path: str):
    try:
        return importlib.import_module(path), None
    except Exception as exc:
        return None, str(exc)


async def _test_v2_chat_router_importable() -> TestResult:
    mod, err = _try_import("app.api.v2.chat")
    if err:
        return TestResult.failed(f"Cannot import app.api.v2.chat: {err}")
    if not hasattr(mod, "router"):
        return TestResult.failed("router not found in app.api.v2.chat")
    return TestResult.passed("app.api.v2.chat.router found ✓")


async def _test_v2_stream_endpoint_registered() -> TestResult:
    mod, err = _try_import("app.api.v2.chat")
    if err:
        return TestResult.skipped("app.api.v2.chat not importable")
    routes = [str(r.path) for r in getattr(mod.router, "routes", [])]
    matching = [r for r in routes if "stream" in r]
    if not matching:
        return TestResult.failed(f"No /stream route found. Routes: {routes}")
    return TestResult.passed(f"v2 stream endpoint: {matching[0]} ✓")


async def _test_chat_v2_request_valid() -> TestResult:
    mod, err = _try_import("app.api.v2.chat")
    if err:
        return TestResult.skipped("app.api.v2.chat not importable")
    if not hasattr(mod, "ChatV2Request"):
        return TestResult.failed("ChatV2Request not found in app.api.v2.chat")
    try:
        req = mod.ChatV2Request(
            query="how does auth work?",
            session_id="sess-001",
            user_id="user-001",
        )
        assert req.stream is True       # default
        assert req.hil_enabled is False # default
        return TestResult.passed("ChatV2Request validates with defaults ✓")
    except Exception as exc:
        return TestResult.error(exc)


async def _test_chat_v2_request_rejects_long_query() -> TestResult:
    mod, err = _try_import("app.api.v2.chat")
    if err:
        return TestResult.skipped("app.api.v2.chat not importable")
    if not hasattr(mod, "ChatV2Request"):
        return TestResult.skipped("ChatV2Request not found")
    try:
        from pydantic import ValidationError  # type: ignore
        mod.ChatV2Request(
            query="x" * 3000,   # > 2048 limit
            session_id="s",
            user_id="u",
        )
        return TestResult.failed("ChatV2Request should reject query > 2048 chars")
    except Exception:
        return TestResult.passed("ChatV2Request correctly rejects query > 2048 chars ✓")


async def _test_sse_event_types_exhaustive() -> TestResult:
    mod, err = _try_import("app.graph.streaming")
    if err:
        return TestResult.skipped("streaming not importable")
    if not hasattr(mod, "SSE_EVENT_TYPES"):
        return TestResult.failed(
            "SSE_EVENT_TYPES constant not found in streaming.py",
            detail=(
                "Add: SSE_EVENT_TYPES = frozenset({"
                "'token','tool_call','tool_result','agent_switch',"
                "'checkpoint','interrupt','done','error'})"
            )
        )
    required = frozenset({
        "token", "tool_call", "tool_result", "agent_switch",
        "checkpoint", "interrupt", "done", "error",
    })
    declared = frozenset(mod.SSE_EVENT_TYPES)
    missing = required - declared
    if missing:
        return TestResult.failed(f"SSE_EVENT_TYPES missing: {missing}")
    return TestResult.passed(f"All {len(required)} SSE event types declared ✓")


async def _test_stream_graph_events_yields_done() -> TestResult:
    """stream_graph_events must always yield a 'done' event at the end."""
    mod, err = _try_import("app.graph.streaming")
    if err:
        return TestResult.skipped("streaming not importable")
    if not hasattr(mod, "stream_graph_events"):
        return TestResult.skipped("stream_graph_events not found")

    # Build a trivial graph that finishes immediately
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
        from typing import TypedDict

        class _S(TypedDict):
            value: int

        b = StateGraph(_S)
        b.add_node("n", lambda s, c: {"value": s["value"] + 1})
        b.set_entry_point("n")
        b.add_edge("n", END)
        g = b.compile(checkpointer=MemorySaver())

        initial = {"value": 0}
        config = {"configurable": {"thread_id": "test-g-007"}}

        events = []
        async for chunk in mod.stream_graph_events(g, initial, config):
            events.append(chunk)

        done_events = [e for e in events if '"type": "done"' in e or '"type":"done"' in e]
        if not done_events:
            return TestResult.failed(
                f"No 'done' event yielded. Got {len(events)} events: {events[-3:]}"
            )
        return TestResult.passed("stream_graph_events yields 'done' event ✓")
    except ImportError:
        return TestResult.skipped("langgraph not installed")
    except Exception as exc:
        return TestResult.error(exc)


async def _test_format_sse_output() -> TestResult:
    mod, err = _try_import("app.graph.streaming")
    if err:
        return TestResult.skipped("streaming not importable")
    if not hasattr(mod, "format_sse"):
        return TestResult.failed("format_sse() not found in streaming.py")
    if not hasattr(mod, "SSEEvent"):
        return TestResult.skipped("SSEEvent not found")

    event = mod.SSEEvent(type="token", data={"content": "hello"},
                         agent="CodeAgent", checkpoint_id="chk-1", ts=0.0)
    output: str = mod.format_sse(event)
    if not output.startswith("data:"):
        return TestResult.failed(
            f"format_sse output must start with 'data:'. Got: {output[:60]!r}"
        )
    if not output.endswith("\n\n"):
        return TestResult.failed(
            "format_sse output must end with double newline (SSE spec)"
        )
    return TestResult.passed("format_sse produces valid SSE line ✓")


async def _test_reconnect_param_wired() -> TestResult:
    mod, err = _try_import("app.api.v2.chat")
    if err:
        return TestResult.skipped("app.api.v2.chat not importable")
    import inspect
    if not hasattr(mod, "chat_stream_v2"):
        return TestResult.failed("chat_stream_v2 endpoint function not found")
    src = inspect.getsource(mod.chat_stream_v2)
    if "Last-Event-ID" not in src and "resume_from_checkpoint" not in src:
        return TestResult.failed(
            "chat_stream_v2 does not handle Last-Event-ID / resume_from_checkpoint"
        )
    return TestResult.passed("Reconnect / Last-Event-ID handling present ✓")


async def _test_v1_endpoint_still_registered() -> TestResult:
    from app.api import chat as v1_chat
    routes = [str(r.path) for r in getattr(v1_chat.router, "routes", [])]
    matching = [r for r in routes if "stream" in r]
    if not matching:
        return TestResult.failed(
            f"v1 /stream route missing — backward compat broken. Routes: {routes}"
        )
    return TestResult.passed(f"v1 stream endpoint still registered: {matching[0]} ✓")


TESTS: list[PhaseTest] = [
    PhaseTest(id="G-001", name="v2 chat router importable",
              description="app.api.v2.chat.router found",
              run=_test_v2_chat_router_importable, critical=True, tags=["api", "streaming"]),
    PhaseTest(id="G-002", name="POST /api/v2/chat/stream registered",
              description="Stream endpoint route exists in v2 router",
              run=_test_v2_stream_endpoint_registered, critical=True, tags=["api"]),
    PhaseTest(id="G-003", name="ChatV2Request validates with defaults",
              description="stream=True, hil_enabled=False by default",
              run=_test_chat_v2_request_valid, critical=False, tags=["api"]),
    PhaseTest(id="G-004", name="ChatV2Request rejects query > 2048 chars",
              description="Pydantic max_length=2048 enforced",
              run=_test_chat_v2_request_rejects_long_query, critical=False, tags=["api"]),
    PhaseTest(id="G-005", name="SSE_EVENT_TYPES constant exhaustive",
              description="All 8 event types declared in streaming.py",
              run=_test_sse_event_types_exhaustive, critical=False, tags=["streaming"]),
    PhaseTest(id="G-006", name="stream_graph_events yields 'done' event",
              description="Every stream ends with a done event",
              run=_test_stream_graph_events_yields_done, critical=True, tags=["streaming"]),
    PhaseTest(id="G-007", name="format_sse produces valid SSE line format",
              description="Output starts with 'data:' and ends with double newline",
              run=_test_format_sse_output, critical=False, tags=["streaming"]),
    PhaseTest(id="G-008", name="Reconnect / Last-Event-ID wired in v2 endpoint",
              description="chat_stream_v2 handles Last-Event-ID or resume_from_checkpoint",
              run=_test_reconnect_param_wired, critical=False, tags=["streaming", "api"]),
    PhaseTest(id="G-009", name="v1 /chat/stream still registered",
              description="Backward compat: v1 stream endpoint not removed",
              run=_test_v1_endpoint_still_registered, critical=True, tags=["api", "compat"]),
]
