"""Centralized debug logger for full-flow pipeline tracing.

Goal
----
Give the operator a *terminal-readable* "play-by-play" of every step the RAG
pipeline takes, with:

    🔵 INFO    – general flow ("entering chat_stream", "ingest done")
    🟡 DEBUG   – method calls, data snippets, internal state
    🟢 SUCCESS – positive outcomes (cache hit, indexed, parsed OK)
    🔴 ERROR   – exceptions, fallbacks

Every log line carries:
    * a wall-clock timestamp (ms precision)
    * the **session_id** of the request (or ``-`` for system events)
    * a square-bracket **[TAG]** identifying the pipeline stage
    * elapsed time when emitted from a ``timed()`` block

This module is the *single source of truth* for log formatting. It also
re-routes the stdlib ``logging`` module into loguru so existing
``logging.getLogger(__name__)`` call-sites in the codebase (LangChain,
ChromaDB, FastAPI, our own services) get the same colourised output without
any code changes.

Public API
~~~~~~~~~~
    from app.core.logger import logger, bind_session, timed, tagged

    bind_session("sess-123")
    logger.bind(tag="[CACHE_CHECK]").info("Looking up semantic cache")

    with timed("[RETRIEVER_START]"):
        results = retriever.retrieve(query)
    # → DEBUG  [RETRIEVER_START] enter
    # → INFO   [RETRIEVER_START] done in 142.3ms

    @tagged("[EMBEDDING]")
    async def embed_chunks(chunks): ...
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, Optional

from loguru import logger as _loguru_logger

# ---------------------------------------------------------------------------
# Session context — propagated across async boundaries via contextvars.
# ---------------------------------------------------------------------------
_session_id_var: ContextVar[str] = ContextVar("session_id", default="-")
_tag_var: ContextVar[str] = ContextVar("tag", default="-")


def bind_session(session_id: Optional[str]) -> None:
    """Bind the current session_id to the active execution context.

    Call this at the *entry point* of any request handler (HTTP route,
    background task, CLI command). The id will appear in every subsequent
    log line emitted from the same async task / thread.
    """
    _session_id_var.set(session_id or "-")


def get_session_id() -> str:
    return _session_id_var.get()


# ---------------------------------------------------------------------------
# Loguru configuration — single colourised stderr sink.
# ---------------------------------------------------------------------------
def _patch_record(record: dict) -> None:
    """Inject the session_id contextvar into every record's ``extra`` dict."""
    record["extra"].setdefault("session_id", _session_id_var.get())
    record["extra"].setdefault("tag", _tag_var.get())


_loguru_logger.remove()  # drop loguru's default handler
_loguru_logger.configure(patcher=_patch_record)

# Custom SUCCESS level (between INFO=20 and WARNING=30) → green.
try:
    _loguru_logger.level("SUCCESS", no=25, color="<green><bold>", icon="🟢")
except (TypeError, ValueError):
    # Already registered (hot-reload / re-import) — ignore.
    pass

_LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{extra[session_id]: <16}</cyan> "
    "<magenta>{extra[tag]: <22}</magenta> "
    "<level>{message}</level>"
)

_loguru_logger.add(
    sys.stderr,
    format=_LOG_FORMAT,
    level="DEBUG",
    colorize=True,
    backtrace=False,
    diagnose=False,
    enqueue=False,  # synchronous; avoids losing logs on ungraceful exit
)

# Public alias.
logger = _loguru_logger


# ---------------------------------------------------------------------------
# Stdlib ``logging`` → loguru bridge.
#
# LangChain, FastAPI, ChromaDB, RAGAS, OpenTelemetry, and most of our own
# files use ``logging.getLogger(__name__)``. Without this bridge those logs
# would print with Python's default formatter (no colour, no session_id).
# We install an InterceptHandler at the root logger so *everything* funnels
# through the loguru sink above and gets the consistent format.
# ---------------------------------------------------------------------------
class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        # Map stdlib level → loguru level name.
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the original caller frame so file:line in logs is correct.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _install_stdlib_bridge() -> None:
    """Replace stdlib logging handlers with the loguru intercept."""
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)
    # Tame chatty third-party loggers.
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_install_stdlib_bridge()


# ---------------------------------------------------------------------------
# timed() — context manager that prints DEBUG enter / INFO exit with elapsed.
# ---------------------------------------------------------------------------
@contextmanager
def timed(tag: str, *, level: str = "INFO") -> Iterator[dict]:
    """Trace a code block: log entry, log exit-with-elapsed, propagate exceptions.

    Usage::

        with timed("[RERANKER]") as ctx:
            scores = reranker.score(pairs)
            ctx["count"] = len(scores)        # extra metadata in exit log

    On success an ``[INFO]`` line is emitted: ``"<tag> done in 142.3ms (count=8)"``.
    On exception an ``[ERROR]`` line is emitted with elapsed and the exception
    is re-raised unchanged.
    """
    token = _tag_var.set(tag)
    extras: dict[str, Any] = {}
    start = time.perf_counter()
    bound = _loguru_logger.bind(tag=tag)
    bound.debug(f"{tag} enter")
    try:
        yield extras
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        bound.error(f"{tag} FAILED after {elapsed_ms:.1f}ms ({type(exc).__name__}: {exc})")
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        suffix = ""
        if extras:
            suffix = " (" + ", ".join(f"{k}={v}" for k, v in extras.items()) + ")"
        bound.log(level, f"{tag} done in {elapsed_ms:.1f}ms{suffix}")
    finally:
        _tag_var.reset(token)


# ---------------------------------------------------------------------------
# tagged() — decorator that wraps a function in timed("[TAG]").
# Works for both sync and async callables.
# ---------------------------------------------------------------------------
def tagged(tag: str, *, level: str = "INFO") -> Callable[[Callable], Callable]:
    """Decorator: log entry+exit+elapsed for any (a)sync function."""

    def _decorate(fn: Callable) -> Callable:
        import asyncio

        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def _async_wrapper(*args, **kwargs):
                with timed(tag, level=level):
                    return await fn(*args, **kwargs)

            return _async_wrapper

        @wraps(fn)
        def _sync_wrapper(*args, **kwargs):
            with timed(tag, level=level):
                return fn(*args, **kwargs)

        return _sync_wrapper

    return _decorate


# ---------------------------------------------------------------------------
# Convenience helpers — encourage consistent tag usage.
# ---------------------------------------------------------------------------
def log_step(tag: str, message: str, *, level: str = "INFO") -> None:
    """One-shot tagged log line. Equivalent to ``logger.bind(tag=tag).log(...)``."""
    _loguru_logger.bind(tag=tag).log(level, message)


def log_success(tag: str, message: str) -> None:
    _loguru_logger.bind(tag=tag).log("SUCCESS", message)


def log_error(tag: str, message: str, *, exc_info: bool = False) -> None:
    bound = _loguru_logger.bind(tag=tag)
    if exc_info:
        bound.opt(exception=True).error(message)
    else:
        bound.error(message)


__all__ = [
    "logger",
    "bind_session",
    "get_session_id",
    "timed",
    "tagged",
    "log_step",
    "log_success",
    "log_error",
]
