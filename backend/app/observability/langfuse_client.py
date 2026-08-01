"""Langfuse client — process-wide singleton for LLM observability.

Langfuse is the primary LLM observability and evaluation platform for
CodeLens_AI. This module owns:

1. A lazily-initialised, thread-safe ``Langfuse`` SDK singleton.
2. A factory for per-request LangChain ``CallbackHandler`` instances that
   auto-trace every LangGraph node, LLM call, tool, retrieval, token count,
   cost, and latency with full parent-child span hierarchy.
3. Helpers to attach trace-level metadata (user_id, session_id, tags), to mint
   a deterministic trace id up front, and to push evaluation scores back onto a
   trace.

SDK compatibility
-----------------
Targets the Langfuse OTEL-based tracing API (v3/v4, tested on 4.14.x). The
LangChain ``CallbackHandler`` reads reserved ``langfuse_*`` metadata keys off
the run config to set trace-level session/user/tags/name, and accepts a
``trace_context={"trace_id": ...}`` to pin the root run to a known id.

Design guarantees
------------------
- **Graceful degradation.** If Langfuse is disabled, the SDK is not installed,
  or credentials are missing, every function here becomes a safe no-op. The
  application must never fail because observability is unavailable.
- **Thread-safe singleton.** One SDK client per process (double-checked lock).
- **Async-compatible.** The SDK batches and flushes on a background thread, so
  callback emission never blocks the event loop.
- **No secrets in code.** All credentials come from ``Settings`` / env vars.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    from langfuse import Langfuse as LangfuseType

# ── Optional dependency: import guarded so the app runs without langfuse ──────
try:
    from langfuse import Langfuse  # type: ignore

    HAS_LANGFUSE = True
except Exception:  # noqa: BLE001  (ImportError or partial-install errors)
    Langfuse = None  # type: ignore
    HAS_LANGFUSE = False


# ─────────────────────────────────────────────────────────────────────────────
# Singleton state
# ─────────────────────────────────────────────────────────────────────────────

_client: "Optional[LangfuseType]" = None
_client_lock = threading.Lock()
_initialised = False
_enabled = False
_flush_timeout = 5
_sample_rate = 1.0


def _load_settings():
    """Load app settings without hard-failing if config import breaks."""
    try:
        from app.core.config import get_settings

        return get_settings()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langfuse] settings unavailable: %s", exc)
        return None


# ── PII / secret masking (M-1) ────────────────────────────────────────────────
# Applied by the SDK to every span/generation input+output BEFORE ingestion
# (official `mask` hook). Redacts credentials and direct identifiers while
# keeping payloads debuggable.
import re as _re

_MASK_PATTERNS: "list[tuple[_re.Pattern, str]]" = [
    # JWTs (three base64url segments)
    (_re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "[REDACTED_JWT]"),
    # API keys / bearer tokens (common vendor prefixes)
    (_re.compile(r"\b(sk|pk|rk|gsk|ghp|gho|xox[bposa])[-_][A-Za-z0-9_-]{10,}\b"), "[REDACTED_KEY]"),
    (_re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}"), r"\1 [REDACTED_TOKEN]"),
    # Explicit secret assignments: password=..., api_key: "...", secret=...
    (_re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key)\b(\s*[:=]\s*)(\"[^\"]+\"|'[^']+'|\S+)"), r"\1\2[REDACTED]"),
    # Email addresses
    (_re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Phone numbers (international-ish, 9+ digits with separators)
    (_re.compile(r"(?<![\w./-])\+?\d[\d\s().-]{8,}\d(?![\w.-])"), "[REDACTED_PHONE]"),
]


def _mask_text(text: str) -> str:
    for pattern, repl in _MASK_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def mask_sensitive_data(*, data: Any, **_: Any) -> Any:
    """Official Langfuse ``mask`` hook — redact PII/secrets recursively.

    Handles strings, dicts, lists/tuples; leaves other types untouched.
    Fails open on the SAFE side: masking errors return a redaction marker,
    never the raw payload. Signature matches the SDK contract
    (keyword ``data``).
    """
    try:
        if isinstance(data, str):
            return _mask_text(data)
        if isinstance(data, dict):
            return {k: mask_sensitive_data(data=v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            masked = [mask_sensitive_data(data=v) for v in data]
            return type(data)(masked) if isinstance(data, tuple) else masked
        return data
    except Exception:  # noqa: BLE001
        return "[MASKING_ERROR]"


def _resolve_config() -> Dict[str, Any]:
    """Merge Settings + env into the config Langfuse needs. Env wins if set."""
    settings = _load_settings()

    def _get(attr: str, env: str, default: Any = None) -> Any:
        val = os.getenv(env)
        if val is not None and val != "":
            return val
        return getattr(settings, attr, default) if settings else default

    enabled_raw = _get("langfuse_enabled", "LANGFUSE_ENABLED", False)
    enabled = str(enabled_raw).lower() in ("1", "true", "yes", "on")

    return {
        "enabled": enabled,
        "host": _get("langfuse_host", "LANGFUSE_HOST", "http://localhost:3000"),
        "public_key": _get("langfuse_public_key", "LANGFUSE_PUBLIC_KEY", None),
        "secret_key": _get("langfuse_secret_key", "LANGFUSE_SECRET_KEY", None),
        "release": _get("langfuse_release", "LANGFUSE_RELEASE", None),
        "environment": _get("langfuse_environment", "LANGFUSE_ENVIRONMENT", None)
        or (getattr(settings, "environment", "development") if settings else "development"),
        "sample_rate": float(_get("langfuse_sample_rate", "LANGFUSE_SAMPLE_RATE", 1.0) or 1.0),
        "debug": str(_get("langfuse_debug", "LANGFUSE_DEBUG", False)).lower()
        in ("1", "true", "yes", "on"),
        "flush_timeout": int(_get("langfuse_flush_timeout_seconds", "LANGFUSE_FLUSH_TIMEOUT_SECONDS", 5) or 5),
    }


def init_langfuse() -> bool:
    """Initialise the Langfuse singleton. Idempotent and never raises.

    Returns
    -------
    bool
        ``True`` if Langfuse is live and ready to receive traces, else
        ``False`` (disabled / not installed / missing credentials).
    """
    global _client, _initialised, _enabled, _flush_timeout, _sample_rate

    if _initialised:
        return _enabled

    with _client_lock:
        if _initialised:
            return _enabled
        _initialised = True

        cfg = _resolve_config()
        _flush_timeout = cfg["flush_timeout"]
        _sample_rate = cfg["sample_rate"]

        if not cfg["enabled"]:
            logger.info("[langfuse] disabled (LANGFUSE_ENABLED is false/unset) — observability no-op")
            _enabled = False
            return False

        if not HAS_LANGFUSE:
            logger.warning(
                "[langfuse] LANGFUSE_ENABLED=true but the 'langfuse' package is not "
                "installed. Run: pip install langfuse. Continuing without tracing."
            )
            _enabled = False
            return False

        if not cfg["public_key"] or not cfg["secret_key"]:
            logger.warning(
                "[langfuse] enabled but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are "
                "missing. Create a project in the Langfuse UI and copy its keys. "
                "Continuing without tracing."
            )
            _enabled = False
            return False

        try:
            # Sampling is enforced application-side via should_sample() so that
            # tracing and evaluation scoring stay perfectly consistent (M-4).
            # The SDK therefore always ingests whatever we choose to send.
            _client = Langfuse(
                public_key=cfg["public_key"],
                secret_key=cfg["secret_key"],
                host=cfg["host"],
                release=cfg["release"],
                environment=cfg["environment"],
                sample_rate=1.0,
                debug=cfg["debug"],
                mask=mask_sensitive_data,  # M-1: PII/secret redaction hook
            )
            # Verify connectivity without crashing startup on failure.
            try:
                if hasattr(_client, "auth_check") and not _client.auth_check():
                    logger.warning("[langfuse] auth_check failed — verify keys/host. Tracing may not work.")
            except Exception as auth_exc:  # noqa: BLE001
                logger.debug("[langfuse] auth_check skipped: %s", auth_exc)

            _enabled = True
            logger.info(
                "✓ Langfuse observability initialised — host=%s environment=%s sample_rate=%.2f",
                cfg["host"], cfg["environment"], cfg["sample_rate"],
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[langfuse] initialisation failed: %s — continuing without tracing", exc)
            _client = None
            _enabled = False
            return False


def is_enabled() -> bool:
    """Return True only when Langfuse is live and accepting traces."""
    if not _initialised:
        init_langfuse()
    return _enabled


def should_sample() -> bool:
    """Decide whether to trace + score THIS request, honoring ``sample_rate``.

    Returns True when Langfuse is live and a per-request head-sampling draw
    passes. Callers use this to keep tracing and evaluation scoring consistent
    (M-4): when a request is not sampled, no trace id is minted and no scores
    are published, so we never emit orphan scores for un-ingested traces.
    Never raises.
    """
    if not is_enabled():
        return False
    rate = _sample_rate
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    try:
        import random
        return random.random() < rate
    except Exception:  # noqa: BLE001
        return True


def get_client() -> "Optional[LangfuseType]":
    """Return the Langfuse SDK singleton, or ``None`` when disabled."""
    if not _initialised:
        init_langfuse()
    return _client


def create_trace_id(seed: Optional[str] = None) -> Optional[str]:
    """Mint a Langfuse trace id up front for deterministic propagation.

    Returns a 32-char hex trace id (optionally derived from ``seed`` so the
    same seed always maps to the same id), or ``None`` when Langfuse is
    disabled. Callers pass this to :func:`get_callback_handler` (as
    ``trace_id``) and store it in graph state so evaluation can score the exact
    originating trace without relying on ambient OTEL context. Never raises.
    """
    client = get_client()
    if client is None:
        return None
    try:
        factory = getattr(client, "create_trace_id", None)
        if factory is None:
            return None
        return factory(seed=seed) if seed is not None else factory()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langfuse] create_trace_id failed: %s", exc)
        return None


def get_callback_handler(
    *,
    trace_id: Optional[str] = None,
) -> Optional[Any]:
    """Build a per-request LangChain ``CallbackHandler`` for the graph.

    Attach the returned handler to a LangGraph ``RunnableConfig["callbacks"]``
    so every node, LLM call, tool, and retrieval is traced automatically with
    a correct parent-child hierarchy, token usage, latency, and cost.

    ``trace_id`` pins the root run to a known trace id (via the SDK's
    ``trace_context``) so evaluation scoring can target the exact same trace.
    Trace-level session/user/tags/name are bound separately through
    :func:`build_trace_metadata` merged into ``config["metadata"]``.

    Returns ``None`` when Langfuse is disabled so callers can simply do::

        cb = get_callback_handler(trace_id=tid)
        callbacks = [cb] if cb else []

    Never raises.
    """
    if not is_enabled():
        return None

    try:
        # OTEL-based SDK (v3/v4) exposes the handler at langfuse.langchain;
        # fall back to the legacy path for older installs.
        try:
            from langfuse.langchain import CallbackHandler  # type: ignore
        except Exception:  # noqa: BLE001  (older SDK layout fallback)
            from langfuse.callback import CallbackHandler  # type: ignore

        if trace_id:
            try:
                return CallbackHandler(trace_context={"trace_id": trace_id})
            except TypeError:
                # Legacy handler without trace_context kwarg.
                return CallbackHandler()
        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[langfuse] could not build CallbackHandler: %s", exc)
        return None


def build_trace_metadata(
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``metadata`` block LangGraph forwards to the Langfuse handler.

    The Langfuse LangChain handler reads reserved ``langfuse_*`` metadata keys
    off the run config to set trace-level session, user, tags, and name. Merge
    the result into ``config["metadata"]`` when invoking the graph.
    """
    md: Dict[str, Any] = dict(extra or {})
    if session_id:
        md["langfuse_session_id"] = session_id
    if user_id:
        md["langfuse_user_id"] = user_id
    if tags:
        md["langfuse_tags"] = tags
    if trace_name:
        md["langfuse_trace_name"] = trace_name
    return md


def get_current_trace_id() -> Optional[str]:
    """Return the active Langfuse trace ID within a traced execution context.

    Callable from inside a LangGraph node running under the Langfuse callback
    handler. Returns ``None`` when Langfuse is disabled or no trace is active.
    Never raises.
    """
    client = get_client()
    if client is None:
        return None
    try:
        getter = getattr(client, "get_current_trace_id", None)
        return getter() if getter else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langfuse] get_current_trace_id failed: %s", exc)
        return None


def score_current_trace(
    *,
    trace_id: str,
    name: str,
    value: float,
    comment: Optional[str] = None,
    data_type: str = "NUMERIC",
) -> None:
    """Attach an evaluation score to an existing trace. Never raises."""
    client = get_client()
    if client is None:
        return
    try:
        # OTEL-based SDK (v3/v4): create_score(); older SDKs: score()
        if hasattr(client, "create_score"):
            client.create_score(
                trace_id=trace_id, name=name, value=value,
                comment=comment, data_type=data_type,
            )
        else:  # pragma: no cover - legacy fallback
            client.score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[langfuse] score_current_trace failed: %s", exc)


def flush() -> None:
    """Flush buffered events to Langfuse. Safe to call anytime. Never raises.

    Bounded by ``LANGFUSE_FLUSH_TIMEOUT_SECONDS`` so a slow/offline Langfuse can
    never hang the caller (e.g. container shutdown). The SDK's ``flush()`` has
    no timeout parameter, so we run it on a daemon thread and wait at most the
    configured timeout, then return regardless.
    """
    client = get_client()
    if client is None:
        return

    done = threading.Event()

    def _do_flush() -> None:
        try:
            client.flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[langfuse] flush failed: %s", exc)
        finally:
            done.set()

    worker = threading.Thread(target=_do_flush, name="langfuse-flush", daemon=True)
    worker.start()
    if not done.wait(timeout=max(0, _flush_timeout)):
        logger.warning(
            "[langfuse] flush did not complete within %ss — continuing without blocking",
            _flush_timeout,
        )


def shutdown() -> None:
    """Flush and release the client on application shutdown. Never blocks past
    the configured timeout, never raises."""
    global _client, _initialised, _enabled

    client = _client
    if client is not None:
        done = threading.Event()

        def _do_shutdown() -> None:
            try:
                # shutdown() flushes internally; call it directly (bounded below).
                if hasattr(client, "shutdown"):
                    client.shutdown()
                else:  # pragma: no cover - legacy fallback
                    client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[langfuse] shutdown failed: %s", exc)
            finally:
                done.set()

        worker = threading.Thread(target=_do_shutdown, name="langfuse-shutdown", daemon=True)
        worker.start()
        if not done.wait(timeout=max(0, _flush_timeout)):
            logger.warning(
                "[langfuse] shutdown did not complete within %ss — abandoning to avoid hang",
                _flush_timeout,
            )

    _client = None
    _initialised = False
    _enabled = False
