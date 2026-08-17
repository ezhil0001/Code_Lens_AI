"""User feedback → Langfuse scores, trace-linked.

Maps end-user feedback signals onto the originating trace:

* thumbs up/down  → ``user_feedback`` BOOLEAN
* 1–5 rating      → ``user_rating`` NUMERIC (normalised 0–1, raw kept in comment)
* free-text       → ``user_comment`` as score comment

Security (M-4)
--------------
* **Ownership**: trace ids are registered per-user at request time
  (:func:`register_trace_owner`, bounded TTL LRU). Feedback for a trace the
  caller doesn't own is rejected — no cross-user score pollution.
* **Dedup / replay**: scores use a deterministic ``score_id`` derived from
  (trace_id, score name, user) so repeat submissions UPSERT instead of
  appending duplicates.

No-op safe when Langfuse is disabled.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Trace-ownership registry (bounded TTL LRU; per-process) ──────────────────
_OWNERS: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
_OWNERS_LOCK = threading.Lock()
_OWNERS_MAX = 10_000
_OWNERS_TTL_S = 24 * 3600.0


def register_trace_owner(trace_id: Optional[str], user_id: Optional[str]) -> None:
    """Record which user a trace belongs to. Never raises.

    Mirrors into ``app.observability.tracing``'s registry, which is what the
    ``/feedback`` endpoint reads. Keeping two independent registries meant a
    trace registered here was rejected there as "Unknown trace".
    """
    if not trace_id or not user_id:
        return
    try:
        now = time.monotonic()
        with _OWNERS_LOCK:
            _OWNERS[trace_id] = (str(user_id), now)
            _OWNERS.move_to_end(trace_id)
            while len(_OWNERS) > _OWNERS_MAX:
                _OWNERS.popitem(last=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.observability.tracing import register_trace_owner as _register
        _register(trace_id, user_id)
    except Exception:  # noqa: BLE001
        pass


def _is_owner(trace_id: str, user_id: Optional[str]) -> bool:
    """True if user owns the trace, or ownership is unknown (process restart:
    fail-open so legitimate feedback isn't lost — scores are still bounded to
    valid-format trace ids and deduped)."""
    try:
        with _OWNERS_LOCK:
            entry = _OWNERS.get(trace_id)
        if entry is None:
            return True  # unknown (restart / other worker) — fail open
        owner, ts = entry
        if time.monotonic() - ts > _OWNERS_TTL_S:
            return True
        return owner == str(user_id)
    except Exception:  # noqa: BLE001
        return True


def _score_id(trace_id: str, name: str, user_id: Optional[str]) -> str:
    """Deterministic score id → repeat submissions upsert (no duplicates)."""
    return hashlib.sha256(f"{trace_id}::{name}::{user_id or 'anon'}".encode()).hexdigest()[:32]


def _valid_trace_id(trace_id: str) -> bool:
    """Langfuse v4 trace ids are 32-char lowercase hex (W3C trace id)."""
    return len(trace_id) == 32 and all(c in "0123456789abcdef" for c in trace_id)


def record_user_feedback(
    *,
    trace_id: str,
    thumbs_up: Optional[bool] = None,
    rating: Optional[int] = None,
    comment: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """Attach user feedback to a trace. Returns True if anything was recorded.

    Rejects malformed trace ids and traces owned by another user. Repeat
    submissions update the existing scores (deterministic score_id).
    Never raises — feedback capture must never break the API request.
    """
    if not trace_id or (thumbs_up is None and rating is None and not comment):
        return False
    if not _valid_trace_id(trace_id):
        logger.info("[eval] feedback rejected — malformed trace_id")
        return False
    if not _is_owner(trace_id, user_id):
        logger.warning("[eval] feedback rejected — trace not owned by user=%s", user_id)
        return False
    try:
        from app.observability.langfuse_client import get_client

        client = get_client()
        if client is None:
            return False

        recorded = False
        base_comment = (comment or "").strip()[:2000] or None

        if thumbs_up is not None:
            client.create_score(
                trace_id=trace_id,
                name="user_feedback",
                value=1.0 if thumbs_up else 0.0,
                data_type="BOOLEAN",
                comment=base_comment,
                score_id=_score_id(trace_id, "user_feedback", user_id),
                metadata={"user_id": user_id, "session_id": session_id},
            )
            recorded = True

        if rating is not None:
            rating = max(1, min(int(rating), 5))
            client.create_score(
                trace_id=trace_id,
                name="user_rating",
                value=round((rating - 1) / 4.0, 2),  # normalise 1–5 → 0–1
                data_type="NUMERIC",
                comment=f"raw_rating={rating}/5" + (f" | {base_comment}" if base_comment else ""),
                score_id=_score_id(trace_id, "user_rating", user_id),
                metadata={"user_id": user_id, "session_id": session_id},
            )
            recorded = True

        if base_comment and thumbs_up is None and rating is None:
            client.create_score(
                trace_id=trace_id,
                name="user_comment",
                value="commented",
                data_type="CATEGORICAL",
                comment=base_comment,
                score_id=_score_id(trace_id, "user_comment", user_id),
                metadata={"user_id": user_id, "session_id": session_id},
            )
            recorded = True

        return recorded
    except Exception as exc:  # noqa: BLE001
        logger.debug("[eval] record_user_feedback failed: %s", exc)
        return False
