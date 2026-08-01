"""Regression dataset management — production traces → reusable datasets.

Curates valuable production interactions into a versioned Langfuse dataset
(default name from ``LANGFUSE_REGRESSION_DATASET``, fallback
``codelens-regression``) so that prompt/model/retrieval changes can be tested
offline against real user queries before deployment (see ``experiments.py``).

Items are linked back to their originating trace via ``source_trace_id``,
giving one-click navigation from a dataset item to the production trace it
came from. Duplicate protection uses a deterministic item ``id`` derived from
the query, so re-adding the same query is an upsert, not a duplicate.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.getenv("LANGFUSE_REGRESSION_DATASET", "codelens-regression")

# Existence cache — avoids a per-curation get_dataset() round-trip (O(items)).
_KNOWN_DATASETS: set = set()


def ensure_dataset(name: str = DEFAULT_DATASET, description: Optional[str] = None) -> bool:
    """Create the dataset if it doesn't exist (idempotent). Never raises."""
    if name in _KNOWN_DATASETS:
        return True
    try:
        from app.observability.langfuse_client import get_client

        client = get_client()
        if client is None:
            return False
        try:
            client.get_dataset(name, fetch_items_page_size=1)
            _KNOWN_DATASETS.add(name)
            return True
        except Exception:  # noqa: BLE001  (not found → create)
            pass
        client.create_dataset(
            name=name,
            description=description
            or "Curated production queries for offline regression experiments.",
            metadata={"source": "codelens-production", "managed_by": "app.observability.evaluation"},
        )
        _KNOWN_DATASETS.add(name)
        logger.info("[eval] created Langfuse dataset '%s'", name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("[eval] ensure_dataset failed: %s", exc)
        return False


def add_interaction_to_dataset(
    *,
    query: str,
    expected_output: Optional[str] = None,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    dataset_name: str = DEFAULT_DATASET,
) -> bool:
    """Add one production interaction as a dataset item (idempotent upsert).

    ``expected_output`` is typically the production answer after human review,
    or left None until annotated. Returns True on success. Never raises.
    """
    if not query or not query.strip():
        return False
    try:
        from app.observability.langfuse_client import get_client

        client = get_client()
        if client is None:
            return False
        if not ensure_dataset(dataset_name):
            return False

        # Deterministic id → same query upserts instead of duplicating.
        item_id = hashlib.sha256(f"{dataset_name}::{query.strip().lower()}".encode()).hexdigest()[:32]
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input={"query": query},
            expected_output=expected_output,
            metadata=metadata or {},
            source_trace_id=trace_id,
        )
        logger.info("[eval] dataset item upserted (dataset=%s, id=%s)", dataset_name, item_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("[eval] add_interaction_to_dataset failed: %s", exc)
        return False


def get_dataset_items(dataset_name: str = DEFAULT_DATASET) -> List[Any]:
    """Fetch dataset items for experiment runs. Empty list on any failure."""
    try:
        from app.observability.langfuse_client import get_client

        client = get_client()
        if client is None:
            return []
        ds = client.get_dataset(dataset_name)
        return list(getattr(ds, "items", []) or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("[eval] get_dataset_items failed: %s", exc)
        return []
