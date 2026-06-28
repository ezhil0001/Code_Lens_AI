"""Memory layer — short-term conversation window + long-term pgvector fact store.

Short-term memory is scoped to a single session (namespaced by user_id to
prevent cross-user leakage). Long-term memory persists key facts across
sessions so the agent can recall previous context without the user repeating
themselves every time.
"""
