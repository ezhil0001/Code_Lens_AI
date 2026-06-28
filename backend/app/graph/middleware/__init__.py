# app/graph/middleware — per-node retry, timeout, and logging hooks.
# Wrapping a node with with_node_middleware() adds automatic retry on
# transient errors and emits structured log lines for each node execution.
