# app/graph/guardrails — input and output safety checks.
# input_guardrail_node blocks prompt injection, enforces token limits, and
# strips PII before the query reaches any retrieval or LLM node.
# output_guardrail_node scans generated responses for secrets and dangerous
# code patterns before they are streamed to the client.
