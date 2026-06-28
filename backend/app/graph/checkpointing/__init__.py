"""Checkpointing — AsyncPostgresSaver integration and thread-ID helpers.

LangGraph checkpoints the full AgentState after every node. This lets us
resume interrupted runs (e.g. after a HIL pause), replay past conversations
for debugging, and branch a thread into a new session without losing history.
"""
