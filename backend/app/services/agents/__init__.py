"""Phase 3: Agent Brain Module.

Advanced Agentic Logic & Contextual Reasoning for RAG Stack.

Components:
1. SemanticExampleSelector: Dynamically selects Few-Shot examples
2. FewShotPromptBuilder: Creates Few-Shot prompts with context-specific instructions
3. ChatMemoryManager: Persistent conversation memory with PostgreSQL
4. Tools: LangChain-compatible tool definitions

Architecture (v2 — LangGraph supervisor):
    User Query → LangGraph supervisor graph → agent sub-graphs → response

Note: AgenticRouter and AgentBrain (v1) have been removed.
All routing is now handled by app.graph.nodes.intent_classifier and
the LangGraph supervisor graph in app.graph.supervisor_graph.
"""

from .semantic_example_selector import (
    ExamplePair,
    SemanticExampleSelector,
)

from .few_shot_prompt import (
    QueryContext,
    FewShotPromptBuilder,
    PromptTemplate,
)

from .chat_memory_manager import (
    ChatMessage,
    ConversationSession,
    ContextAwareSummarizer,
)

from .langchain_memory_manager import ChatMemoryManager

from .tools import (
    ToolType,
    CodeSearchTool,
    DocumentationSearchTool,
    CodeAnalyzeTool,
    CodeExecuteTool,
    LintTool,
    RefactorTool,
    ToolRegistry,
)

__all__ = [
    # Example Selector
    "ExamplePair",
    "SemanticExampleSelector",

    # Prompt Building
    "QueryContext",
    "FewShotPromptBuilder",
    "PromptTemplate",

    # Memory
    "ChatMessage",
    "ConversationSession",
    "ContextAwareSummarizer",
    "ChatMemoryManager",

    # Tools
    "ToolType",
    "CodeSearchTool",
    "DocumentationSearchTool",
    "CodeAnalyzeTool",
    "CodeExecuteTool",
    "LintTool",
    "RefactorTool",
    "ToolRegistry",
]
