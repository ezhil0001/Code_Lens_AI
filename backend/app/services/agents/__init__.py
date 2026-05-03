"""Phase 3: Agent Brain Module.

Advanced Agentic Logic & Contextual Reasoning for RAG Stack.

Components:
1. SemanticExampleSelector: Dynamically selects Few-Shot examples
2. FewShotPromptBuilder: Creates Few-Shot prompts with context-specific instructions
3. AgenticRouter: Routes queries to appropriate retrieval/processing paths
4. AgentBrain: Main orchestrator (ties everything together)
5. ChatMemoryManager: Persistent conversation memory with PostgreSQL
6. Tools: LangChain-compatible tool definitions

Architecture:
    User Query
        ↓
    Agentic Router
        ├─→ Routing Decision
        ↓
    Phase 2 Retriever (with dynamic weights)
        └─→ Dual-path search, reranking, context assembly
        ↓
    Few-Shot Example Selector
        └─→ Find similar examples
        ↓
    Prompt Builder
        └─→ Inject instructions + examples + context
        ↓
    LLM (Llama 3.1)
        └─→ Generate response
        ↓
    Memory Manager
        └─→ Store in PostgreSQL
        ↓
    User

Key Features:
✓ Intent-aware routing (codebase, KT, both, or tool)
✓ Few-Shot learning with curated examples
✓ Context-aware memory (remembers technical details)
✓ Automatic summarization (keeps token usage low)
✓ Streaming responses
✓ Tool invocation for specialized tasks
✓ Production-ready memory management
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

from .agentic_router import (
    RoutingDecision,
    RetrievalPriority,
    RoutingConfig,
    RoutingResult,
    AgenticRouter,
    RoutingOrchestrator,
)

from .agent_brain import (
    AgentConfig,
    AgentRequest,
    AgentResponse,
    AgentBrain,
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
    
    # Routing
    "RoutingDecision",
    "RetrievalPriority",
    "RoutingConfig",
    "RoutingResult",
    "AgenticRouter",
    "RoutingOrchestrator",
    
    # Agent Brain
    "AgentConfig",
    "AgentRequest",
    "AgentResponse",
    "AgentBrain",
    
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
