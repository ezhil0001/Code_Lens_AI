"""Phase 3: Few-Shot Prompt Templates.

Creates dynamic Few-Shot prompt templates that inject contextually relevant
examples into the LLM prompt. Includes separate templates for code explanation
vs documentation queries.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

try:
    from langchain_core.output_parsers import PydanticOutputParser
    HAS_PYDANTIC_PARSER = True
except ImportError:  # pragma: no cover
    PydanticOutputParser = None  # type: ignore
    HAS_PYDANTIC_PARSER = False



logger = logging.getLogger(__name__)


# Module-level parser — single source of truth for structured output format.
ANSWER_PARSER = None 


class QueryContext(str, Enum):
    """Categorizes the type of query for prompt selection."""
    
    CODE_EXPLANATION = "code_explanation"  # "How does this code work?"
    KT_DOCUMENTATION = "kt_documentation"  # "Explain the architecture"
    TECHNICAL_QUERY = "technical_query"    # "How to implement X?"
    TROUBLESHOOTING = "troubleshooting"    # "Why is Y failing?"
    ARCHITECTURE = "architecture"          # "System design questions"


class FewShotPromptBuilder:
    """Builds Few-Shot prompt templates dynamically based on query context.
    
    Architecture:
    1. Analyzes user query to determine context
    2. Selects appropriate example set (code vs KT vs architecture)
    3. Injects high-quality examples into prompt
    4. Adds context-specific instructions
    5. Returns final prompt for LLM
    
    Key Insight:
    - Code explanation examples show: "Here's how code works"
    - KT documentation examples show: "Here's the conceptual understanding"
    - Prompt instructions vary: prioritize structure vs concepts
    """
    
    # System prompt (constant for all queries) - ANTI-HALLUCINATION FOCUSED
    SYSTEM_PROMPT = """You are an expert code assistant helping developers understand complex codebases.

CORE PRINCIPLES:
1. Accuracy: Provide correct, verified information ONLY
2. Context-Aware: Use provided code context to give specific answers
3. Complete: Include enough detail for understanding without overwhelming
4. Actionable: Explain not just what, but how and why

HALLUCINATION PREVENTION (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1: INFORMATION BOUNDARIES
- ONLY use information from the provided context
- If information is NOT in the context, clearly state: "This information is not in the provided context"
- DO NOT guess, infer, or make up details about undiscussed code
- NEVER claim certainty about things not explicitly shown

RULE 2: ADMISSION OF UNCERTAINTY
- When unsure about details: Say "Based on the context, I can see X, but the exact implementation of Y is not shown"
- Suggest looking at related files or documentation for missing information
- Admit knowledge gaps rather than speculating

RULE 3: VERIFICATION
Before answering:
  ✓ Is this information in the provided context?
  ✓ Is this a direct quote or derived from explicit facts?
  ✓ Could there be alternative explanations not shown?
  If ANY answer is "no" or "maybe" → Admit uncertainty

RULE 4: CONTEXT VALIDATION
- Quote relevant code snippets when explaining
- Reference specific line numbers or function names from context
- If context is incomplete, note it: "The context shows function X but not function Y it calls"
- Don't fill gaps with general knowledge

EXAMPLE GOOD RESPONSE:
"The authenticate function validates tokens by:
1. Extracting the token from headers (line 45)
2. Calling jwt.decode() (line 48) 
3. Looking up the user in the database (line 52)

The exact validation logic isn't shown in the context, but these are the explicit steps."

EXAMPLE BAD RESPONSE (HALLUCINATION):
"The authenticate function validates tokens using standard JWT practices including:
1. XYZ validation (NOT SHOWN IN CONTEXT)
2. ABC checking (MADE UP)
3. DEF verification (GENERIC KNOWLEDGE, NOT FROM CODE)"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT CONSTRAINTS:

- NEVER hallucinate code that wasn't provided
- ALWAYS reference the provided code/context
- NEVER claim certainty about code you haven't seen
- ALWAYS acknowledge when something is outside provided context"""
    
    # Context-specific instruction templates
    CODE_EXPLANATION_INSTRUCTIONS = """You are explaining code from a real codebase.

INSTRUCTIONS FOR CODE EXPLANATION:
1. Structure First: Explain the function/class structure
2. Logic Flow: Walk through the execution step-by-step
3. Dependencies: Mention what this code depends on
4. Purpose: Explain why this code exists (not just what it does)
5. Examples: Provide concrete examples of how it's used

PRIORITIZE:
- Code structure and flow
- Type hints and signatures
- Error handling patterns
- Performance implications

FORMAT YOUR RESPONSE AS:
## Overview
[1-2 sentences on what this code does]

## Structure
[Function/class components]

## Logic Flow
[Step-by-step execution]

## Key Points
[Important considerations]

## Related Code
[Functions/classes that interact with this]"""
    
    KT_DOCUMENTATION_INSTRUCTIONS = """You are explaining knowledge transfer concepts.

INSTRUCTIONS FOR KT DOCUMENTATION:
1. Conceptual Foundation: Start with the "why"
2. Relationships: Explain how concepts connect
3. Patterns: Identify recurring themes/architectures
4. Trade-offs: Discuss design decisions and alternatives
5. Examples: Provide real examples from the codebase

PRIORITIZE:
- Conceptual understanding
- Architectural patterns
- Design philosophy
- Learning trajectory (beginner → advanced)

FORMAT YOUR RESPONSE AS:
## Concept
[What is this concept?]

## Why It Matters
[Strategic importance]

## Core Principles
[Fundamental ideas]

## Implementation in This Codebase
[How it's realized]

## Common Patterns
[Recurring applications]"""
    
    TECHNICAL_QUERY_INSTRUCTIONS = """You are helping a developer solve a technical problem.

INSTRUCTIONS FOR TECHNICAL QUERIES:
1. Problem Identification: Clarify the problem
2. Root Cause: Analyze why this is happening
3. Solutions: Provide concrete solutions (prioritized by effectiveness)
4. Trade-offs: Discuss pros/cons of each approach
5. Implementation: Show how to implement the solution

PRIORITIZE:
- Practical solutions
- Performance implications
- Edge cases and error handling
- Integration with existing code"""
    
    TROUBLESHOOTING_INSTRUCTIONS = """You are helping debug a problem.

INSTRUCTIONS FOR TROUBLESHOOTING:
1. Error Analysis: What does the error mean?
2. Likely Causes: Most probable reasons (ordered by likelihood)
3. Diagnostic Steps: How to confirm the cause
4. Solutions: Fix options (quick fix vs proper fix)
5. Prevention: How to avoid this in the future

PRIORITIZE:
- Immediate resolution
- Understanding the root cause
- Proper error handling
- Defensive programming patterns"""
    
    def __init__(self, example_selector=None):
        """Initialize the Few-Shot prompt builder.
        
        Args:
            example_selector: SemanticExampleSelector instance
        """
        self.example_selector = example_selector
        logger.info("Initialized FewShotPromptBuilder")
    
    def build_prompt(
        self,
        user_query: str,
        retrieved_context: str,
        context_type: Optional[QueryContext] = None,
        num_examples: int = 2,
    ) -> str:
        """Build a Few-Shot prompt for the LLM.
        
        Args:
            user_query: Original user query
            retrieved_context: Context from Phase 2 retrieval (code chunks + parent)
            context_type: Type of query (auto-detected if None)
            num_examples: Number of Few-Shot examples to include
        
        Returns:
            Complete prompt ready for LLM
        
        Structure:
            1. System prompt (core principles)
            2. Context-specific instructions
            3. Few-Shot examples (if selector available)
            4. Retrieved code context
            5. User query
        """
        try:
            # Determine query context if not provided
            if context_type is None:
                context_type = self._detect_context(user_query)
            
            # Get context-specific instructions
            instructions = self._get_instructions(context_type)
            
            # Get Few-Shot examples
            examples_section = ""
            if self.example_selector:
                examples = self.example_selector.select_examples(
                    query=user_query,
                    k=num_examples,
                    categories=[context_type.value],
                )
                if examples:
                    examples_section = self._format_examples(examples, context_type)
            
            # Build final prompt
            # Build final prompt — plain text response
            prompt = f"""{self.SYSTEM_PROMPT}

{instructions}

{examples_section}

## CODE CONTEXT (From Codebase)
```
{retrieved_context}
```

## USER QUERY
{user_query}

## YOUR RESPONSE
"""
            
            logger.debug(
                f"Built prompt for {context_type.value}: {len(prompt)} tokens"
            )
            return prompt
        
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Return minimal prompt on error
            return f"""Answer the following question based on the provided code context.

CODE CONTEXT:
{retrieved_context}

QUESTION: {user_query}

RESPONSE:"""
    
    def _detect_context(self, query: str) -> QueryContext:
        """Detect the type of query from natural language.
        
        Uses keyword matching to categorize:
        - "how to" → technical query
        - "why", "explain" → code explanation
        - "architecture", "design" → architecture
        - "error", "fix", "bug" → troubleshooting
        - "document", "what is" → KT documentation
        """
        query_lower = query.lower()
        
        # Troubleshooting keywords
        troubleshooting_keywords = ["error", "fix", "bug", "fail", "crash", "not working", "why not"]
        if any(kw in query_lower for kw in troubleshooting_keywords):
            return QueryContext.TROUBLESHOOTING
        
        # Architecture keywords
        architecture_keywords = ["architecture", "design", "pattern", "structure", "flow", "system"]
        if any(kw in query_lower for kw in architecture_keywords):
            return QueryContext.ARCHITECTURE
        
        # Technical query keywords
        technical_keywords = ["how to", "how do", "implement", "build", "create"]
        if any(kw in query_lower for kw in technical_keywords):
            return QueryContext.TECHNICAL_QUERY
        
        # Explanation keywords
        explanation_keywords = ["explain", "what does", "how does", "show me"]
        if any(kw in query_lower for kw in explanation_keywords):
            return QueryContext.CODE_EXPLANATION
        
        # Default to code explanation
        return QueryContext.CODE_EXPLANATION
    
    def _get_instructions(self, context_type: QueryContext) -> str:
        """Get context-specific instructions for the query type."""
        mapping = {
            QueryContext.CODE_EXPLANATION: self.CODE_EXPLANATION_INSTRUCTIONS,
            QueryContext.KT_DOCUMENTATION: self.KT_DOCUMENTATION_INSTRUCTIONS,
            QueryContext.TECHNICAL_QUERY: self.TECHNICAL_QUERY_INSTRUCTIONS,
            QueryContext.TROUBLESHOOTING: self.TROUBLESHOOTING_INSTRUCTIONS,
            QueryContext.ARCHITECTURE: self.KT_DOCUMENTATION_INSTRUCTIONS,  # Same as KT
        }
        return mapping.get(context_type, self.CODE_EXPLANATION_INSTRUCTIONS)
    
    def _format_examples(self, examples: List[Any], context_type: QueryContext) -> str:
        """Format examples for inclusion in prompt.
        
        Args:
            examples: List of ExamplePair objects
            context_type: Type of query context
        
        Returns:
            Formatted examples section
        """
        if not examples:
            return ""
        
        examples_text = "## EXAMPLES (Similar Questions Answered)\n\n"
        
        for i, example in enumerate(examples, 1):
            examples_text += f"### Example {i}: {example.question}\n"
            examples_text += f"**Answer:**\n{example.answer}\n\n"
        
        return examples_text
    
    def build_system_prompt(self) -> str:
        """Return just the system prompt."""
        return self.SYSTEM_PROMPT

    @staticmethod
    def get_output_parser():
        """Return the shared PydanticOutputParser bound to AnswerSchema.

        Callers can use `parser.parse(llm_text)` to coerce the LLM output
        into a validated `AnswerSchema` instance.
        """
        return ANSWER_PARSER
    
    def build_instructions(self, context_type: QueryContext) -> str:
        """Return just the instructions for a query type."""
        return self._get_instructions(context_type)


class PromptTemplate:
    """Pre-built prompt templates for common scenarios."""
    
    @staticmethod
    def code_walkthrough_template(code: str, query: str) -> str:
        """Template for walking through code line-by-line."""
        return f"""Please walk me through this code step-by-step:

```python
{code}
```

Question: {query}

Please explain:
1. What this code does overall
2. What happens on each line
3. Any important edge cases
4. How this code integrates with the rest of the system"""
    
    @staticmethod
    def architecture_explanation_template(context: str, query: str) -> str:
        """Template for explaining system architecture."""
        return f"""Based on this codebase context:

{context}

{query}

Please explain:
1. The architectural pattern being used
2. Key components and their responsibilities
3. How data flows through the system
4. Important design decisions and why they were made"""
    
    @staticmethod
    def debugging_template(error: str, code: str, query: str) -> str:
        """Template for debugging help."""
        return f"""I'm getting this error:
{error}

In this code:
```python
{code}
```

Question: {query}

Please:
1. Explain what this error means
2. List likely causes (most probable first)
3. Show how to fix it
4. Suggest how to prevent this in the future"""
    
    @staticmethod
    def best_practices_template(code: str, query: str) -> str:
        """Template for best practices questions."""
        return f"""Looking at this code:

```python
{code}
```

{query}

Please provide:
1. Best practices for this pattern
2. Common mistakes to avoid
3. Performance considerations
4. Security implications (if applicable)
5. Example of improved implementation"""
