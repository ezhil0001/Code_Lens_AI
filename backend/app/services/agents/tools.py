"""Phase 3: LangChain Tool Definitions.

Defines RetrieverEngine and other tools as LangChain Tools
that can be invoked by the Agent Brain for specialized operations.
"""

import logging
from typing import Any, Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    """Types of tools available to the agent."""
    
    CODE_SEARCH = "code_search"              # Query codebase
    DOC_SEARCH = "doc_search"                # Query KT documentation
    CODE_ANALYZE = "code_analyze"            # Analyze code structure
    EXECUTE = "execute"                      # Execute code snippets
    LINT = "lint"                            # Code linting
    REFACTOR = "refactor"                    # Code refactoring suggestions


class CodeSearchTool:
    """Tool for searching codebase.
    
    Wraps the Phase 2 RetrieverEngine as a LangChain-compatible tool.
    """
    
    name = "code_search"
    description = """Search the codebase for code snippets, functions, classes, or patterns.
    
    Use this when you need to find specific code, understand implementation details,
    or locate functions/classes in the codebase.
    
    Input: A natural language query about code
    Output: Relevant code snippets with full context (parent functions/classes)
    """
    
    def __init__(self, retriever_engine):
        """Initialize code search tool.
        
        Args:
            retriever_engine: Phase 2 RetrieverEngine instance
        """
        self.retriever = retriever_engine
    
    async def __call__(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute code search.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            Search results with code and metadata
        """
        try:
            results = self.retriever.retrieve(query, top_k=top_k)
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
            }
        
        except Exception as e:
            logger.error(f"Code search error: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
            }


class DocumentationSearchTool:
    """Tool for searching KT documentation.
    
    Searches documentation about architecture, patterns, and knowledge transfer.
    """
    
    name = "doc_search"
    description = """Search the Knowledge Transfer documentation for concepts, patterns, and architecture.
    
    Use this when you need conceptual understanding, architectural patterns,
    design decisions, or system-level information.
    
    Input: A natural language query about concepts/architecture
    Output: Relevant documentation with explanations
    """
    
    def __init__(self, doc_retriever=None):
        """Initialize documentation search tool.
        
        Args:
            doc_retriever: Optional separate retriever for documentation
        """
        self.retriever = doc_retriever
    
    async def __call__(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Execute documentation search."""
        try:
            if not self.retriever:
                return {
                    "status": "error",
                    "query": query,
                    "error": "Documentation retriever not configured",
                    "results": [],
                }
            
            results = self.retriever.retrieve(query, top_k=top_k)
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
            }
        
        except Exception as e:
            logger.error(f"Documentation search error: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
            }


class CodeAnalyzeTool:
    """Tool for analyzing code structure.
    
    Provides static analysis: identify functions, classes, dependencies.
    """
    
    name = "code_analyze"
    description = """Analyze code structure to understand functions, classes, and dependencies.
    
    Use this to understand:
    - Function signatures and documentation
    - Class hierarchies and relationships
    - Module dependencies
    - Code patterns and conventions
    
    Input: Code snippet or query about code structure
    Output: Structural analysis with relationships
    """
    
    async def __call__(self, code: str) -> Dict[str, Any]:
        """Analyze code structure."""
        try:
            analysis = {
                "status": "success",
                "functions": self._extract_functions(code),
                "classes": self._extract_classes(code),
                "imports": self._extract_imports(code),
                "complexity": self._estimate_complexity(code),
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _extract_functions(self, code: str) -> List[str]:
        """Extract function definitions."""
        functions = []
        for line in code.split('\n'):
            if line.strip().startswith('def '):
                func_name = line.strip().split('(')[0].replace('def ', '')
                functions.append(func_name)
        return functions
    
    def _extract_classes(self, code: str) -> List[str]:
        """Extract class definitions."""
        classes = []
        for line in code.split('\n'):
            if line.strip().startswith('class '):
                class_name = line.strip().split('(')[0].replace('class ', '')
                classes.append(class_name)
        return classes
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements."""
        imports = []
        for line in code.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line.strip())
        return imports
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity (simplified)."""
        lines = len(code.split('\n'))
        cyclomatic = code.count('if ') + code.count('else') + code.count('elif')
        
        if cyclomatic < 5 and lines < 50:
            return "low"
        elif cyclomatic < 10 and lines < 150:
            return "medium"
        else:
            return "high"


class CodeExecuteTool:
    """Tool for executing code snippets (in sandbox).
    
    WARNING: Only executes in controlled sandbox environment.
    Never execute untrusted code.
    """
    
    name = "execute"
    description = """Execute a code snippet in a safe sandbox environment.
    
    SECURITY: Only for demonstration/testing code.
    Never execute untrusted or production code.
    
    Input: Python code snippet
    Output: Execution result or error
    """
    
    async def __call__(self, code: str, timeout_seconds: int = 5) -> Dict[str, Any]:
        """Execute code in sandbox."""
        logger.warning(f"Code execution requested (sandbox mode)")
        
        return {
            "status": "disabled",
            "message": "Code execution disabled in production. Use test environments only.",
            "code": code[:100] + "...",
        }


class LintTool:
    """Tool for linting code.
    
    Checks for style, errors, and best practices.
    """
    
    name = "lint"
    description = """Lint code for style issues, potential bugs, and best practices.
    
    Use this to review code quality, find issues, and suggest improvements.
    
    Input: Code snippet
    Output: Linting report with issues and suggestions
    """
    
    async def __call__(self, code: str) -> Dict[str, Any]:
        """Lint code snippet."""
        issues = []
        
        # Simple checks
        if '\t' in code:
            issues.append({
                "type": "style",
                "severity": "warning",
                "message": "Uses tabs instead of spaces",
            })
        
        if len([l for l in code.split('\n') if len(l) > 100]) > 0:
            issues.append({
                "type": "style",
                "severity": "info",
                "message": "Some lines exceed 100 characters",
            })
        
        if code.count('except:') > 0:
            issues.append({
                "type": "error",
                "severity": "error",
                "message": "Bare except clauses are problematic",
            })
        
        return {
            "status": "success",
            "issues_count": len(issues),
            "issues": issues,
        }


class RefactorTool:
    """Tool for suggesting code refactorings."""
    
    name = "refactor"
    description = """Suggest code refactoring improvements for readability and maintainability.
    
    Use this to improve code quality, reduce complexity, and follow best practices.
    
    Input: Code snippet
    Output: Refactoring suggestions
    """
    
    async def __call__(self, code: str) -> Dict[str, Any]:
        """Suggest refactorings."""
        suggestions = []
        
        # Detect long functions
        if len(code.split('\n')) > 30:
            suggestions.append({
                "type": "complexity",
                "suggestion": "Consider breaking this into smaller functions",
                "reason": "Function is quite long (>30 lines)",
            })
        
        # Detect nested if statements
        if code.count('    if') > 2:
            suggestions.append({
                "type": "readability",
                "suggestion": "Consider extracting nested conditions into helper methods",
                "reason": "Deep nesting reduces readability",
            })
        
        # Detect magic numbers
        import re
        numbers = re.findall(r'\b\d{2,}\b', code)
        if numbers:
            suggestions.append({
                "type": "maintainability",
                "suggestion": "Consider replacing magic numbers with named constants",
                "reason": "Makes code more maintainable",
                "examples": numbers[:3],
            })
        
        return {
            "status": "success",
            "suggestions_count": len(suggestions),
            "suggestions": suggestions,
        }


class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Any] = {}
        logger.info("Initialized ToolRegistry")
    
    def register(self, tool_name: str, tool: Any) -> None:
        """Register a tool.
        
        Args:
            tool_name: Name of tool
            tool: Tool instance
        """
        self.tools[tool_name] = tool
        logger.info(f"Registered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get tool description."""
        tool = self.tools.get(tool_name)
        return getattr(tool, 'description', None)
    
    def create_langchain_tools(self) -> List[Any]:
        """Create LangChain-compatible tool objects.
        
        This would integrate with LangChain's Tool class in production.
        For now, returns basic structure.
        """
        langchain_tools = []
        
        for tool_name, tool in self.tools.items():
            tool_obj = {
                "name": tool_name,
                "description": getattr(tool, 'description', ''),
                "tool": tool,
            }
            langchain_tools.append(tool_obj)
        
        return langchain_tools
