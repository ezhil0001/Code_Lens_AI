"""Phase 3: Context-Aware Chat Memory Manager.

Manages persistent conversation history with:
- PostgreSQL backend for durability
- Automatic history summarization for token efficiency
- Context-awareness (remembers technical details)
- Session management
- User preferences tracking
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Single message in conversation."""
    
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    session_id: str
    user_id: str
    message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens: int = 0  # Estimated token count


@dataclass
class ConversationSession:
    """Represents a conversation session."""
    
    session_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    context_tags: Optional[List[str]] = None  # ["auth", "database", etc.]
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True
    message_count: int = 0


class ContextAwareSummarizer:
    """Summarizes conversation history while preserving technical context.
    
    Strategy:
    1. Identify "key technical concepts" (functions, classes, decisions)
    2. Keep first and last few messages (conversation boundaries)
    3. Summarize middle section, preserving technical details
    4. Result: Compressed history that maintains understanding
    
    Example:
    
    Original (15 messages, 5000 tokens):
    - User: "How does authenticate work?"
    - Assistant: [Full explanation of authenticate function]
    - User: "What about token refresh?"
    - Assistant: [Explanation of refresh]
    - [... 10 more messages ...]
    - User: "So how do I implement this?"
    
    Summarized (4 messages, 1500 tokens):
    - [System summary]: "Previous discussion covered:
        - authenticate() function: Uses JWT, signature verification, DB lookup
        - token refresh: Supports refresh tokens for extended sessions
        - Key decision: Implemented exponential backoff for failed attempts"
    - User: "So how do I implement this?"
    
    Benefits:
    ✓ Fits more context in token window
    ✓ Preserves technical understanding
    ✓ Maintains conversation coherence
    ✓ User can still follow the thread
    """
    
    def __init__(self, max_tokens: int = 4096):
        """Initialize summarizer.
        
        Args:
            max_tokens: Maximum tokens before summarization triggers
        """
        self.max_tokens = max_tokens
        self.technical_keywords = [
            "function", "class", "method", "variable", "database", "api",
            "authentication", "error", "exception", "pattern", "design",
            "architecture", "module", "import", "depends", "calls"
        ]
    
    def should_summarize(self, messages: List[ChatMessage]) -> bool:
        """Check if history should be summarized.
        
        Args:
            messages: List of messages in history
        
        Returns:
            True if total tokens exceed max_tokens
        """
        total_tokens = sum(m.tokens for m in messages)
        return total_tokens > self.max_tokens
    
    def summarize(self, messages: List[ChatMessage]) -> str:
        """Create a summary of message history.
        
        Args:
            messages: List of messages to summarize
        
        Returns:
            Summary text suitable for prepending to new messages
        """
        if not messages:
            return ""
        
        # Keep first and last messages
        boundary_size = max(2, len(messages) // 5)
        boundary_messages = messages[:boundary_size] + messages[-boundary_size:]
        middle_messages = messages[boundary_size:-boundary_size]
        
        # Extract technical concepts from middle messages
        technical_concepts = self._extract_concepts(middle_messages)
        
        # Build summary
        summary = "## CONVERSATION CONTEXT SUMMARY\n\n"
        
        # Key technical points
        if technical_concepts:
            summary += "### Key Technical Points Discussed:\n"
            for concept in technical_concepts[:10]:  # Top 10 concepts
                summary += f"- {concept}\n"
            summary += "\n"
        
        # Recent context
        summary += "### Recent Questions & Answers:\n"
        for msg in boundary_messages[-3:]:  # Last 3 messages
            if msg.role == "user":
                summary += f"- User: {msg.content[:100]}...\n"
            elif msg.role == "assistant":
                summary += f"- Assistant provided detailed explanation (see above)\n"
        
        return summary
    
    def _extract_concepts(self, messages: List[ChatMessage]) -> List[str]:
        """Extract key technical concepts from messages."""
        concepts = set()
        
        for msg in messages:
            if msg.role == "assistant":  # Focus on assistant responses
                content_lower = msg.content.lower()
                
                # Look for technical keywords + context
                for keyword in self.technical_keywords:
                    if keyword in content_lower:
                        # Extract surrounding text (simple approach)
                        idx = content_lower.find(keyword)
                        start = max(0, idx - 30)
                        end = min(len(msg.content), idx + 60)
                        concept = msg.content[start:end].strip()
                        if len(concept) > 10:  # Avoid noise
                            concepts.add(concept[:80])  # Limit length
        
        return list(concepts)[:10]  # Return top 10


class ChatMemoryManager:
    """Manages conversation history with PostgreSQL backend.
    
    Features:
    - Store messages in PostgreSQL
    - Automatic summarization when history grows
    - Session management
    - User preferences tracking
    - Query history and analytics
    
    Database Schema:
    
    conversations:
      - session_id (PK)
      - user_id (FK)
      - title
      - created_at
      - updated_at
      - context_tags
      - is_active
    
    messages:
      - message_id (PK)
      - session_id (FK)
      - role
      - content
      - timestamp
      - tokens
      - metadata (JSON)
    
    summaries:
      - summary_id (PK)
      - session_id (FK)
      - summary_text
      - created_at
      - message_range (start - end message IDs)
    """
    
    def __init__(
        self,
        db_connection=None,
        max_memory_tokens: int = 4096,
        summarization_enabled: bool = True,
    ):
        """Initialize memory manager.
        
        Args:
            db_connection: Database connection (mock for now, real DB in production)
            max_memory_tokens: Max tokens before summarization
            summarization_enabled: Enable automatic summarization
        """
        self.db = db_connection
        self.max_memory_tokens = max_memory_tokens
        self.summarization_enabled = summarization_enabled
        self.summarizer = ContextAwareSummarizer(max_tokens=max_memory_tokens)
        
        # In-memory cache for active sessions
        self.session_cache: Dict[str, ConversationSession] = {}
        self.message_cache: Dict[str, List[ChatMessage]] = {}
        
        logger.info("Initialized ChatMemoryManager")
    
    async def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Add a message to conversation.
        
        Args:
            session_id: Session ID
            user_id: User ID
            role: "user", "assistant", or "system"
            content: Message content
            metadata: Optional metadata
        
        Returns:
            ChatMessage that was stored
        """
        try:
            # Create or get session
            if session_id not in self.session_cache:
                self.session_cache[session_id] = ConversationSession(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            
            # Create message
            message = ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.now(),
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tokens=self._estimate_tokens(content),
            )
            
            # Add to cache
            if session_id not in self.message_cache:
                self.message_cache[session_id] = []
            
            self.message_cache[session_id].append(message)
            
            # Update session
            session = self.session_cache[session_id]
            session.updated_at = datetime.now()
            session.message_count += 1
            
            # Store in database (would be actual DB in production)
            if self.db:
                await self._store_message_db(message)
            
            logger.debug(f"Added message to session {session_id}")
            
            return message
        
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise
    
    async def get_history(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        limit: int = 50,
    ) -> str:
        """Get conversation history formatted for LLM context.
        
        Args:
            session_id: Session ID
            max_tokens: Maximum tokens (uses self.max_memory_tokens if None)
            limit: Maximum messages to retrieve
        
        Returns:
            Formatted history string
        """
        try:
            max_tokens = max_tokens or self.max_memory_tokens
            
            # Get messages from cache or DB
            messages = self.message_cache.get(session_id, [])
            if not messages and self.db:
                messages = await self._fetch_messages_db(session_id, limit)
            
            # Check if summarization needed
            if self.summarization_enabled and self.summarizer.should_summarize(messages):
                # Create summary
                summary = self.summarizer.summarize(messages)
                
                # Keep only recent messages + summary
                recent_messages = messages[-10:]  # Keep last 10
                
                history = summary + "\n\n## RECENT MESSAGES\n\n"
                for msg in recent_messages:
                    history += f"**{msg.role.upper()}**: {msg.content}\n\n"
                
                return history
            
            # No summarization needed, return full history
            history = ""
            for msg in messages[-limit:]:
                history += f"**{msg.role.upper()}**: {msg.content}\n\n"
            
            return history
        
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return ""
    
    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> ConversationSession:
        """Create a new conversation session.
        
        Args:
            user_id: User ID
            title: Session title
            context_tags: Tags for categorization (e.g., ["auth", "database"])
        
        Returns:
            Created ConversationSession
        """
        try:
            import uuid
            
            session_id = str(uuid.uuid4())
            
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                title=title,
                context_tags=context_tags or [],
            )
            
            self.session_cache[session_id] = session
            self.message_cache[session_id] = []
            
            if self.db:
                await self._store_session_db(session)
            
            logger.info(f"Created session {session_id} for user {user_id}")
            
            return session
        
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session information."""
        try:
            if session_id in self.session_cache:
                return self.session_cache[session_id]
            
            if self.db:
                return await self._fetch_session_db(session_id)
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> List[ConversationSession]:
        """List sessions for a user."""
        try:
            # From cache + DB
            user_sessions = [
                s for s in self.session_cache.values()
                if s.user_id == user_id
                and (include_inactive or s.is_active)
            ]
            
            if self.db:
                db_sessions = await self._fetch_user_sessions_db(
                    user_id, limit, include_inactive
                )
                # Merge, preferring cache for active sessions
                session_ids = {s.session_id for s in user_sessions}
                user_sessions.extend([s for s in db_sessions if s.session_id not in session_ids])
            
            # Sort by updated_at, most recent first
            user_sessions.sort(key=lambda s: s.updated_at, reverse=True)
            
            return user_sessions[:limit]
        
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
    
    async def close_session(self, session_id: str) -> None:
        """Close a session (mark as inactive)."""
        try:
            if session_id in self.session_cache:
                self.session_cache[session_id].is_active = False
            
            if self.db:
                await self._update_session_db(session_id, {"is_active": False})
            
            logger.info(f"Closed session {session_id}")
        
        except Exception as e:
            logger.error(f"Error closing session: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation).
        
        Rule of thumb: 1 token ≈ 4 characters for English text
        """
        return max(1, len(text) // 4)
    
    # Database methods (PostgreSQL with async support)
    
    async def _store_message_db(self, message: ChatMessage) -> None:
        """Store message in PostgreSQL database.
        
        Schema:
            CREATE TABLE messages (
                message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id UUID NOT NULL,
                user_id UUID NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                tokens INT DEFAULT 0,
                metadata JSONB DEFAULT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (session_id) REFERENCES conversations(session_id)
            );
            CREATE INDEX idx_messages_session ON messages(session_id);
            CREATE INDEX idx_messages_user ON messages(user_id);
            CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);
        """
        if not self.db:
            logger.warning("Database not configured, skipping message storage")
            return
        
        try:
            query = """
                INSERT INTO messages 
                (session_id, user_id, role, content, timestamp, tokens, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING message_id;
            """
            
            result = await self.db.execute(
                query,
                message.session_id,
                message.user_id,
                message.role,
                message.content,
                message.timestamp,
                message.tokens,
                json.dumps(message.metadata) if message.metadata else None,
            )
            
            message.message_id = result[0]["message_id"] if result else None
            logger.debug(f"Stored message {message.message_id} in session {message.session_id}")
        
        except Exception as e:
            logger.error(f"Error storing message: {e}")
    
    async def _fetch_messages_db(
        self,
        session_id: str,
        limit: int,
    ) -> List[ChatMessage]:
        """Fetch messages from PostgreSQL database.
        
        Retrieves most recent messages for a session, ordered by timestamp DESC.
        """
        if not self.db:
            return []
        
        try:
            query = """
                SELECT message_id, session_id, user_id, role, content, 
                       timestamp, tokens, metadata
                FROM messages
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s;
            """
            
            rows = await self.db.fetch(query, session_id, limit)
            
            messages = []
            for row in rows:
                msg = ChatMessage(
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    message_id=row["message_id"],
                    tokens=row["tokens"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                )
                messages.append(msg)
            
            # Reverse to get oldest first (natural conversation order)
            messages.reverse()
            logger.debug(f"Fetched {len(messages)} messages for session {session_id}")
            return messages
        
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []
    
    async def _store_session_db(self, session: ConversationSession) -> None:
        """Store or create conversation session in PostgreSQL.
        
        Schema:
            CREATE TABLE conversations (
                session_id UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                title VARCHAR(255),
                context_tags TEXT[] DEFAULT NULL,
                metadata JSONB DEFAULT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                message_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
            CREATE INDEX idx_conversations_user ON conversations(user_id);
            CREATE INDEX idx_conversations_active ON conversations(is_active) WHERE is_active = TRUE;
        """
        if not self.db:
            logger.warning("Database not configured, skipping session storage")
            return
        
        try:
            query = """
                INSERT INTO conversations 
                (session_id, user_id, title, context_tags, metadata, is_active, message_count, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    context_tags = EXCLUDED.context_tags,
                    metadata = EXCLUDED.metadata,
                    is_active = EXCLUDED.is_active,
                    message_count = EXCLUDED.message_count,
                    updated_at = NOW();
            """
            
            await self.db.execute(
                query,
                session.session_id,
                session.user_id,
                session.title,
                session.context_tags,
                json.dumps(session.metadata) if session.metadata else None,
                session.is_active,
                session.message_count,
                session.created_at,
                session.updated_at,
            )
            
            logger.debug(f"Stored session {session.session_id} for user {session.user_id}")
        
        except Exception as e:
            logger.error(f"Error storing session: {e}")
    
    async def _fetch_session_db(self, session_id: str) -> Optional[ConversationSession]:
        """Fetch conversation session from PostgreSQL by ID."""
        if not self.db:
            return None
        
        try:
            query = """
                SELECT session_id, user_id, title, context_tags, metadata, 
                       is_active, message_count, created_at, updated_at
                FROM conversations
                WHERE session_id = %s;
            """
            
            row = await self.db.fetchrow(query, session_id)
            
            if not row:
                return None
            
            session = ConversationSession(
                session_id=row["session_id"],
                user_id=row["user_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                title=row["title"],
                context_tags=row["context_tags"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                is_active=row["is_active"],
                message_count=row["message_count"],
            )
            
            logger.debug(f"Fetched session {session_id}")
            return session
        
        except Exception as e:
            logger.error(f"Error fetching session: {e}")
            return None
    
    async def _fetch_user_sessions_db(
        self,
        user_id: str,
        limit: int,
        include_inactive: bool,
    ) -> List[ConversationSession]:
        """Fetch all sessions for a user from PostgreSQL.
        
        Ordered by recent activity (updated_at DESC).
        Optionally filters to active sessions only.
        """
        if not self.db:
            return []
        
        try:
            where_clause = "WHERE user_id = %s"
            if not include_inactive:
                where_clause += " AND is_active = TRUE"
            
            query = f"""
                SELECT session_id, user_id, title, context_tags, metadata, 
                       is_active, message_count, created_at, updated_at
                FROM conversations
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT %s;
            """
            
            rows = await self.db.fetch(query, user_id, limit)
            
            sessions = []
            for row in rows:
                session = ConversationSession(
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    title=row["title"],
                    context_tags=row["context_tags"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                    is_active=row["is_active"],
                    message_count=row["message_count"],
                )
                sessions.append(session)
            
            logger.debug(f"Fetched {len(sessions)} sessions for user {user_id}")
            return sessions
        
        except Exception as e:
            logger.error(f"Error fetching user sessions: {e}")
            return []
    
    async def _update_session_db(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session fields in PostgreSQL.
        
        Dynamically builds SET clause based on provided updates.
        Automatically updates updated_at timestamp.
        """
        if not self.db:
            logger.warning("Database not configured, skipping session update")
            return
        
        try:
            # Allowed fields to update (security: prevent injection)
            allowed_fields = {"title", "context_tags", "is_active", "message_count", "metadata"}
            filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not filtered_updates:
                logger.debug("No valid fields to update")
                return
            
            # Build SET clause
            set_parts = []
            values = []
            for i, (field, value) in enumerate(filtered_updates.items(), 1):
                if field == "metadata":
                    set_parts.append(f"{field} = ${i}")
                    values.append(json.dumps(value) if value else None)
                elif field == "context_tags":
                    set_parts.append(f"{field} = ${i}")
                    values.append(value)
                else:
                    set_parts.append(f"{field} = ${i}")
                    values.append(value)
            
            set_clause = ", ".join(set_parts)
            values.append(session_id)  # For WHERE clause
            
            query = f"""
                UPDATE conversations
                SET {set_clause}, updated_at = NOW()
                WHERE session_id = ${len(values)};
            """
            
            await self.db.execute(query, *values)
            logger.debug(f"Updated session {session_id} with {len(filtered_updates)} fields")
        
        except Exception as e:
            logger.error(f"Error updating session: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        total_messages = sum(len(msgs) for msgs in self.message_cache.values())
        total_tokens = sum(
            sum(m.tokens for m in msgs)
            for msgs in self.message_cache.values()
        )
        
        return {
            "active_sessions": len([s for s in self.session_cache.values() if s.is_active]),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "avg_tokens_per_message": (
                total_tokens / total_messages if total_messages > 0 else 0
            ),
            "cache_size": len(self.message_cache),
        }
