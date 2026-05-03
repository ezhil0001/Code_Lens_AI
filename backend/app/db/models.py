"""Database Models"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from app.db.base import Base


class User(Base):
    """User model for storing user information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Repository(Base):
    """Repository model for storing GitHub repository metadata"""
    __tablename__ = "repositories"
    
    id = Column(Integer, primary_key=True, index=True)
    repo_url = Column(String(512), unique=True, index=True, nullable=False)
    repo_name = Column(String(255), nullable=False)
    description = Column(Text)
    ingestion_status = Column(String(50), default="pending")  # pending, completed, failed
    last_ingested_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ConversationHistory(Base):
    """Conversation history model for chat interactions"""
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    repository_id = Column(Integer, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context_used = Column(Text)  # Store retrieved context
    created_at = Column(DateTime(timezone=True), server_default=func.now())
