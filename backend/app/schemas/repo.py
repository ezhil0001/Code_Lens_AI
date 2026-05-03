"""Repository Ingestion and Management Schemas"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from datetime import datetime


class RepositoryIngest(BaseModel):
    """Request body for repository ingestion"""
    repo_url: HttpUrl = Field(..., description="GitHub repository URL")
    description: Optional[str] = Field(None, description="Repository description")


class RepositoryResponse(BaseModel):
    """Repository response schema"""
    id: int
    repo_url: str
    repo_name: str
    description: Optional[str]
    ingestion_status: str
    last_ingested_at: Optional[datetime]
    created_at: datetime


class RepositoryListResponse(BaseModel):
    """List of repositories response"""
    repositories: list[RepositoryResponse]
    total: int
