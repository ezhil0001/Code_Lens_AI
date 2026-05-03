"""Main API Router Setup"""

from fastapi import APIRouter

# This file will be used to aggregate all endpoint routers
# Example imports (to be added as endpoints are created):
# from app.api.endpoints import chat, repository, ingestion

router = APIRouter(prefix="/api/v1")

# Include endpoint routers here
# router.include_router(chat.router, prefix="/chat", tags=["Chat"])
# router.include_router(repository.router, prefix="/repositories", tags=["Repositories"])
# router.include_router(ingestion.router, prefix="/ingest", tags=["Ingestion"])
