from fastapi import FastAPI
from app.routers import ingestion, query


def register_routers(app: FastAPI) -> None:
    """
    Register all API routers.
    """
    app.include_router(ingestion.router, prefix="/ingest", tags=["ingestion"])
    app.include_router(query.router, prefix="/query", tags=["query"])