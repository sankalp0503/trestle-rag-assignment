from fastapi import FastAPI
from app.core.logging_config import setup_logging
from app.api import register_routers


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="RAG Backend Service",
        description="Retrieval-Augmented Generation backend for document-grounded QA.",
        version="0.1.0",
    )

    register_routers(app)

    return app


app = create_app()


