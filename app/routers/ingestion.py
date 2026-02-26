import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.schemas import IngestResponse
from app.services.rag_service import get_rag_service


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents into the vector store",
    description=(
        "Upload one or more documents (PDF / images / TXT / Markdown). "
        "The service will extract text, chunk it, generate embeddings, and store them in the vector database."
    ),
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Documents to ingest."),
    chunk_size: int = Form(default=None, description="Optional override of default chunk size."),
    chunk_overlap: int = Form(default=None, description="Optional override of default chunk overlap."),
) -> IngestResponse:
    settings = get_settings()
    effective_chunk_size = chunk_size or settings.chunk_size
    effective_chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Add print statements to log each step of the ingestion process
    print("Starting document ingestion...")
    print(f"Effective chunk size: {effective_chunk_size}, Effective chunk overlap: {effective_chunk_overlap}")

    if effective_chunk_overlap >= effective_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be less than chunk_size.",
        )

    items: list[tuple[str, bytes, str | None]] = []
    for file in files:
        print(f"Processing file: {file.filename}")
        try:
            content = await file.read()
            items.append((file.filename, content, file.content_type))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read uploaded file %s: %s", file.filename, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read file: {file.filename}",
            ) from exc

    rag_service = get_rag_service()
    document_names, chunks_indexed = rag_service.ingest_documents(
        items=items,
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
    )

    print(f"Document names: {document_names}, Chunks indexed: {chunks_indexed}")

    if not document_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text could be extracted from the uploaded documents.",
        )

    return IngestResponse(documents=document_names, chunks_indexed=chunks_indexed)

