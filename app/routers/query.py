import logging
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import get_rag_service


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question over ingested documents",
    description=(
        "Embed the question, retrieve the top-k most similar chunks from the vector store, "
        "inject them into the LLM prompt, and generate an answer grounded in the documents."
    ),
)
def query_documents(payload: QueryRequest) -> QueryResponse:
    try:
        rag_service = get_rag_service()
        answer, confidence, retrieved_chunks = rag_service.answer_question(
            question=payload.question,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc: 
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to answer query.",
        ) from exc

    return {
        "answer": answer,
        "confidence": confidence,
        "retrieved_chunks": retrieved_chunks
    }

