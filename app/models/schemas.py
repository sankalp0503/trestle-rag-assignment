from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    documents: List[str]
    chunks_indexed: int


class RetrievedChunk(BaseModel):
    document_name: str
    chunk_id: int
    text: str
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    retrieved_chunks: List[RetrievedChunk]

