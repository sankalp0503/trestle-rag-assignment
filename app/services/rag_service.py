import logging
from functools import lru_cache
from typing import Iterable, List, Tuple
import re
from typing import List
from openai import OpenAI
from openai import RateLimitError
from app.core.config import get_settings
from app.models.schemas import RetrievedChunk
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingService
from app.services.text_extraction import TextExtractor
from app.services.vector_store import FaissVectorStore, StoredMetadata


logger = logging.getLogger(__name__)


class RAGService:
    """
    Orchestrates ingestion and question answering over the vector store.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        # Client is optional – if no API key is configured we still allow
        # the service to run using local embeddings and a heuristic answer.
        self.client: OpenAI | None = None
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)

        self.embedding_service = EmbeddingService(client=self.client)
        self.text_extractor = TextExtractor()
        self.vector_store = FaissVectorStore()

    # -------- Ingestion --------

    def ingest_documents(
        self,
        items: Iterable[tuple[str, bytes, str | None]],
        chunk_size: int,
        chunk_overlap: int,
    ) -> Tuple[List[str], int]:
        """
        Ingest a collection of documents into the vector store.

        :param items: Iterable of (file_name, file_bytes, content_type).
        :return: (list of document names, total chunks indexed)
        """
        document_names: List[str] = []
        all_chunks: List[str] = []
        all_metadatas: List[StoredMetadata] = []

        for file_name, file_bytes, content_type in items:
            try:
                text = self.text_extractor.extract_text(
                    file_name=file_name,
                    file_bytes=file_bytes,
                    content_type=content_type,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to extract text from %s: %s", file_name, exc)
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                continue

            start_chunk_id = len(all_chunks)
            for offset, chunk in enumerate(chunks):
                chunk_id = start_chunk_id + offset
                all_chunks.append(chunk)
                all_metadatas.append(
                    StoredMetadata(
                        text=chunk,
                        document_name=file_name,
                        chunk_id=chunk_id,
                    )
                )

            document_names.append(file_name)

        if not all_chunks:
            return document_names, 0

        embeddings = self.embedding_service.embed_texts(all_chunks)
        added = self.vector_store.add(embeddings, all_metadatas)
        logger.info("Indexed %d chunks from %d documents.", added, len(document_names))

        return document_names, added
    

    # -------- Question Answering --------

    def answer_question(self, question: str, top_k: int | None = None) -> tuple[str, float, List[RetrievedChunk]]:
        """
        Answer a question using retrieved context and an LLM.

        :returns: (answer, confidence, retrieved_chunks)
        """
        if self.vector_store.is_empty:
            raise ValueError("No documents have been ingested yet.")

        k = top_k or self.settings.top_k

        query_embedding = self.embedding_service.embed_query(question)
        search_results = self.vector_store.search(query_embedding, top_k=k)
        if not search_results:
            return "I could not find any relevant information in the documents.", 0.0, []

        retrieved_chunks: List[RetrievedChunk] = []
        similarity_scores: List[float] = []
        context_parts: List[str] = []

        for meta, score in search_results:
            retrieved_chunks.append(
                RetrievedChunk(
                    document_name=meta.document_name,
                    chunk_id=meta.chunk_id,
                    text=meta.text,
                    score=score,
                )
            )
            similarity_scores.append(score)
            context_parts.append(f"[{meta.document_name} - chunk {meta.chunk_id}]\n{meta.text}")

        context = "\n\n".join(context_parts)
         # Call LLM or fallback
        answer = self._call_llm(question=question, retrieved_chunks=retrieved_chunks)
        confidence = float(sum(similarity_scores) / len(similarity_scores))

        return answer, confidence, retrieved_chunks
    


    def _call_llm(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """
        Call the configured LLM with the retrieved chunks.
        Fallback to a summarized context-only answer if quota is exceeded or client is not configured.
        """
        if self.client is None:
            # No API key: summarize retrieved chunks
            summary = self._summarize_chunks(retrieved_chunks)
            return (
                "LLM model is not configured (no OPENAI_API_KEY set). "
                "Here is a concise summary from the documents:\n\n"
                f"{summary}"
            )

        # Prepare full context
        context = "\n\n".join([f"[{c.document_name} - chunk {c.chunk_id}]\n{c.text}" for c in retrieved_chunks])
        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on the provided context. "
            "If the context does not contain the answer, say you do not know. "
            "Do not hallucinate or use external knowledge."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            logger.warning("OpenAI quota exceeded. Falling back to summarized context-only answer.")
            summary = self._summarize_chunks(retrieved_chunks)
            return f"⚠️ OpenAI quota exceeded. Here is a concise summary:\n\n{summary}"

        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            summary = self._summarize_chunks(retrieved_chunks)
            return f"⚠️ LLM call failed. Returning summarized context:\n\n{summary}"


    def _summarize_chunks(
        self,
        chunks: List[RetrievedChunk],
        max_chunks: int = 3,
        max_lines: int = 5
    ) -> str:
        """
        Improved fallback summarizer:
        - Normalizes text before deduplication
        - Removes near-duplicate overlapping chunks
        - Removes repeated paragraphs inside a chunk
        - Limits chunks and lines cleanly
        """

        seen_signatures = set()
        formatted_chunks = []

        for chunk in chunks:
            # 1️⃣ Normalize text (remove extra spaces + lowercase)
            normalized = re.sub(r"\s+", " ", chunk.text.strip().lower())

            # 2️⃣ Use prefix-based signature to remove overlapping duplicates
            signature = normalized[:500]  # first 500 chars enough for similarity

            if signature in seen_signatures:
                continue

            seen_signatures.add(signature)

            # 3️⃣ Remove repeated paragraphs inside same chunk
            paragraphs = []
            seen_paragraphs = set()

            for para in chunk.text.strip().split("\n"):
                cleaned_para = para.strip()
                if cleaned_para and cleaned_para not in seen_paragraphs:
                    paragraphs.append(cleaned_para)
                    seen_paragraphs.add(cleaned_para)

            snippet = "\n".join(paragraphs[:max_lines])

            formatted_chunks.append(
                f"[Chunk {chunk.chunk_id}] {chunk.document_name}\n{snippet} ..."
            )

            if len(formatted_chunks) >= max_chunks:
                break

        if not formatted_chunks:
            return "No relevant text could be summarized from the documents."

        return "\n\n".join(formatted_chunks)



@lru_cache()
def get_rag_service() -> "RAGService":
    """
    Lazily create a singleton RAGService instance.
    """
    return RAGService()


