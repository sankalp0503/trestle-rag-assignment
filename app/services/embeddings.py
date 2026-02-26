from typing import List

import numpy as np
from openai import OpenAI, RateLimitError
from app.core.config import get_settings


class EmbeddingService:
    """
    Wraps embedding generation using OpenAI embeddings API.

    If no `OPENAI_API_KEY` is configured, falls back to a simple
    deterministic local embedding based on hashing tokens. This allows
    the service to be tested without external dependencies, while still
    exercising the RAG flow end-to-end.
    """

    def __init__(self, client: OpenAI | None = None) -> None:
        settings = get_settings()
        if settings.openai_api_key:
            self.client: OpenAI | None = client or OpenAI(api_key=settings.openai_api_key)
            self.model: str | None = settings.embedding_model
        else:
            # No API key: operate in local-embedding mode.
            self.client = None
            self.model = None
        # Dimension for local hash-based embeddings
        self._local_dim = 384

    def _local_embed(self, texts: List[str]) -> np.ndarray:
        """
        Very simple hash-based embedding function for offline testing.
        """
        dim = self._local_dim
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for tok in tokens:
                h = hash(tok) % dim
                vectors[i, h] += 1.0
        # L2-normalize to ensure embeddings are in vector space
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        return vectors / norms

    

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._local_dim), dtype="float32")

        # If no client → use local
        if self.client is None or self.model is None:
            return self._local_embed(texts)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            vectors = np.array(
                [item.embedding for item in response.data],
                dtype="float32"
            )

            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            # Add logging to verify embeddings
            print(f"Generated embeddings for {len(texts)} texts. First embedding: {vectors[0] if len(vectors) > 0 else 'None'}")
            return vectors / norms

        except RateLimitError as e:
            print("OpenAI quota exceeded. Falling back to local embeddings.")
            return self._local_embed(texts)

        except Exception as e:
            print(f"Embedding API failed: {e}")
            return self._local_embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])

