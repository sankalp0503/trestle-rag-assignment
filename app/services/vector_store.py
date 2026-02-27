import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple
import faiss
import numpy as np
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class StoredMetadata:
    text: str
    document_name: str
    chunk_id: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredMetadata":
        return cls(
            text=data["text"],
            document_name=data["document_name"],
            chunk_id=data["chunk_id"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "document_name": self.document_name,
            "chunk_id": self.chunk_id,
        }


class FaissVectorStore:
    """
    Simple FAISS-based vector store with JSON metadata persistence.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.index: faiss.Index | None = None
        self.metadata: List[StoredMetadata] = []
        self._lock = Lock()
        self._load()

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0
    

    def _load(self) -> None:
        """ Load the FAISS index and metadata from disk, if they exist."""
        index_path = self.settings.faiss_index_path
        metadata_path = self.settings.metadata_path

        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with metadata_path.open("r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                self.metadata = [StoredMetadata.from_dict(m) for m in raw_meta]
                logger.info(
                    "Loaded FAISS index with %d vectors from %s",
                    self.index.ntotal,
                    index_path,
                )
            except Exception as exc: 
                logger.exception("Failed to load vector store: %s", exc)
                self.index = None
                self.metadata = []


    def _save(self) -> None:
        """ Save the FAISS index and metadata to disk."""
        if self.index is None:
            return

        index_path = self.settings.faiss_index_path
        metadata_path = self.settings.metadata_path
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.metadata], f, ensure_ascii=False, indent=2)


    def add(self, embeddings: np.ndarray, metadatas: List[StoredMetadata]) -> int:
        """
        Add embedding vectors and associated metadata to the store.
        Returns the number of vectors added.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadata entries.")

        with self._lock:
            num_vectors, dim = embeddings.shape
            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)
            elif self.index.d != dim:
                raise ValueError(f"Incompatible embedding dimension: expected {self.index.d}, got {dim}.")

            self.index.add(embeddings.astype("float32"))
            self.metadata.extend(metadatas)
            self._save()

        return num_vectors
    

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[StoredMetadata, float]]:
        """
        Search the index with a query embedding.
        Returns a list of (metadata, similarity_score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = max(1, min(top_k, self.index.ntotal))

        with self._lock:
            distances, indices = self.index.search(query_embedding.astype("float32"), top_k)

        idxs = indices[0]
        dists = distances[0]

        results: List[Tuple[StoredMetadata, float]] = []
        for idx, dist in zip(idxs, dists):
            if idx < 0 or idx >= len(self.metadata):
                continue
            # Convert L2 distance to a bounded similarity score in (0, 1]
            similarity = 1.0 / (1.0 + float(dist))
            results.append((self.metadata[idx], similarity))

        return results

