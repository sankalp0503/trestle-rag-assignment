from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # OpenAI / LLM configuration
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    llm_model: str = Field("gpt-4.1-mini", env="LLM_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # Chunking / retrieval configuration
    chunk_size: int = Field(1000, description="Default character-level chunk size.")
    chunk_overlap: int = Field(200, description="Default character overlap between chunks.")
    top_k: int = Field(5, description="Default number of chunks to retrieve for a query.")

    # Vector store / persistence
    data_dir: Path = Field(Path("data"), description="Base directory for vector store files.")
    faiss_index_file: str = Field("faiss.index", description="Filename for FAISS index.")
    metadata_file: str = Field("metadata.json", description="Filename for stored metadata.")

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / self.faiss_index_file

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / self.metadata_file


@lru_cache()
def get_settings() -> Settings:
    """
    Cached application settings loaded from environment / .env.
    """
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings

