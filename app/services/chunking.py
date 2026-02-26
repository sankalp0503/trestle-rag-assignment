from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]

