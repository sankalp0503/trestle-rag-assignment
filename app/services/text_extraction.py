import io
import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional
from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, TextLoader


logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Handles text extraction from different document types:
    - PDF (via LangChain `PyPDFLoader`)
    - Plain text / Markdown (via LangChain `TextLoader`)
    - Images (via Tesseract OCR)
    """

    def extract_text(
        self,
        file_name: str,
        file_bytes: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        logger.info("Extracting text from file: %s", file_name)
        suffix = Path(file_name).suffix.lower()
        guessed_type, _ = mimetypes.guess_type(file_name)
        content_type = content_type or guessed_type or ""

        if suffix == ".pdf" or content_type == "application/pdf":
            text = self._extract_from_pdf(file_bytes)
        elif suffix in {".txt", ".md"} or content_type.startswith("text/"):
            text = self._extract_from_text(file_bytes, suffix=suffix)
        elif content_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg"}:
            text = self._extract_from_image(file_bytes)
        else:
            logger.warning("Unsupported file type for %s (content_type=%s)", file_name, content_type)
            raise ValueError(f"Unsupported file type: {file_name}")

        return text
    

    @staticmethod
    def _extract_from_pdf(file_bytes: bytes) -> str:
        """
        Use LangChain's PyPDFLoader on a temporary file.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
        return "\n".join(doc.page_content for doc in docs).strip()
    

    @staticmethod
    def _extract_from_text(file_bytes: bytes, suffix: str) -> str:
        """
        Use LangChain's TextLoader on a temporary file for .txt / .md and
        other textual content.
        """
        if suffix not in {".txt", ".md"}:
            suffix = ".txt"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            loader = TextLoader(tmp.name, encoding="utf-8")
            docs = loader.load()
        return "\n".join(doc.page_content for doc in docs).strip()
    

    @staticmethod
    def _extract_from_image(file_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()

