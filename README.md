## RAG Backend Service (AI Engineer Assignment)

This project implements a Retrieval-Augmented Generation (RAG) backend using FastAPI.  
It supports document ingestion (PDF / images / TXT / Markdown), vector indexing with FAISS, and question answering grounded only in the uploaded documents.

---

### 1. Features

- **Document ingestion** via `POST /ingest/`
  - Accepts multiple files (PDF / images / TXT / Markdown)
  - Text extraction using **LangChain** loaders (PDF via `PyPDFLoader`, text/Markdown via `TextLoader`) and **image OCR via Tesseract**
  - Configurable **chunk size** and **overlap**
  - Embedding generation using **OpenAI embeddings** (with a deterministic local fallback when no API key is configured)
  - Storage in a **FAISS** vector store (with JSON metadata)
- **Question answering** via `POST /query/`
  - Embeds query and performs **top-k retrieval**
  - Injects retrieved chunks into LLM prompt
  - Generates an answer using an **OpenAI chat model** (or a context-only fallback if no API key is configured)
  - Returns answer, retrieved chunks, and an approximate **confidence score**
- **FastAPI** auto-generated docs at `/docs` (Swagger) and `/redoc`
- Clean separation into:
  - `core` (config, logging)
  - `services` (text extraction, chunking, embeddings, vector store, RAG orchestration)
  - `routers` (HTTP endpoints)

---

### 2. Project Structure

```text
app/
  __init__.py
  main.py              # FastAPI application factory
  api.py               # Centralized router registration
  core/
    __init__.py
    config.py          # App settings (env-based), paths, defaults
    logging_config.py  # Logging setup
  models/
    __init__.py
    schemas.py         # Pydantic request/response models
  routers/
    __init__.py
    ingestion.py       # /ingest endpoint
    query.py           # /query endpoint
  services/
    __init__.py
    chunking.py        # Text chunking with overlap
    text_extraction.py # Text extraction via LangChain loaders + image OCR
    embeddings.py      # OpenAI embeddings wrapper (with local fallback)
    vector_store.py    # FAISS + metadata persistence
    rag_service.py     # Orchestration service (ingest + QA)

requirements.txt
.env
Dockerfile
README.md
```

### Project Architecture Overview
- main
  - Defines the create_app() factory.
  - Configures logging.
  - Creates the FastAPI instance.

- api
  - Centralizes router registration.
  - Attaches all route modules to the application instance.
- core –> configuration and logging
- models –> Pydantic schemas for request/response validation and serialization
- routers –> HTTP endpoints
- services –> business logic (RAG pipeline components)

---

### 3. Configuration

Configuration is handled via `pydantic` settings in `app/core/config.py`.  
You can use environment variables or a `.env` file in the project root.

**Important environment variables:**

- `OPENAI_API_KEY` (optional but recommended): OpenAI API key.
  - If set, the service uses OpenAI embeddings and an OpenAI chat model.
  - If **not** set, it falls back to deterministic local embeddings and a context-only answer (no external LLM call).
- `LLM_MODEL` (optional, default: `gpt-4.1-mini`): Chat model for answering.
- `EMBEDDING_MODEL` (optional, default: `text-embedding-3-small`): Embedding model.
- `CHUNK_SIZE` (optional, default: `1000`): Default character chunk size.
- `CHUNK_OVERLAP` (optional, default: `200`): Default overlap.
- `TOP_K` (optional, default: `5`): Default number of chunks retrieved for a query.

You can also override data paths via:

- `DATA_DIR` (default: `data`)
- `FAISS_INDEX_FILE` (default: `faiss.index`)
- `METADATA_FILE` (default: `metadata.json`)

Example `.env`:

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

---

### 4. Running Locally (without Docker)

1. **Create a virtual environment** (recommended):

```bash
cd Trestle_labs_assignment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Set environment variables** (or create `.env` as shown above).  
   The service will run even without `OPENAI_API_KEY`, but answers will use the offline fallback described below.

4. **Run the API**:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API documentation (Swagger UI)**
   Open `http://localhost:8000/docs` to view and test the API (Swagger UI).

---

### 5. Running with Docker

1. **Build the image**:

```bash
docker build -t rag-backend .
```

2. **Run the container** (optionally pass your OpenAI API key):

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  rag-backend
```

3. Open `http://localhost:8000/docs` in your browser.

> Note: Tesseract OCR is installed in the image for image-based documents.

---

### 6. API Endpoints

#### 6.1 `POST /ingest/`

**Description**: Ingest one or more documents into the vector store.

- **Method**: `POST`
- **URL**: `/ingest/`
- **Content-Type**: `multipart/form-data`

**Form-data fields:**

- `files`: one or more files (PDF, images, `.txt`, `.md`)
- `chunk_size` (optional, int): override default chunk size
- `chunk_overlap` (optional, int): override default chunk overlap

**Response (201)**:

```json
{
  "documents": ["doc1.pdf", "notes.txt"],
  "chunks_indexed": 42
}
```

If no text can be extracted (e.g. unsupported types), a `400` is returned.

---

#### 6.2 `POST /query/`

**Description**: Ask a question over the ingested corpus.

- **Method**: `POST`
- **URL**: `/query/`
- **Content-Type**: `application/json`

**Request body:**

```json
{
  "question": "What is the refund policy?",
  "top_k": 5
}
```

- `question` (string, required): User question.
- `top_k` (int, optional): Override default number of retrieved chunks.

**Response (200)**:

```json
{
  "answer": "The refund policy states that ...",
  "confidence": 0.83,
  "retrieved_chunks": [
    {
      "document_name": "policy.pdf",
      "chunk_id": 12,
      "text": "Refunds are available within 30 days ...",
      "score": 0.91
    }
  ]
}
```

Possible error responses:

- `400`: if no documents have been ingested yet.
- `500`: on unexpected server error.



###Notes:
- If OpenAI API key is not configured or the API quota is exceeded, the service:
    - Returns a context-only answer based on retrieved chunks.
    - Deduplicates repeated chunks.
    - Shows document name and chunk ID for clarity.
    - Limits the number of lines per chunk for readability.
    - Ensures the API remains functional even offline.

- Possible error responses:
    - 400: if no documents have been ingested yet.
    - 500: on unexpected server error.

---

### 7. Implementation Notes

- **Vector store**: `FaissVectorStore` uses `IndexFlatL2` and stores aligned metadata in a JSON file.
  - Similarity is derived from L2 distance via \( \text{similarity} = \frac{1}{1 + d} \).
  - The **confidence** score in `/query` is the average similarity across the returned chunks.
- **Text extraction**:
  - PDFs: LangChain `PyPDFLoader`
  - Text / Markdown: LangChain `TextLoader`
  - Images: `pytesseract.image_to_string` (requires Tesseract, provided in Docker image)
- **Chunking**: LangChain `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
- **Embeddings**:
  - With `OPENAI_API_KEY`: OpenAI embedding model (default `text-embedding-3-small`).
  - Fail-safe / local fallback:
      -If OPENAI_API_KEY is not set or API quota is exceeded:
        -Uses deterministic local embeddings for queries.
        -Skips the LLM call and returns a summarized context-only answer.
        -Retrieved chunks are deduplicated and truncated for readability.
        -Provides a concise summary instead of repeating entire chunks.
        -Ensures ingestion and query APIs still function end-to-end, allowing offline or limited-quota testing.
- **LLM behavior**:
  - With `OPENAI_API_KEY`: uses the configured chat model, with a system prompt that instructs the model to answer **only from provided context** and say “I don’t know” when the context is insufficient.
  - Without OPENAI_API_KEY or on quota exhaustion: triggers the summarization-based fallback mechanism, providing a natural, readable answer from retrieved chunks.

---

### 8. ngrok Deployment (Optional)

You can expose the API to the internet for testing using ngrok.
**Important behaviors (Free plan)**

  - Every restart of ngrok generates a new public URL.
  - Any previously shared URL will no longer work (404 for users).
  - The first-time user visiting the URL sees a confirmation page:

   `You are about to visit <subdomain>.ngrok-free.dev`
   `[Visit Site]`

  - Clicking Visit Site opens the FastAPI Swagger UI (/docs).

**Usage Steps**
  - Install ngrok:
    `brew install ngrok`
  - Start FastAPI:
    `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
  - Start ngrok:
    `ngrok http 8000`
  - Share the new ngrok URL:
    `https://<random-subdomain>.ngrok-free.dev/docs`

  Free plan: URL changes on each restart.
  Paid plan: Supports reserved subdomains for fixed URLs.

---

### 9. How to Extend

- Swap in a different **vector store** (e.g., Pinecone, Weaviate, Chroma) by implementing a compatible service instead of `FaissVectorStore`.
- Use another **LLM provider** (Gemini, Claude, local model) by replacing the `_call_llm` method in `rag_service.py`.
- Improve **chunking** to be token-based (e.g., with `tiktoken`) instead of simple characters.
- Add authentication, rate limiting, or multi-tenant support as needed.
- Improve the deployment process, e.g., automated Docker builds, CI/CD pipelines, or persistent hosting instead of relying on ngrok for external access.

