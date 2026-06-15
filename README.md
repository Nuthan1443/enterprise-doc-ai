# Enterprise Document AI

A FastAPI-based RAG application for ingesting business documents, scrubbing PII, embedding content into ChromaDB, and answering questions with a Groq-backed LLM.

## What this project does

This service provides:

- Document ingestion for PDF, DOCX, TXT, and CSV files
- PII detection and redaction using Presidio
- Chunking and embeddings for semantic retrieval
- A retrieval-augmented generation (RAG) question-answer flow
- Semantic caching for repeated queries
- Health, cache, and LangSmith tracing endpoints
- Optional RAGAS evaluation support for quality checks

## Architecture at a glance

- API: FastAPI
- LLM: Groq via LangChain
- Embeddings: sentence-transformers
- Vector store: ChromaDB
- PII scrubbing: Presidio + spaCy
- Evaluation: RAGAS

## Prerequisites

- Python 3.11+
- A Groq API key
- Optional: LangSmith API key for tracing

## Quick start

1. Create and activate a virtual environment

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Create a local environment file

   Create a `.env` file in the project root with values such as:

   ```env
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL_NAME=llama-3.3-70b-versatile
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   CHROMA_PERSIST_DIR=./chroma_db
   CHROMA_COLLECTION_NAME=enterprise_docs
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_key
   LANGCHAIN_PROJECT=enterprise-doc-ai
   APP_ENV=development
   LOG_LEVEL=INFO
   MAX_UPLOAD_SIZE_MB=50
   ```

4. Download the spaCy model used by Presidio

   ```bash
   python -m spacy download en_core_web_lg
   ```

5. Start the API server

   ```bash
   uvicorn app.main:app --reload
   ```

6. Open the docs

   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Example API calls

### Health check

```bash
curl http://localhost:8000/health
```

### Ingest a document

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/sample_docs/test.txt"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the company leave policy?"}'
```

### View cache stats

```bash
curl http://localhost:8000/cache/stats
```

### Clear cache

```bash
curl -X DELETE http://localhost:8000/cache
```

## Docker

Run the service in Docker with the included compose file:

```bash
docker compose up --build
```

The app will be available at:

- http://localhost:8000

## Sample data

A sample PDF generator is included under `data/sample_docs/`.

To generate the example HR policy PDF used in tests:

```bash
python data/sample_docs/create_sample_pdf.py
```

## Running tests

```bash
pytest
```

## Notes

- The application uses ChromaDB persistence in `./chroma_db` by default.
- PII scrubbing is applied before chunking and embedding to reduce sensitive data exposure.
- The semantic cache is useful for repeated prompts and can be cleared via the `/cache` endpoint.
