import logging
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.ingestion import DocumentIngestionService
from app.services.pii_scrubber import PIIScrubber
from app.services.chunker import ChunkingService
from app.services.embedder import EmbeddingService
from app.schemas.ingest_schema import IngestResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Services instantiated once at module level
# In production: use FastAPI dependency injection for testability
ingestion_svc = DocumentIngestionService()
pii_scrubber = PIIScrubber()
chunker = ChunkingService()
embedding_svc = EmbeddingService()


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the RAG pipeline.

    Pipeline: parse → PII scrub → chunk → embed → store in ChromaDB

    Supported formats: PDF, DOCX, TXT, CSV
    """
    # Validate file extension before doing any work
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in DocumentIngestionService.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: '{extension}'. "
                   f"Supported: {DocumentIngestionService.SUPPORTED_EXTENSIONS}"
        )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. "
                   f"Maximum: {settings.max_upload_size_mb}MB"
        )

    # Write to temp file — ingestion services expect a file path
    # tempfile ensures cleanup even if an exception occurs
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Stage 1: Parse
        logger.info(f"Ingesting: {file.filename}")
        document = ingestion_svc.ingest(tmp_path)
        document.metadata["source"] = file.filename  # Use original filename

        # Stage 2: PII scrub
        scrubbed_text, detected_entities = pii_scrubber.scrub(document.content)
        logger.info(f"PII scrubbed: {len(detected_entities)} entities removed")

        # Stage 3: Chunk
        chunks = chunker.chunk(document, scrubbed_text)
        logger.info(f"Chunked into {len(chunks)} chunks")

        # Stage 4: Embed and store
        ids = embedding_svc.embed_chunks(chunks)
        logger.info(f"Embedded and stored {len(ids)} chunks")

        return IngestResponse(
            message=f"Successfully ingested '{file.filename}'",
            source=file.filename,
            file_size_bytes=len(contents),
            pii_entities_found=len(detected_entities),
            chunks_created=len(chunks),
            collection_name=settings.chroma_collection_name,
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file
        os.unlink(tmp_path)