import pytest
from app.services.ingestion import DocumentIngestionService
from pathlib import Path
from app.services.pii_scrubber import PIIScrubber
from app.services.chunker import ChunkingService

service = DocumentIngestionService()
sample_dir = Path("data/sample_docs")
sample_dir.mkdir(parents=True, exist_ok=True)
scrubber = PIIScrubber()
chunker = ChunkingService()


def test_txt_ingestion():
    test_file = sample_dir / "test.txt"
    test_file.write_text("This is a test document.\nIt has two lines.")
    result = service.ingest(test_file)
    assert result.metadata["extension"] == ".txt"
    assert "test" in result.content.lower()


def test_unsupported_format_rejected():
    test_file = sample_dir / "test.xyz"
    test_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported format"):
        service.ingest(test_file)


def test_empty_file_rejected():
    test_file = sample_dir / "empty.txt"
    test_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        service.ingest(test_file)


def test_pdf_ingestion():
    pdf_path = sample_dir / "hr_policy.pdf"
    if not pdf_path.exists():
        pytest.skip("hr_policy.pdf not found — run create_sample_pdf.py first")
    result = service.ingest(pdf_path)
    assert result.metadata["extension"] == ".pdf"
    assert "leave" in result.content.lower()
    assert "john smith" in result.content.lower()
    assert "@acmecorp.com" in result.content.lower()

def test_pii_scrubbing_detects_email():
    text = "Contact john.smith@acmecorp.com for details."
    scrubbed, entities = scrubber.scrub(text)
    assert "john.smith@acmecorp.com" not in scrubbed
    assert "<EMAIL_ADDRESS>" in scrubbed
    assert any(e["entity_type"] == "EMAIL_ADDRESS" for e in entities)


def test_pii_scrubbing_detects_person():
    text = "John Smith submitted the request."
    scrubbed, entities = scrubber.scrub(text)
    assert "John Smith" not in scrubbed
    assert any(e["entity_type"] == "PERSON" for e in entities)

def test_pii_scrubbing_preserves_non_pii():
    text = "Employees are entitled to 18 days of annual leave."
    scrubbed, entities = scrubber.scrub(text)
    assert "annual leave" in scrubbed
    assert "Employees" in scrubbed
    # DATE_TIME is intentionally excluded — duration text is not PII
    assert "EMAIL_ADDRESS" not in [e["entity_type"] for e in entities]


def test_pii_scrubbing_full_pdf_content():
    """
    End-to-end: ingest the HR policy PDF then scrub PII.
    Verifies the two modules work together correctly.
    """
    from app.services.ingestion import DocumentIngestionService
    from pathlib import Path

    pdf_path = Path("data/sample_docs/hr_policy.pdf")
    if not pdf_path.exists():
        pytest.skip("hr_policy.pdf not found")

    svc = DocumentIngestionService()
    doc = svc.ingest(pdf_path)

    scrubbed, entities = scrubber.scrub(doc.content)

    # PII must be gone
    assert "John Smith" not in scrubbed
    assert "john.smith@acmecorp.com" not in scrubbed
    assert "+91-9876543210" not in scrubbed

    # Policy content must survive
    assert "18 days" in scrubbed
    assert "annual leave" in scrubbed.lower()

    print(f"\n[PII] Entities found: {len(entities)}")
    for e in entities:
        print(f"  {e['entity_type']:20} | score: {e['score']} | value: {e['original_value']}")

    print(f"\n[SCRUBBED] Preview:\n{scrubbed[:600]}")


def test_chunking_produces_chunks():
    from pathlib import Path
    pdf_path = Path("data/sample_docs/hr_policy.pdf")
    if not pdf_path.exists():
        pytest.skip("hr_policy.pdf not found")

    from app.services.ingestion import DocumentIngestionService
    svc = DocumentIngestionService()
    doc = svc.ingest(pdf_path)
    scrubbed, _ = scrubber.scrub(doc.content)
    chunks = chunker.chunk(doc, scrubbed)

    assert len(chunks) > 0
    print(f"\n[CHUNKS] Stats: {chunker.get_chunk_stats(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk.content)} chars | "
              f"metadata keys: {list(chunk.metadata.keys())}")
        print(f"    Preview: {chunk.content[:80]}...")


def test_chunking_preserves_metadata():
    from pathlib import Path
    from app.services.ingestion import DocumentIngestionService

    svc = DocumentIngestionService()
    doc = svc.ingest(Path("data/sample_docs/test.txt"))
    scrubbed, _ = scrubber.scrub(doc.content)
    chunks = chunker.chunk(doc, scrubbed)

    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "extension" in chunk.metadata


def test_chunking_overlap_creates_continuity():
    """
    Verifies that overlap causes boundary content to appear
    in adjacent chunks — the core purpose of overlap.
    """
    long_text = " ".join([f"Sentence number {i} contains important policy information." 
                          for i in range(50)])
    from app.services.ingestion import IngestedDocument
    doc = IngestedDocument(
        content=long_text,
        metadata={"source": "test.txt", "extension": ".txt", "file_size_bytes": 0}
    )
    chunks = chunker.chunk(doc, long_text)

    assert len(chunks) > 1
    # Verify overlap: end of chunk N should appear in start of chunk N+1
    if len(chunks) >= 2:
        end_of_first = chunks[0].content[-100:]
        start_of_second = chunks[1].content[:200]
        # Some content from end of first chunk must appear in second
        words_from_first = set(end_of_first.split()[-5:])
        words_in_second = set(start_of_second.split())
        overlap_found = bool(words_from_first & words_in_second)
        assert overlap_found, "No overlap detected between adjacent chunks"
        print(f"\n[OVERLAP] Confirmed overlap between chunk 0 and chunk 1")


from app.services.embedder import EmbeddingService


def test_embedding_stores_chunks():
    """
    Full pipeline test: ingest → scrub → chunk → embed
    Verifies chunks are stored in ChromaDB and retrievable.
    """
    from pathlib import Path
    from app.services.ingestion import DocumentIngestionService

    pdf_path = Path("data/sample_docs/hr_policy.pdf")
    if not pdf_path.exists():
        pytest.skip("hr_policy.pdf not found")

    # Full pipeline
    svc = DocumentIngestionService()
    doc = svc.ingest(pdf_path)
    scrubbed, _ = scrubber.scrub(doc.content)
    chunks = chunker.chunk(doc, scrubbed)

    # Embed
    embedding_svc = EmbeddingService()
    ids = embedding_svc.embed_chunks(chunks)

    assert len(ids) == len(chunks)
    print(f"\n[EMBED] Stored {len(ids)} chunks")
    print(f"[EMBED] IDs: {ids}")

    # Verify stored in ChromaDB
    stats = embedding_svc.get_collection_stats()
    print(f"[EMBED] Collection stats: {stats}")
    assert stats["total_documents"] >= len(chunks)


def test_embedding_semantic_search():
    """
    Verifies that a semantic query retrieves the correct chunk.
    This is the core value of the embedding layer.
    """
    embedding_svc = EmbeddingService()

    results = embedding_svc.vector_store.similarity_search(
        query="How many days of annual leave do employees get?",
        k=2,
    )

    assert len(results) > 0
    # The leave policy chunk must be in top results
    top_result = results[0].page_content
    print(f"\n[SEARCH] Top result: {top_result[:200]}")
    assert "leave" in top_result.lower() or "days" in top_result.lower()


from app.services.retriever import RetrieverService


def test_mmr_retrieval_returns_chunks():
    retriever = RetrieverService(k=2, fetch_k=4, lambda_mult=0.5)
    docs = retriever.retrieve("How many days of annual leave do employees get?")

    assert len(docs) > 0
    assert len(docs) <= 2
    top = docs[0].page_content
    assert "leave" in top.lower() or "days" in top.lower()
    print(f"\n[MMR] Retrieved {len(docs)} chunks")
    for i, doc in enumerate(docs):
        print(f"  Doc {i}: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")


def test_mmr_retrieval_with_scores():
    retriever = RetrieverService(k=2, fetch_k=4, lambda_mult=0.5)
    results = retriever.retrieve_with_scores(
        "What is the remote work policy?"
    )

    assert len(results) > 0
    print(f"\n[MMR SCORES]")
    for doc, score in results:
        print(f"  Score: {score:.4f} | {doc.page_content[:80]}...")