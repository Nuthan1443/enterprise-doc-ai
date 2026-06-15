import logging
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.ingestion import IngestedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Output contract of the chunking layer.
    Each chunk is one retrievable unit in ChromaDB.
    """
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


class ChunkingService:
    """
    Splits scrubbed document text into overlapping chunks
    using recursive character splitting.

    Strategy: tries to split at paragraph breaks → sentence breaks
    → word breaks → character breaks, in that order.
    This preserves semantic coherence at chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Separator priority: paragraph → sentence → word → character
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"ChunkingService initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk(
        self,
        document: IngestedDocument,
        scrubbed_text: str,
    ) -> list[DocumentChunk]:
        """
        Splits scrubbed text into chunks, preserving document metadata
        in each chunk for downstream retrieval and citation.

        Args:
            document: Original IngestedDocument (for metadata)
            scrubbed_text: PII-scrubbed text from PIIScrubber

        Returns:
            List of DocumentChunk objects ready for embedding
        """
        if not scrubbed_text or not scrubbed_text.strip():
            raise ValueError("Cannot chunk empty or whitespace-only text")

        raw_chunks = self.splitter.split_text(scrubbed_text)

        if not raw_chunks:
            raise ValueError(f"Splitter produced no chunks for: {document.metadata.get('source')}")

        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Skip chunks that are pure whitespace or very short
            # (artifacts from PDF extraction formatting)
            if len(chunk_text.strip()) < 50:
                logger.debug(f"Skipping short chunk {idx}: '{chunk_text[:30]}...'")
                continue

            chunks.append(DocumentChunk(
                content=chunk_text.strip(),
                metadata={
                    **document.metadata,        # source, extension, file_size
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                    "total_chunks": len(raw_chunks),
                },
                chunk_index=idx,
            ))

        logger.info(
            f"Chunked '{document.metadata.get('source')}': "
            f"{len(raw_chunks)} raw → {len(chunks)} valid chunks"
        )

        return chunks

    def get_chunk_stats(self, chunks: list[DocumentChunk]) -> dict:
        """
        Returns statistics about chunk size distribution.
        Useful for tuning chunk_size and overlap parameters.
        """
        if not chunks:
            return {}

        sizes = [len(c.content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes)),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "chunk_size_setting": self.chunk_size,
            "overlap_setting": self.chunk_overlap,
        }