import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.services.chunker import DocumentChunk
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads the HuggingFace embedding model.
    Model downloads on first call (~90MB), then cached locally.
    Uses CPU by default — acceptable for portfolio, use GPU in production.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model_name}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # Required for cosine similarity
            "batch_size": 32,
        },
    )


def get_vector_store(
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """
    Returns a ChromaDB vector store backed by the HuggingFace embedding model.
    Creates the collection if it doesn't exist.
    Persists to disk at settings.chroma_persist_dir.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


class EmbeddingService:
    """
    Embeds DocumentChunks and stores them in ChromaDB.
    Handles batching to avoid memory issues with large documents.
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store(self.embedding_model)
        logger.info("EmbeddingService initialized")

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """
        Embeds a list of DocumentChunks and stores them in ChromaDB.

        Returns:
            List of ChromaDB document IDs for the stored chunks.
        """
        if not chunks:
            raise ValueError("Cannot embed empty chunk list")

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate unique IDs: source_filename + chunk_index
        # This allows re-ingestion to overwrite existing chunks
        # for the same document rather than creating duplicates
        ids = [
            f"{chunk.metadata.get('source', 'unknown')}__chunk_{chunk.chunk_index}"
            for chunk in chunks
        ]

        logger.info(f"Embedding {len(chunks)} chunks into ChromaDB...")

        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Stored {len(chunks)} chunks in collection "
                    f"'{settings.chroma_collection_name}'")
        return ids

    def get_collection_stats(self) -> dict:
        """Returns current state of the ChromaDB collection."""
        collection = self.vector_store._collection
        count = collection.count()
        return {
            "collection_name": settings.chroma_collection_name,
            "total_documents": count,
            "persist_dir": settings.chroma_persist_dir,
        }