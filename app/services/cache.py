import logging
import time
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.services.embedder import get_embedding_model
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CACHE_COLLECTION_NAME = "query_cache"
SIMILARITY_THRESHOLD = 0.2  # cosine distance — lower = more similar


class SemanticCache:
    """
    Semantic similarity cache for RAG query responses.

    Uses a separate ChromaDB collection to store query-answer pairs.
    On cache lookup, embeds the incoming query and checks if any
    stored query is within SIMILARITY_THRESHOLD cosine distance.

    This prevents redundant LLM calls for semantically equivalent questions.
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()

        # Separate ChromaDB client for cache — isolated from document store
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )

        self.collection = self.client.get_or_create_collection(
            name=CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"SemanticCache initialized. "
            f"Cache size: {self.collection.count()} entries"
        )

    def get(self, query: str) -> dict | None:
        """
        Looks up a cached response for a semantically similar query.

        Returns cached result dict if hit, None if miss.
        """
        if self.collection.count() == 0:
            return None

        # Embed the incoming query
        query_embedding = self.embedding_model.embed_query(query)

        # Search for similar cached queries
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        if not results["distances"] or not results["distances"][0]:
            return None

        distance = results["distances"][0][0]
        metadata = results["metadatas"][0][0]

        logger.debug(f"Cache lookup distance: {distance:.4f} (threshold: {SIMILARITY_THRESHOLD})")

        if distance <= SIMILARITY_THRESHOLD:
            logger.info(f"Cache HIT — distance: {distance:.4f}")
            return {
                "question": query,
                "answer": metadata["answer"],
                "sources": metadata["sources"].split(","),
                "chunks_used": int(metadata["chunks_used"]),
                "cached": True,
                "cache_distance": round(distance, 4),
            }

        logger.info(f"Cache MISS — distance: {distance:.4f}")
        return None

    def set(self, query: str, result: dict) -> None:
        """
        Stores a query-answer pair in the cache.

        Metadata stores the answer and sources as strings
        (ChromaDB metadata values must be str/int/float/bool).
        """
        query_embedding = self.embedding_model.embed_query(query)

        # Use timestamp as unique ID
        cache_id = f"cache_{int(time.time() * 1000)}"

        self.collection.add(
            embeddings=[query_embedding],
            ids=[cache_id],
            metadatas=[{
                "query": query,
                "answer": result["answer"],
                "sources": ",".join(result.get("sources", [])),
                "chunks_used": result.get("chunks_used", 0),
            }],
        )

        logger.info(f"Cached response for query: '{query[:60]}...'")

    def clear(self) -> None:
        """Clears all cache entries. Useful for testing."""
        self.client.delete_collection(CACHE_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cache cleared")

    def stats(self) -> dict:
        return {
            "cache_size": self.collection.count(),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "collection_name": CACHE_COLLECTION_NAME,
        }