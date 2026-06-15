import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.services.embedder import get_embedding_model, get_vector_store
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrieverService:
    """
    Retrieves relevant chunks from ChromaDB using MMR.

    MMR (Maximal Marginal Relevance) balances:
    - Relevance: how similar is the chunk to the query?
    - Diversity: how different is the chunk from already-selected chunks?

    This prevents returning near-duplicate chunks when multiple
    similar chunks exist for the same topic.
    """

    def __init__(
        self,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        """
        Args:
            k: Number of chunks to return to the LLM
            fetch_k: Number of candidates to retrieve before MMR re-ranking
                     Must be > k. Higher = better diversity candidates.
            lambda_mult: MMR balance parameter (0=diversity, 1=similarity)
        """
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

        embedding_model = get_embedding_model()
        self.vector_store = get_vector_store(embedding_model)

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult,
            },
        )

        logger.info(
            f"RetrieverService initialized: "
            f"k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}"
        )

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieves top-k diverse, relevant chunks for a query.

        Returns:
            List of LangChain Document objects with page_content and metadata.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Retrieving chunks for query: '{query[:80]}...'")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks")
        return docs

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """
        Returns chunks with their similarity scores.
        Useful for debugging retrieval quality and RAGAS evaluation.
        Note: scores are from initial similarity search, before MMR re-ranking.
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=self.k,
        )