import logging
from fastapi import APIRouter, HTTPException
from app.services.rag_chain import RAGChain
from app.services.cache import SemanticCache
from app.schemas.query_schema import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

rag_chain = RAGChain()
semantic_cache = SemanticCache()


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG pipeline with a natural language question.

    Checks semantic cache first — returns cached response if a
    similar question was recently answered. Falls through to the
    full RAG chain on cache miss.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # 1. Check cache first
        cached = semantic_cache.get(request.question)
        if cached:
            logger.info("Returning cached response")
            return QueryResponse(**cached)

        # 2. Cache miss — run full RAG chain
        result = await rag_chain.ainvoke(request.question)

        # 3. Store in cache for future use
        semantic_cache.set(request.question, result)

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            cached=False,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))