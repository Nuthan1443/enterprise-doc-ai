import os
from dotenv import load_dotenv
load_dotenv()
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import get_settings
from app.routers import ingest, query


settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting Enterprise Doc AI — env: {settings.app_env}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Enterprise Document AI Assistant",
    description="RAG pipeline with PII scrubbing, semantic caching, and RAGAS evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "env": settings.app_env,
        "model": settings.groq_model_name,
        "embedding_model": settings.embedding_model_name,
    }

from app.services.cache import SemanticCache

_cache = SemanticCache()

@app.get("/cache/stats")
async def cache_stats():
    return _cache.stats()

@app.delete("/cache")
async def clear_cache():
    _cache.clear()
    return {"message": "Cache cleared"}

@app.get("/traces")
async def trace_info():
    """Returns LangSmith project info for observability verification."""
    return {
        "langsmith_project": settings.langchain_project,
        "tracing_enabled": settings.langchain_tracing_v2,
        "trace_url": f"https://api.smith.langchain.com/{settings.langchain_project}",
    }