from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    groq_model_name: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "enterprise_docs"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "enterprise-doc-ai"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50

    model_config = {"env_file": ".env", "case_sensitive": False}



@lru_cache()
def get_settings() -> Settings:
    return Settings()