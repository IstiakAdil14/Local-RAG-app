import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env (or .env.example fallback)
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_file = project_root / ".env" if (project_root / ".env").exists() else project_root / ".env.example"
load_dotenv(dotenv_path=env_file)

class Settings:
    PROJECT_ROOT: Path = project_root
    
    # Qdrant Database Settings
    # If QDRANT_URL is set (e.g., "http://localhost:6333" or "https://xyz.cloud.qdrant.io:6333"),
    # vector_search will connect to the remote/Docker Qdrant DB server.
    # If QDRANT_URL is empty or None, it will fall back to local disk storage.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "").strip()
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "").strip() or None
    QDRANT_STORAGE_PATH: str = os.getenv("QDRANT_STORAGE_PATH", "./data/qdrant_db")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "production_coll")
    
    # BM25 Storage Settings
    BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./data/production_bm25.pkl")

    # RAG Model Settings
    GENERATOR_MODEL_ID: str = os.getenv("GENERATOR_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    RERANKER_MODEL_ID: str = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-base")

settings = Settings()
