import os
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
    
    # Database
    DATABASE_URL: str = "sqlite:///./knowledge_companion.db"
    
    # ML Models
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    QA_MODEL: str = "google/flan-t5-base"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Streamlit Settings
    STREAMLIT_PORT: int = 8501
    
    # Vector DB Settings
    FAISS_INDEX_PATH: str = "faiss_index"
    
    class Config:
        env_file = ".env"

# Create directories
settings = Settings()
for dir_path in [settings.DATA_DIR, settings.MODELS_DIR, settings.UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)
