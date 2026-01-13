import os
from functools import lru_cache
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, BaseModel

class AppSettings(BaseSettings):
    # --- General ---
    APP_ENV: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # --- AWS (Storage) ---
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS Access Key")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS Secret Key")
    AWS_DEFAULT_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str

    # --- Redis (Broker & State) ---
    REDIS_URL: str = "redis://redis:6379/0"

    # --- Database ---
    DATABASE_URL: str

    # --- AI / RAG ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "coding-books"
    PINECONE_NAMESPACE: str = "books_rag"
    
    # --- Ingestion ---
    GROBID_URL: str = "http://grobid:8070/api"
    
    # --- Settings Config (Pydantic V2) ---
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

# --- Sub-config objects for internal logic ---
class LLMConfig(BaseModel):
    model_name: str = "models/gemma-3-27b-it"
    temperature: float = 0.7
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))

class VectorDBConfig(BaseModel):
    # Adapter to match old structure if needed
    api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    index_name: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "coding-books"))
    namespace: str = Field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "books_rag"))
    metadata_namespace: str = "books_metadata"
    embedding_model: str = "BAAI/bge-m3"
    dimension: int = 1024

class IngestionConfig(BaseModel):
    similarity_threshold: float = 0.75
    min_chunk_size: int = 200
    max_chunk_size: int = 1000
    overlap: int = 100
    batch_size: int = 32
    use_grobid: bool = True
    grobid_url: str = Field(default_factory=lambda: os.getenv("GROBID_URL", "http://grobid:8070/api"))
    grobid_timeout: int = 300

class RetrievalConfig(BaseModel):
    pass1_top_k: int = 50        
    pass2_top_k: int = 15        
    pass3_enabled: bool = True   
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_score_threshold: float = 0.3 
    target_context_tokens: int = 22000 
    max_context_tokens: int = 30000
    semantic_dedup_threshold: float = 0.92

class AppConfig(BaseModel):
    # Wrapper to maintain backward compatibility with existing code structure
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # Add access to the modern settings
    settings: AppSettings = Field(default_factory=AppSettings)
    
    @property
    def log_level(self):
        return self.settings.LOG_LEVEL
        
    @property
    def environment(self):
        return self.settings.APP_ENV

@lru_cache()
def get_config():
    return AppConfig()

@lru_cache()
def get_settings():
    return AppSettings()