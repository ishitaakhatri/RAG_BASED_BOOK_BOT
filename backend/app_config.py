import os
from pydantic import BaseModel, Field
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

# Ensure env vars are loaded before config is instantiated
load_dotenv()

class LLMConfig(BaseModel):
    # Model Settings
    model_name: str = Field(default="models/gemma-3-27b-it") # Or os.getenv("LLM_MODEL_NAME") if you want that dynamic too
    temperature: float = 0.7
    
    # Provider keys
    # FIX: Use default_factory with os.getenv
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))

class VectorDBConfig(BaseModel):
    # FIX: Use default_factory with os.getenv
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    index_name: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "coding-books"))
    namespace: str = Field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "books_rag"))
    metadata_namespace: str = "books_metadata"
    
    embedding_model: str = Field(default="BAAI/bge-m3")
    dimension: int = 1024

class IngestionConfig(BaseModel):
    # Chunking
    similarity_threshold: float = 0.75
    min_chunk_size: int = 200
    max_chunk_size: int = 1000
    overlap: int = 100
    
    # Processing
    batch_size: int = 32
    use_grobid: bool = True
    # FIX: Use default_factory with os.getenv
    grobid_url: str = Field(default_factory=lambda: os.getenv("GROBID_URL", "http://localhost:8070/api"))
    grobid_timeout: int = Field(default=300)

class RetrievalConfig(BaseModel):
    # Pipeline Pass Settings
    pass1_top_k: int = 50        
    pass2_top_k: int = 15        
    pass3_enabled: bool = True   
    
    # Reranker
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_score_threshold: float = 0.3 
    
    # Context Window
    target_context_tokens: int = 22000 
    max_context_tokens: int = 30000
    
    # Deduplication
    semantic_dedup_threshold: float = 0.92

class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # Global App Settings
    log_level: str = "INFO"
    environment: str = Field(default_factory=lambda: os.getenv("APP_ENV", "production"))

@lru_cache()
def get_config():
    return AppConfig()