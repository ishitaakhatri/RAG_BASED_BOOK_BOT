import os
from pydantic import BaseModel, Field
from functools import lru_cache
from typing import Optional

class LLMConfig(BaseModel):
    # Model Settings
    # Supports Google Gemini models (e.g. gemini-1.5-flash, gemini-1.5-pro)
    model_name: str = Field(default="models/gemini-1.5-flash", env="LLM_MODEL_NAME")
    temperature: float = 0.7
    
    # Provider keys
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")

class VectorDBConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    index_name: str = Field(default="coding-books", env="PINECONE_INDEX_NAME")
    namespace: str = Field(default="books_rag", env="PINECONE_NAMESPACE")
    metadata_namespace: str = "books_metadata"
    
    # Critical: Keep this consistent across ingestion and retrieval
    embedding_model: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL")
    dimension: int = 1024  # BGE-M3 dense dimension

class IngestionConfig(BaseModel):
    # Chunking
    similarity_threshold: float = 0.75
    min_chunk_size: int = 200
    max_chunk_size: int = 1000  # Standardized from 1500/1000 discrepancy
    overlap: int = 100
    
    # Processing
    batch_size: int = 32
    use_grobid: bool = True
    grobid_url: str = Field(default="http://localhost:8070/api", env="GROBID_URL")
    grobid_timeout: int = Field(default=300, env="GROBID_TIMEOUT")

class RetrievalConfig(BaseModel):
    # Pipeline Pass Settings
    pass1_top_k: int = 50        # Vector Search
    pass2_top_k: int = 15        # Reranking
    pass3_enabled: bool = True   # Multi-hop
    
    # Reranker
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_score_threshold: float = 0.3 # If score < 0.3, consider irrelevant
    
    # Context Window (Input to LLM)
    # The compressor uses this to decide how much text fits in the prompt
    target_context_tokens: int = 22000 
    max_context_tokens: int = 30000
    
    # Deduplication
    semantic_dedup_threshold: float = 0.92

class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    ingestion: IngestionConfig = IngestionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    
    # Global App Settings
    log_level: str = "INFO"
    environment: str = Field(default="production", env="APP_ENV")

@lru_cache()
def get_config():
    return AppConfig()