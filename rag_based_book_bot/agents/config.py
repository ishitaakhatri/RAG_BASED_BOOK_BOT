"""
Configuration settings for the RAG pipeline.
Centralized config for easy tuning and environment-based overrides.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    use_gpu: bool = False


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    preserve_code_blocks: bool = True
    min_chunk_size: int = 100


@dataclass
class RetrievalConfig:
    """Configuration for vector search and reranking."""
    vector_search_top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.3
    
    # Reranking weights
    similarity_weight: float = 0.4
    topic_match_boost: float = 0.2
    keyword_match_boost: float = 0.05
    code_chunk_boost: float = 0.25
    text_chunk_boost: float = 0.1


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.api_key:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    enabled: bool = True
    level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "rag_agent.log"


@dataclass
class PipelineConfig:
    """Master configuration combining all configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    fail_fast: bool = True
    enable_caching: bool = False


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_default_config() -> PipelineConfig:
    """Returns default configuration."""
    return PipelineConfig()


def get_dev_config() -> PipelineConfig:
    """Configuration optimized for development/testing."""
    return PipelineConfig(
        embedding=EmbeddingConfig(use_gpu=False),
        chunking=ChunkingConfig(max_chunk_size=300),
        retrieval=RetrievalConfig(vector_search_top_k=5, rerank_top_k=3),
        llm=LLMConfig(temperature=0.5, max_tokens=1024),
        logging=LoggingConfig(enabled=True, level="DEBUG")
    )


def get_production_config() -> PipelineConfig:
    """Configuration optimized for production."""
    return PipelineConfig(
        embedding=EmbeddingConfig(use_gpu=True, batch_size=64),
        chunking=ChunkingConfig(max_chunk_size=600, chunk_overlap=100),
        retrieval=RetrievalConfig(vector_search_top_k=15, rerank_top_k=7),
        llm=LLMConfig(temperature=0.3, max_tokens=4096),
        logging=LoggingConfig(enabled=True, level="WARNING", log_to_file=True)
    )


def get_fast_config() -> PipelineConfig:
    """Configuration optimized for speed over quality."""
    return PipelineConfig(
        embedding=EmbeddingConfig(batch_size=64),
        chunking=ChunkingConfig(max_chunk_size=800),
        retrieval=RetrievalConfig(vector_search_top_k=5, rerank_top_k=3),
        llm=LLMConfig(max_tokens=512),
        logging=LoggingConfig(enabled=False)
    )