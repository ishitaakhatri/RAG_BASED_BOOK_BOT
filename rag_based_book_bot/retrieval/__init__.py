"""
5-Pass Retrieval System for RAG Book Bot

This module provides advanced multi-pass retrieval capabilities:
- Pass 1: Coarse semantic search
- Pass 2: Cross-encoder reranking
- Pass 3: Multi-hop query expansion
- Pass 4: Graph/cluster expansion
- Pass 5: Compression and deduplication
"""

from .five_pass_retrieval import (
    FivePassRetriever,
    RetrievalConfig,
    RetrievalResult
)

from .integration_module import (
    query_with_five_pass,
    get_retriever,
    integrate_five_pass_retrieval
)

try:
    from .concept_graph import (
        ConceptGraphBuilder,
        ConceptNode,
        ConceptEdge
    )
    HAS_CONCEPT_GRAPH = True
except ImportError:
    HAS_CONCEPT_GRAPH = False

__version__ = "1.0.0"

__all__ = [
    "FivePassRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "query_with_five_pass",
    "get_retriever",
    "integrate_five_pass_retrieval",
    "ConceptGraphBuilder",
    "ConceptNode",
    "ConceptEdge",
    "HAS_CONCEPT_GRAPH"
]