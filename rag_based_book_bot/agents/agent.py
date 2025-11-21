"""
RAG Agent orchestration using graph-based execution.
"""

from typing import Optional

from states import AgentState
from config import PipelineConfig, get_default_config
from graph import (
    Graph, ExecutionResult,
    build_indexing_graph, build_query_graph
)
from utils import setup_logger, validate_pdf_path, validate_query


class RAGAgent:
    """
    Main RAG Agent that orchestrates the pipeline using graph execution.
    
    Pipeline flow:
    1. PDF Loading (one-time, during indexing)
    2. Chunking & Embedding (one-time, during indexing)
    3. User Query Parsing (per query)
    4. Vector Search (per query)
    5. Reranking (per query)
    6. Context Assembly (per query)
    7. LLM Reasoning (per query)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()
        self.state = AgentState()
        self.logger = setup_logger(self.config.logging)
        
        # Build graphs
        self.indexing_graph = build_indexing_graph()
        self.query_graph = build_query_graph()
        
        self._indexed = False
    
    def index_document(self, pdf_path: str) -> ExecutionResult:
        """
        Indexes a document using the indexing graph (nodes 1-2).
        """
        self.logger.info(f"Starting document indexing: {pdf_path}")
        
        # Validate
        valid, error = validate_pdf_path(pdf_path)
        if not valid:
            self.logger.error(f"Validation failed: {error}")
            return ExecutionResult(
                success=False, final_state=self.state,
                error_message=error
            )
        
        # Set PDF path and execute indexing graph
        self.state.pdf_path = pdf_path
        self.indexing_graph.reset()
        
        result = self.indexing_graph.execute(self.state)
        
        if result.success:
            self._indexed = True
            self.state = result.final_state
            self.logger.info(f"Indexing complete. {len(self.state.chunks)} chunks created.")
        else:
            self.logger.error(f"Indexing failed at {result.failed_node}: {result.error_message}")
        
        return result
    
    def query(self, user_query: str) -> dict:
        """
        Processes a user query using the query graph (nodes 3-7).
        """
        if not self._indexed:
            return {"error": "No document indexed. Call index_document() first."}
        
        # Validate
        valid, error = validate_query(user_query)
        if not valid:
            return {"error": error}
        
        self.logger.info(f"Processing query: {user_query[:50]}...")
        
        # Reset query-specific state (preserve indexed data)
        self.state.user_query = user_query
        self.state.parsed_query = None
        self.state.retrieved_chunks = []
        self.state.reranked_chunks = []
        self.state.assembled_context = ""
        self.state.response = None
        self.state.errors = []
        
        # Execute query graph
        self.query_graph.reset()
        result = self.query_graph.execute(self.state)
        
        if result.success:
            self.state = result.final_state
            self.logger.info("Query processing complete.")
            return self._format_response()
        else:
            self.logger.error(f"Query failed at {result.failed_node}: {result.error_message}")
            return {"error": result.error_message, "failed_node": result.failed_node}
    
    def _format_response(self) -> dict:
        """Formats the response for output."""
        resp = self.state.response
        return {
            "answer": resp.answer,
            "code_snippets": resp.code_snippets,
            "sources": [
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "chapter": rc.chunk.chapter,
                    "relevance": rc.relevance_percentage,
                    "type": rc.chunk.chunk_type
                }
                for rc in self.state.reranked_chunks
            ],
            "confidence": resp.confidence,
            "query_info": {
                "intent": self.state.parsed_query.intent.value,
                "topics": self.state.parsed_query.topics,
                "keywords": self.state.parsed_query.keywords
            }
        }
    
    def get_stats(self) -> dict:
        """Returns statistics about the indexed document."""
        return {
            "indexed": self._indexed,
            "total_pages": self.state.total_pages,
            "total_chunks": len(self.state.chunks),
            "code_chunks": sum(1 for c in self.state.chunks if c.chunk_type == "code"),
            "text_chunks": sum(1 for c in self.state.chunks if c.chunk_type == "text")
        }
    
    def get_graph_visualization(self, graph_type: str = "query") -> str:
        """Returns text visualization of specified graph."""
        if graph_type == "indexing":
            return self.indexing_graph.visualize()
        return self.query_graph.visualize()


def create_agent(pdf_path: str, config: Optional[PipelineConfig] = None) -> RAGAgent:
    """Creates and initializes a RAG agent with a document."""
    agent = RAGAgent(config)
    agent.index_document(pdf_path)
    return agent


# =============================================================================
# CLI / EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize agent
    config = get_default_config()
    agent = RAGAgent(config)
    
    # Show graph structure
    print("\n" + "=" * 60)
    print("INDEXING GRAPH:")
    print(agent.get_graph_visualization("indexing"))
    print("\n" + "=" * 60)
    print("QUERY GRAPH:")
    print(agent.get_graph_visualization("query"))
    print("=" * 60 + "\n")
    
    # Index document
    result = agent.index_document("hands_on_ml.pdf")
    
    if result.success:
        print("\nStats:", agent.get_stats())
        
        # Test queries
        test_queries = [
            "How do I implement gradient descent in Python?",
            "What is the difference between CNN and RNN?",
            "Explain neural networks for a beginner"
        ]
        
        for q in test_queries:
            print("\n" + "=" * 60)
            print(f"QUERY: {q}")
            print("=" * 60)
            response = agent.query(q)
            
            if "error" not in response:
                print(f"\nAnswer: {response['answer'][:300]}...")
                print(f"\nConfidence: {response['confidence']}")
                print(f"Intent: {response['query_info']['intent']}")
                print(f"Sources: {len(response['sources'])} chunks")
            else:
                print(f"Error: {response['error']}")