"""
LangGraph Query Processing Pipeline

Builds the main query processing graph using LangGraph's StateGraph.
This is the core orchestration engine for the RAG pipeline.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .states import AgentState
from .nodes import (
    user_query_node,
    query_rewriter_node,
    vector_search_node,
    reranking_node,
    multi_hop_expansion_node,
    cluster_expansion_node,
    context_assembly_node,
    llm_reasoning_node
)

from .memory_nodes import (
    query_context_resolution_node,
    answer_from_history_node
)
    

def route_after_context_resolution(
    state: AgentState
) -> Literal["answer_from_history", "query_rewriter"]:
    """
    Route based on whether retrieval is needed.
    
    If needs_retrieval=False, answer directly from conversation history.
    If needs_retrieval=True, proceed to retrieval pipeline.
    """
    needs_retrieval = state.get("needs_retrieval", True)
    
    if needs_retrieval:
        print(f"[Router] needs_retrieval=True → Proceeding to retrieval pipeline")
        return "query_rewriter"
    else:
        print(f"[Router] needs_retrieval=False → Answering from history")
        return "answer_from_history"


def build_query_graph(enable_persistence: bool = True):
    """
    Build the full query processing graph with conversation memory support.
    
    Pipeline Flow:
    1. Query Parser → Parse user query
    2. Context Resolution → Resolve pronouns, detect if needs retrieval
    3A. Answer from History (if no retrieval needed)
    3B. Query Rewriter → Retrieval Pipeline (if retrieval needed)
    4. Vector Search (Pass 1)
    5. Cross-Encoder Reranking (Pass 2)
    6. Multi-Hop Expansion (Pass 3)
    7. Cluster Expansion (Pass 4)
    8. Context Assembly (Pass 5)
    9. LLM Reasoning (Final answer)
    
    Args:
        enable_persistence: Whether to enable checkpointing
    
    Returns:
        Compiled LangGraph application
    """
    
    # Create StateGraph
    workflow = StateGraph(AgentState)
    
    print("Building LangGraph pipeline...")
    
    # Stage 1: Query Understanding
    workflow.add_node("query_parser", user_query_node)
    
    # Stage 2: Context Resolution
    workflow.add_node("context_resolution", query_context_resolution_node)
    
    # Stage 3A: Answer from Memory (conditional)
    workflow.add_node("answer_from_history", answer_from_history_node)
    
    # Stage 3B: Query Rewriting (for retrieval path)
    workflow.add_node("query_rewriter", query_rewriter_node)
    
    # Stage 4-9: Retrieval Pipeline
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("cross_encoder_reranking", reranking_node)
    workflow.add_node("multi_hop_expansion", multi_hop_expansion_node)
    workflow.add_node("cluster_expansion", cluster_expansion_node)
    workflow.add_node("context_compression", context_assembly_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    
    # Always start with query parser
    workflow.add_edge("query_parser", "context_resolution")
    
    # CONDITIONAL BRANCHING after context resolution
    workflow.add_conditional_edges(
        "context_resolution",
        route_after_context_resolution,
        {
            "answer_from_history": "answer_from_history",
            "query_rewriter": "query_rewriter"
        }
    )
    
    # Retrieval Pipeline (linear flow)
    workflow.add_edge("query_rewriter", "vector_search")
    workflow.add_edge("vector_search", "cross_encoder_reranking")
    workflow.add_edge("cross_encoder_reranking", "multi_hop_expansion")
    workflow.add_edge("multi_hop_expansion", "cluster_expansion")
    workflow.add_edge("cluster_expansion", "context_compression")
    workflow.add_edge("context_compression", "llm_reasoning")
    
    # Entry point
    workflow.set_entry_point("query_parser")
    
    # Terminal nodes (two possible endpoints)
    workflow.add_edge("llm_reasoning", END)
    workflow.add_edge("answer_from_history", END)
    
    if enable_persistence:
        # Use MemorySaver for session persistence
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        print("✅ LangGraph compiled WITH persistence")
    else:
        app = workflow.compile()
        print("✅ LangGraph compiled WITHOUT persistence")
    
    return app
