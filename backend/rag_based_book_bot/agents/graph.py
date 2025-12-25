"""
Graph definition for the RAG pipeline using LangGraph.
Contains only the Query Graph.
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END

from rag_based_book_bot.agents.states import AgentState
from rag_based_book_bot.agents.nodes import (
    query_parser_node,
    context_resolution_node,
    answer_from_history_node,
    query_rewriter_node,
    vector_search_node,
    reranking_node,
    context_assembly_node,
    llm_reasoning_node
)

def check_retrieval_needed(state: AgentState) -> Literal["retrieve", "history"]:
    """Conditional edge: retrieval vs memory"""
    if state.needs_retrieval:
        return "retrieve"
    return "history"

def build_query_graph():
    """
    Builds the main RAG query pipeline.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("query_parser", query_parser_node)
    workflow.add_node("context_resolution", context_resolution_node)
    workflow.add_node("answer_from_history", answer_from_history_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("reranking", reranking_node)
    workflow.add_node("context_assembly", context_assembly_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    
    # Define Edges
    workflow.add_edge(START, "query_parser")
    workflow.add_edge("query_parser", "context_resolution")
    
    workflow.add_conditional_edges(
        "context_resolution",
        check_retrieval_needed,
        {
            "history": "answer_from_history",
            "retrieve": "query_rewriter"
        }
    )
    
    workflow.add_edge("query_rewriter", "vector_search")
    workflow.add_edge("vector_search", "reranking")
    workflow.add_edge("reranking", "context_assembly")
    workflow.add_edge("context_assembly", "llm_reasoning")
    
    workflow.add_edge("answer_from_history", END)
    workflow.add_edge("llm_reasoning", END)
    
    return workflow.compile()