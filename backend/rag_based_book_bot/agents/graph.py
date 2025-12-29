"""
Updated Graph with Full 5-Pass Retrieval Pipeline
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional


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

# NEW IMPORTS: Memory nodes
from .memory_nodes import (
    query_context_resolution_node,
    answer_from_history_node
)



class NodeStatus(Enum):
    """Status of node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Node:
    """Represents a node in the graph."""
    name: str
    func: Callable[[AgentState], AgentState]
    description: str = ""
    status: NodeStatus = NodeStatus.PENDING


@dataclass
class Edge:
    """Represents an edge between nodes."""
    from_node: str
    to_node: str
    condition: Optional[Callable[[AgentState], bool]] = None


@dataclass 
class ExecutionResult:
    """Result of graph execution."""
    success: bool
    final_state: AgentState
    executed_nodes: list[str] = field(default_factory=list)
    failed_node: Optional[str] = None
    error_message: Optional[str] = None


class Graph:
    """Directed graph for pipeline execution."""
    
    def __init__(self, name: str = "rag_pipeline"):
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: Optional[str] = None
        self.end_points: set[str] = set()
    
    def add_node(self, name: str, func: Callable, description: str = "") -> "Graph":
        """Adds a node to the graph."""
        self.nodes[name] = Node(name=name, func=func, description=description)
        return self
    
    def add_edge(self, from_node: str, to_node: str, 
                 condition: Optional[Callable[[AgentState], bool]] = None) -> "Graph":
        """Adds an edge between nodes."""
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' not found")
        if to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' not found")
        
        self.edges.append(Edge(from_node=from_node, to_node=to_node, condition=condition))
        return self
    
    def set_entry_point(self, node_name: str) -> "Graph":
        """Sets the starting node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        self.entry_point = node_name
        return self
    
    def set_end_point(self, node_name: str) -> "Graph":
        """Marks a node as an end point."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        self.end_points.add(node_name)
        return self
    
    def get_next_nodes(self, current: str, state: AgentState) -> list[str]:
        """Gets next nodes based on edges and conditions."""
        next_nodes = []
        for edge in self.edges:
            if edge.from_node == current:
                if edge.condition is None or edge.condition(state):
                    next_nodes.append(edge.to_node)
        return next_nodes
    
    def execute(self, state: AgentState, start_from: Optional[str] = None) -> ExecutionResult:
        """Executes the graph."""
        current = start_from or self.entry_point
        if not current:
            return ExecutionResult(
                success=False, final_state=state,
                error_message="No entry point defined"
            )
        
        executed = []
        
        while current:
            node = self.nodes.get(current)
            if not node:
                return ExecutionResult(
                    success=False, final_state=state, executed_nodes=executed,
                    failed_node=current, error_message=f"Node '{current}' not found"
                )
            
            node.status = NodeStatus.RUNNING
            try:
                state = node.func(state)
                
                if state.errors:
                    node.status = NodeStatus.FAILED
                    return ExecutionResult(
                        success=False, final_state=state, executed_nodes=executed,
                        failed_node=current, error_message=str(state.errors)
                    )
                
                node.status = NodeStatus.COMPLETED
                executed.append(current)
                
            except Exception as e:
                node.status = NodeStatus.FAILED
                return ExecutionResult(
                    success=False, final_state=state, executed_nodes=executed,
                    failed_node=current, error_message=str(e)
                )
            
            if current in self.end_points:
                break
            
            next_nodes = self.get_next_nodes(current, state)
            current = next_nodes[0] if next_nodes else None
        
        return ExecutionResult(success=True, final_state=state, executed_nodes=executed)
    
    def reset(self):
        """Resets all node statuses."""
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
    
    def visualize(self) -> str:
        """Returns a text visualization of the graph."""
        lines = [f"Graph: {self.name}", "=" * 60]
        
        for node_name, node in self.nodes.items():
            marker = "→" if node_name == self.entry_point else " "
            end_marker = "◉" if node_name in self.end_points else " "
            lines.append(f"{marker} [{node.status.value:^10}] {node_name} {end_marker}")
            
            for edge in self.edges:
                if edge.from_node == node_name:
                    cond = " (conditional)" if edge.condition else ""
                    lines.append(f"      ↓ {cond}")
                    lines.append(f"      → {edge.to_node}")
        
        return "\n".join(lines)
    

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


# ============================================================================
# UPDATED GRAPH BUILDERS WITH 5-PASS PIPELINE
# ============================================================================

def build_indexing_graph():
    """
    Build graph for document indexing.
    
    Note: This is a placeholder. Document ingestion is handled
    by enhanced_ingestion.py, not through the graph pipeline.
    """
    from .nodes import pdf_loader_node, chunking_embedding_node
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("pdf_loader", pdf_loader_node)
    workflow.add_node("chunking_embedding", chunking_embedding_node)
    
    workflow.add_edge("pdf_loader", "chunking_embedding")
    workflow.add_edge("chunking_embedding", END)
    
    workflow.set_entry_point("pdf_loader")
    
    return workflow.compile()


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
    
    # ========================================================================
    # ADD ALL NODES
    # ========================================================================
    
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
    
    # ========================================================================
    # DEFINE EDGES
    # ========================================================================
    
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
    
    # ========================================================================
    # SET ENTRY AND EXIT POINTS
    # ========================================================================
    
    # Entry point
    workflow.set_entry_point("query_parser")
    
    # Terminal nodes (two possible endpoints)
    workflow.add_edge("llm_reasoning", END)
    workflow.add_edge("answer_from_history", END)
    
    # ========================================================================
    # COMPILE GRAPH
    # ========================================================================
    
    if enable_persistence:
        # Use MemorySaver for session persistence
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        print("✅ LangGraph compiled WITH persistence")
    else:
        app = workflow.compile()
        print("✅ LangGraph compiled WITHOUT persistence")
    
    return app


def build_full_graph() -> Graph:
    """Builds the complete RAG pipeline (indexing + query)."""
    from nodes import (
        pdf_loader_node, chunking_embedding_node, user_query_node,
        vector_search_node, reranking_node, multi_hop_expansion_node,
        cluster_expansion_node, context_assembly_node, llm_reasoning_node
    )
    
    graph = Graph(name="full_5_pass_rag")
    
    # Indexing nodes
    graph.add_node("pdf_loader", pdf_loader_node)
    graph.add_node("chunking_embedding", chunking_embedding_node)
    
    # Query nodes (5-pass)
    graph.add_node("query_parser", user_query_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("cross_encoder_reranking", reranking_node)
    graph.add_node("multi_hop_expansion", multi_hop_expansion_node)
    graph.add_node("cluster_expansion", cluster_expansion_node)
    graph.add_node("context_compression", context_assembly_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    
    # Edges
    graph.add_edge("pdf_loader", "chunking_embedding")
    graph.add_edge("chunking_embedding", "query_parser")
    graph.add_edge("query_parser", "vector_search")
    graph.add_edge("vector_search", "cross_encoder_reranking")
    graph.add_edge("cross_encoder_reranking", "multi_hop_expansion")
    graph.add_edge("multi_hop_expansion", "cluster_expansion")
    graph.add_edge("cluster_expansion", "context_compression")
    graph.add_edge("context_compression", "llm_reasoning")
    
    graph.set_entry_point("pdf_loader")
    graph.set_end_point("llm_reasoning")
    
    return graph