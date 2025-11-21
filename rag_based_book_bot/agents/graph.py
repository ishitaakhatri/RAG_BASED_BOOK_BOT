"""
Graph definition and execution engine for the RAG pipeline.
Defines nodes, edges, and traversal logic.
"""

from typing import Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from states import AgentState


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
    condition: Optional[Callable[[AgentState], bool]] = None  # For conditional routing


@dataclass 
class ExecutionResult:
    """Result of graph execution."""
    success: bool
    final_state: AgentState
    executed_nodes: list[str] = field(default_factory=list)
    failed_node: Optional[str] = None
    error_message: Optional[str] = None


class Graph:
    """
    Directed graph for pipeline execution.
    Supports linear flow and conditional branching.
    """
    
    def __init__(self, name: str = "rag_pipeline"):
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: Optional[str] = None
        self.end_points: set[str] = set()
    
    def add_node(self, name: str, func: Callable, description: str = "") -> "Graph":
        """Adds a node to the graph. Returns self for chaining."""
        self.nodes[name] = Node(name=name, func=func, description=description)
        return self
    
    def add_edge(self, from_node: str, to_node: str, 
                 condition: Optional[Callable[[AgentState], bool]] = None) -> "Graph":
        """Adds an edge between nodes. Returns self for chaining."""
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
        """
        Executes the graph starting from entry point or specified node.
        """
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
            
            # Execute node
            node.status = NodeStatus.RUNNING
            try:
                state = node.func(state)
                
                # Check for errors in state
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
            
            # Check if we've reached an end point
            if current in self.end_points:
                break
            
            # Get next node(s)
            next_nodes = self.get_next_nodes(current, state)
            current = next_nodes[0] if next_nodes else None
        
        return ExecutionResult(success=True, final_state=state, executed_nodes=executed)
    
    def reset(self):
        """Resets all node statuses."""
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
    
    def visualize(self) -> str:
        """Returns a simple text visualization of the graph."""
        lines = [f"Graph: {self.name}", "=" * 40]
        
        for node_name, node in self.nodes.items():
            marker = "→" if node_name == self.entry_point else " "
            end_marker = "◉" if node_name in self.end_points else " "
            lines.append(f"{marker} [{node.status.value:^10}] {node_name} {end_marker}")
            
            # Show outgoing edges
            for edge in self.edges:
                if edge.from_node == node_name:
                    cond = " (conditional)" if edge.condition else ""
                    lines.append(f"      ↓ {cond}")
                    lines.append(f"      → {edge.to_node}")
        
        return "\n".join(lines)


# =============================================================================
# GRAPH BUILDERS
# =============================================================================

def build_indexing_graph() -> Graph:
    """Builds the graph for document indexing (nodes 1-2)."""
    from nodes import pdf_loader_node, chunking_embedding_node
    
    graph = Graph(name="indexing_pipeline")
    
    graph.add_node("pdf_loader", pdf_loader_node, "Load and extract PDF content")
    graph.add_node("chunking_embedding", chunking_embedding_node, "Chunk text and generate embeddings")
    
    graph.add_edge("pdf_loader", "chunking_embedding")
    
    graph.set_entry_point("pdf_loader")
    graph.set_end_point("chunking_embedding")
    
    return graph


def build_query_graph() -> Graph:
    """Builds the graph for query processing (nodes 3-7)."""
    from nodes import (
        user_query_node, vector_search_node, reranking_node,
        context_assembly_node, llm_reasoning_node
    )
    
    graph = Graph(name="query_pipeline")
    
    graph.add_node("query_parser", user_query_node, "Parse user query")
    graph.add_node("vector_search", vector_search_node, "Retrieve relevant chunks")
    graph.add_node("reranking", reranking_node, "Rerank by relevance")
    graph.add_node("context_assembly", context_assembly_node, "Assemble LLM context")
    graph.add_node("llm_reasoning", llm_reasoning_node, "Generate response")
    
    graph.add_edge("query_parser", "vector_search")
    graph.add_edge("vector_search", "reranking")
    graph.add_edge("reranking", "context_assembly")
    graph.add_edge("context_assembly", "llm_reasoning")
    
    graph.set_entry_point("query_parser")
    graph.set_end_point("llm_reasoning")
    
    return graph


def build_full_graph() -> Graph:
    """Builds the complete RAG pipeline graph."""
    from nodes import (
        pdf_loader_node, chunking_embedding_node, user_query_node,
        vector_search_node, reranking_node, context_assembly_node,
        llm_reasoning_node
    )
    
    graph = Graph(name="full_rag_pipeline")
    
    # Add all nodes
    graph.add_node("pdf_loader", pdf_loader_node, "Load and extract PDF")
    graph.add_node("chunking_embedding", chunking_embedding_node, "Chunk and embed")
    graph.add_node("query_parser", user_query_node, "Parse query")
    graph.add_node("vector_search", vector_search_node, "Vector search")
    graph.add_node("reranking", reranking_node, "Rerank results")
    graph.add_node("context_assembly", context_assembly_node, "Build context")
    graph.add_node("llm_reasoning", llm_reasoning_node, "LLM response")
    
    # Linear flow
    graph.add_edge("pdf_loader", "chunking_embedding")
    graph.add_edge("chunking_embedding", "query_parser")
    graph.add_edge("query_parser", "vector_search")
    graph.add_edge("vector_search", "reranking")
    graph.add_edge("reranking", "context_assembly")
    graph.add_edge("context_assembly", "llm_reasoning")
    
    graph.set_entry_point("pdf_loader")
    graph.set_end_point("llm_reasoning")
    
    return graph