"""
Updated Graph with Full 5-Pass Retrieval Pipeline and LangSmith Tracing
"""

from typing import Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from langsmith import traceable, get_current_run_tree  # UPDATED

from .states import AgentState  # Existing import

# NEW IMPORTS: Memory nodes
from rag_based_book_bot.agents.memory_nodes import (
    query_context_resolution_node,
    conversation_search_node,
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
        """
        Executes the graph with proper LangSmith trace hierarchy.
        
        Each node execution will be properly nested under this execution.
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
            
            node.status = NodeStatus.RUNNING
            
            try:
                # Execute node with explicit tracing context
                print(f"\n🔄 Executing node: {current}")
                state = self._execute_node_traced(node, state)
                
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
    
    @traceable(run_type="chain")
    def _execute_node_traced(self, node: Node, state: AgentState) -> AgentState:
        """
        Execute a single node with tracing.
        
        This wrapper ensures each node appears as a child in the trace tree.
        """
        # Add node metadata to trace
        run = get_current_run_tree()
        if run:
            run.name = f"Node: {node.name}"
            run.metadata.update({
                "node_name": node.name,
                "node_description": node.description,
                "node_status": node.status.value,
                "graph_name": self.name
            })
        
        # Execute the actual node function (which has its own @traceable)
        return node.func(state)
    
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


# ============================================================================
# UPDATED GRAPH BUILDERS WITH 5-PASS PIPELINE AND TRACING
# ============================================================================

def build_indexing_graph() -> Graph:
    """Builds the graph for document indexing."""
    from .nodes import pdf_loader_node, chunking_embedding_node
    
    graph = Graph(name="indexing_pipeline")
    
    graph.add_node("pdf_loader", pdf_loader_node, "Load and extract PDF content")
    graph.add_node("chunking_embedding", chunking_embedding_node, "Chunk text and generate embeddings")
    
    graph.add_edge("pdf_loader", "chunking_embedding")
    
    graph.set_entry_point("pdf_loader")
    graph.set_end_point("chunking_embedding")
    
    return graph


@traceable(name="build_query_graph", run_type="chain")
def build_query_graph() -> Graph:
    """
    Builds the FULL query graph with CONVERSATION MEMORY support and LangSmith tracing
    
    Pipeline with Memory:
    1. Query Parser
    2. Context Resolution (NEW) - Resolve pronouns, detect if needs retrieval
    3A. Answer from History (NEW) - If can answer from memory
    3B. Query Rewriter - If needs retrieval
    4. Vector Search (Pass 1: Coarse) - Uses resolved standalone query
    5. Cross-Encoder Reranking (Pass 2: Precision)
    6. Multi-Hop Expansion (Pass 3: Cross-chapter)
    7. Context Compression & Assembly (Pass 5: Token management)
    8. LLM Reasoning (Final answer) - Includes referenced turn context
    
    Conditional Branching:
    - After context_resolution, if needs_retrieval=False → answer_from_history
    - After context_resolution, if needs_retrieval=True → query_rewriter → retrieval pipeline
    """
    from .nodes import (
        user_query_node, query_rewriter_node, vector_search_node, reranking_node,
        multi_hop_expansion_node, cluster_expansion_node,
        context_assembly_node, llm_reasoning_node
    )
    
    graph = Graph(name="rag_with_conversation_memory")
    
    # ============================================================
    # Add all nodes
    # ============================================================
    
    # Stage 1: Query Understanding
    graph.add_node("query_parser", user_query_node, 
                   "Parse user query and detect intent")
    
    # Stage 2: Context Resolution (NEW)
    graph.add_node("context_resolution", query_context_resolution_node,
                   "Resolve query using conversation history")
    
    # Stage 3A: Answer from Memory (NEW - conditional)
    graph.add_node("answer_from_history", answer_from_history_node,
                   "Answer directly from conversation history")
    
    # Stage 3B: Query Rewriting (for retrieval path)
    graph.add_node("query_rewriter", query_rewriter_node,
                   "Generate alternative query formulations")
    
    # Stage 4-8: Retrieval Pipeline
    graph.add_node("vector_search", vector_search_node, 
                   "PASS 1: Coarse vector search")
    graph.add_node("cross_encoder_reranking", reranking_node, 
                   "PASS 2: Cross-encoder reranking")
    graph.add_node("multi_hop_expansion", multi_hop_expansion_node, 
                   "PASS 3: Multi-hop retrieval")
    graph.add_node("cluster_expansion", cluster_expansion_node, 
                   "PASS 4: Cluster-based expansion")
    graph.add_node("context_compression", context_assembly_node, 
                   "PASS 5: Context assembly")
    graph.add_node("llm_reasoning", llm_reasoning_node, 
                   "Generate final answer with LLM")
    
    # ============================================================
    # Connect nodes with edges
    # ============================================================
    
    # Always start with query parser
    graph.add_edge("query_parser", "context_resolution")
    
    # ============================================================
    # CONDITIONAL BRANCHING based on needs_retrieval
    # ============================================================
    
    # Path A: Can answer from history (skip retrieval)
    graph.add_edge(
        "context_resolution", 
        "answer_from_history",
        condition=lambda s: not s.needs_retrieval
    )
    
    # Path B: Need retrieval (go to query rewriter)
    graph.add_edge(
        "context_resolution",
        "query_rewriter",
        condition=lambda s: s.needs_retrieval
    )
    
    # ============================================================
    # Retrieval Pipeline (Path B continues)
    # ============================================================
    graph.add_edge("query_rewriter", "vector_search")
    graph.add_edge("vector_search", "cross_encoder_reranking")
    graph.add_edge("cross_encoder_reranking", "multi_hop_expansion")
    graph.add_edge("multi_hop_expansion", "cluster_expansion")
    graph.add_edge("cluster_expansion", "context_compression")
    graph.add_edge("context_compression", "llm_reasoning")
    
    # ============================================================
    # Set entry and end points
    # ============================================================
    graph.set_entry_point("query_parser")
    
    # Two possible end points:
    graph.set_end_point("llm_reasoning")      # End after retrieval + LLM
    graph.set_end_point("answer_from_history") # End after answering from memory
    
    return graph


def build_full_graph() -> Graph:
    """Builds the complete RAG pipeline (indexing + query)."""
    from .nodes import (
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
