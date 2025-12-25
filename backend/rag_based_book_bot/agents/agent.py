"""
RAG Agent orchestration using LangGraph.
Focuses solely on the Query Pipeline.
"""

from typing import Optional, Dict, Any, List
from rag_based_book_bot.agents.states import AgentState
from rag_based_book_bot.agents.graph import build_query_graph

class RAGAgent:
    """
    Main RAG Agent that orchestrates the pipeline using LangGraph.
    """
    
    def __init__(self):
        # Build compiled graph (Query only)
        self.query_app = build_query_graph()
    
    def query(self, 
              user_query: str, 
              history: List[dict] = None, 
              **config) -> Dict[str, Any]:
        """
        Processes a user query using the query graph.
        """
        # Prepare initial state
        inputs = AgentState(
            user_query=user_query,
            conversation_history=history or [],
            **config
        )
        
        print(f"Invoking Query Graph for: {user_query}")
        
        # Invoke LangGraph
        final_state = self.query_app.invoke(inputs)
        
        return self._format_response(final_state)
    
    def _format_response(self, state: Dict) -> dict:
        """Formats the output dictionary (State) into API response format."""
        resp = state.get("response")
        
        if not resp:
            return {"error": "No response generated", "errors": state.get("errors", [])}
        
        # Helper to handle dict or object access
        def get_val(obj, key, default=None):
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        answer = get_val(resp, "answer")
        code_snippets = get_val(resp, "code_snippets", [])
        confidence = get_val(resp, "confidence", 0.0)
            
        return {
            "answer": answer,
            "code_snippets": code_snippets,
            "sources": [
                {
                    "chunk_id": rc["chunk"]["chunk_id"],
                    "relevance": rc.get("relevance_percentage", 0),
                    "book": rc["chunk"].get("book_title", ""),
                    "chapter": rc["chunk"].get("chapter", ""),
                    "type": rc["chunk"].get("chunk_type", "text")
                }
                for rc in state.get("reranked_chunks", [])
            ],
            "confidence": confidence,
            "intent": state["parsed_query"].intent if state.get("parsed_query") else "unknown",
            "stats": {
                 "steps": len(state.get("pipeline_snapshots", [])),
                 "snapshots": state.get("pipeline_snapshots", [])
            }
        }
        
    def get_graph_visualization(self):
        """Returns ASCII visualization of the graph"""
        try:
            return self.query_app.get_graph().draw_ascii()
        except Exception as e:
            return f"Visualization unavailable: {e}"

def create_agent() -> RAGAgent:
    return RAGAgent()