"""
Memory and context resolution nodes for the RAG pipeline

These nodes handle:
1. Context resolution - Resolving ambiguous queries using conversation history
2. Answering from history - Responding without retrieval when possible
3. Conversation search - Finding relevant past context
"""

import json
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

from rag_based_book_bot.agents.states import AgentState, ConversationTurn, LLMResponse
from rag_based_book_bot.memory.conversation_store import search_conversation_context

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemma-3-27b-it", # Use the official model string (Gemma 3 might require models/ prefix)
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_retries=0, # FIX: Prevents passing the unexpected keyword argument
    convert_system_message_to_human=True # Keep this for Gemma models
)

def query_context_resolution_node(state: AgentState) -> Dict:
    """
    LLM-based context resolution node (LangGraph compatible)
    
    Returns dict with:
    - resolved_query: Standalone query
    - needs_retrieval: Whether retrieval is needed
    - referenced_turn: Which turn was referenced
    - current_node: Node identifier
    """
    
    current_query = state.get("parsed_query").raw_query if state.get("parsed_query") else state.get("user_query")
    conversation_history = state.get("conversation_history", [])
    
    # No history? Skip resolution
    if not conversation_history:
        print(f"\n[Context Resolution] No conversation history - using query as-is")
        return {
            "resolved_query": current_query,
            "needs_retrieval": True,
            "referenced_turn": None,
            "current_node": "context_resolution"
        }
    
    try:
        print(f"\n[Context Resolution] Analyzing query with conversation history...")
        print(f"  Current query: '{current_query}'")
        print(f"  History turns: {len(conversation_history)}")
        
        # Build conversation context
        conversation_context = ""
        recent_history = conversation_history[-5:]
        
        for i, turn in enumerate(recent_history, 1):
            conversation_context += f"[Turn {i}]\n"
            conversation_context += f"Q: {turn.user_query}\n"
            response_preview = turn.assistant_response[:200]
            if len(turn.assistant_response) > 200:
                response_preview += "..."
            conversation_context += f"A: {response_preview}\n\n"
        
        # LLM analysis prompt
        analysis_prompt = f"""{conversation_context}

**Current Query:** "{current_query}"

Analyze this query in the context of the conversation above. Return a JSON response:

{{
  "needs_conversation_context": true/false,
  "references_turn": null or turn_number (1-5),
  "can_answer_from_history": true/false,
  "standalone_query": "rewritten query that doesn't need conversation history",
  "reasoning": "brief explanation of your analysis"
}}

**Rules:**
1. If query uses pronouns (it, this, that, they) or vague references → needs_conversation_context=true
2. If the answer already exists in conversation history → can_answer_from_history=true
3. If new information needed → can_answer_from_history=false
4. Always provide a standalone_query that makes sense without history
5. Detect implicit references (e.g., "what about X?" after discussing Y)

Return ONLY valid JSON, no markdown formatting."""

        # Call LLM
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        response_text = response.content.strip()
        
        # Clean up markdown
        if response_text.startswith("```"):
            response_text = (
                response_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
        
        # Parse JSON
        analysis = json.loads(response_text)
        
        # Extract results
        resolved_query = analysis.get('standalone_query', current_query)
        needs_retrieval = not analysis.get('can_answer_from_history', True)
        referenced_turn = analysis.get('references_turn')
        
        print(f"  ✅ Analysis complete:")
        print(f"     Original: '{current_query}'")
        print(f"     Resolved: '{resolved_query}'")
        print(f"     Needs retrieval: {needs_retrieval}")
        print(f"     References turn: {referenced_turn}")
        print(f"     Reasoning: {analysis.get('reasoning', 'N/A')}")
        
        return {
            "resolved_query": resolved_query,
            "needs_retrieval": needs_retrieval,
            "referenced_turn": referenced_turn,
            "current_node": "context_resolution"
        }
        
    except json.JSONDecodeError as e:
        print(f"  ⚠️ Failed to parse LLM response as JSON: {e}")
        print(f"  → Using fallback (treat as new query)")
        return {
            "resolved_query": current_query,
            "needs_retrieval": True,
            "referenced_turn": None,
            "current_node": "context_resolution"
        }
        
    except Exception as e:
        print(f"  ⚠️ Context resolution failed: {e}")
        print(f"  → Using fallback (treat as new query)")
        return {
            "resolved_query": current_query,
            "needs_retrieval": True,
            "referenced_turn": None,
            "current_node": "context_resolution"
        }

def conversation_search_node(state: AgentState) -> Dict:
    """
    Semantic search over conversation history (LangGraph compatible)
    
    Returns dict with:
    - relevant_past_turns: List of relevant conversation turns
    """
    
    conversation_history = state.get("conversation_history", [])
    
    if not conversation_history:
        return {"relevant_past_turns": []}
    
    # For now, skip this search - would need session_id tracking
    print(f"\n[Conversation Search] Skipping - no explicit session_id in state")
    
    return {
        "relevant_past_turns": [],
        "current_node": "conversation_search"
    }



def answer_from_history_node(state: AgentState) -> Dict:
    """
    Answer directly from conversation history (LangGraph compatible)
    
    Returns dict with:
    - response: LLMResponse with answer from history
    - pipeline_snapshots: Updated snapshots
    - current_node: Node identifier
    """
    
    conversation_history = state.get("conversation_history", [])
    resolved_query = state.get("resolved_query") or state.get("user_query")
    referenced_turn = state.get("referenced_turn")
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    
    if not conversation_history:
        return {
            "errors": state.get("errors", []) + ["No conversation history available"],
            "current_node": "answer_from_history"
        }
    
    try:
        print(f"\n[Answer from History] Generating answer from previous conversation...")
        
        # Build context from conversation history
        history_context = ""
        
        if referenced_turn is not None:
            # Use specific referenced turn
            turn_idx = referenced_turn - 1
            if 0 <= turn_idx < len(conversation_history):
                turn = conversation_history[turn_idx]
                history_context = f"""**Referenced Turn:**
Question: {turn.user_query}
Answer: {turn.assistant_response}

"""
                print(f"  → Using referenced turn #{referenced_turn}")
        else:
            # Use last 3 turns
            for i, turn in enumerate(conversation_history[-3:], 1):
                history_context += f"**Turn {i}:**\n"
                history_context += f"Q: {turn.user_query}\n"
                history_context += f"A: {turn.assistant_response}\n\n"
            print(f"  → Using last 3 turns")
        
        # Generate answer from history
        prompt = f"""{history_context}

**Current Question:** {resolved_query}

Extract or synthesize the answer from the conversation above. 

**Important:**
- Only use information from the conversation history
- Don't make up new information
- If the conversation doesn't fully answer the question, say so
- Be concise and direct

Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Create response object
        llm_response = LLMResponse(
            answer=response.content,
            sources=[],
            confidence=0.75,
            code_snippets=[]
        )
        
        # Add snapshot
        new_snapshot = {
            "stage": "answer_from_history",
            "chunk_count": 0,
            "chunks": [],
            "answered_from_memory": True
        }
        
        print(f"  ✅ Answer generated from conversation history")
        print(f"     Answer length: {len(response.content)} characters")
        
        return {
            "response": llm_response,
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot],
            "current_node": "answer_from_history"
        }
        
    except Exception as e:
        print(f"  ❌ Failed to answer from history: {e}")
        return {
            "errors": state.get("errors", []) + [f"Failed to answer from history: {str(e)}"],
            "current_node": "answer_from_history"
        }
