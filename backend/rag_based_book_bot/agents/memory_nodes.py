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

    Responsibility:
    - Decide if the query depends on prior context
    - Rewrite it into a standalone query if needed
    - Decide whether retrieval is required (for routing)
    """

    parsed_query = state.get("parsed_query")
    current_query = (
        parsed_query.raw_query
        if parsed_query
        else state.get("user_query")
    )

    conversation_history = state.get("conversation_history", [])

    # No history → must retrieve
    if not conversation_history:
        return {
            "resolved_query": current_query,
            "needs_context": False,
            "needs_retrieval": True,   # 🔥 IMPORTANT
            "context_type": "none",
            "current_node": "context_resolution",
        }

    # Build compact history (last 5 turns)
    context = ""
    for turn in conversation_history[-5:]:
        context += f"User: {turn.user_query}\n"
        assistant_text = str(turn.assistant_response or "")
        context += f"Assistant: {assistant_text}...\n\n"

    prompt = f"""
Conversation context:
{context}

Current user query:
"{current_query}"

Task:
1. Decide if the query requires previous conversation context to be understood correctly.
2. Rewrite the query so it is fully standalone and unambiguous.

Return ONLY valid JSON:
{{
  "needs_context": true/false,
  "standalone_query": "rewritten query",
  "context_type": "previous_code | previous_explanation | follow_up | none",
  "reason": "short explanation"
}}
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        analysis = json.loads(text)

        needs_context = analysis.get("needs_context", False)
        context_type = analysis.get("context_type", "none")

        return {
            "resolved_query": analysis.get("standalone_query", current_query),
            "needs_context": needs_context,
            "needs_retrieval": not needs_context,  # 🔥 CORE FIX
            "context_type": context_type,
            "current_node": "context_resolution",
        }

    except Exception as e:
        print(f"⚠️ Context resolution failed: {e}")
        return {
            "resolved_query": current_query,
            "needs_context": False,
            "needs_retrieval": True,   # 🔥 fail-safe → retrieve
            "context_type": "none",
            "current_node": "context_resolution",
        }


async def conversation_search_node(state: AgentState) -> Dict:
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



async def answer_from_history_node(state: AgentState) -> Dict:
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
                if hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                    history_context = f"""**Referenced Turn:**
Question: {turn.user_query}
Answer: {turn.assistant_response}

"""
                    print(f"  → Using referenced turn #{referenced_turn}")
                else:
                    print(f"  ⚠️ Referenced turn #{referenced_turn} has missing attributes")
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
