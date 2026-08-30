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
import time
import traceback
from rag_based_book_bot.agents.states import AgentState, ConversationTurn, LLMResponse
from rag_based_book_bot.agents.llm_utils import extract_text_from_response
from rag_based_book_bot.memory.conversation_store import search_conversation_context
from app_config import get_config
load_dotenv()
settings = get_config()

# Initialize LLM with correct parameters for modern LangChain
llm = ChatGoogleGenerativeAI(
    model=settings.llm.model_name,
    google_api_key=settings.llm.google_api_key,
    temperature=settings.llm.temperature,
    max_retries=1, 
    convert_system_message_to_human=True 
)

def query_context_resolution_node(state: AgentState) -> Dict:
    """
    LLM-based context resolution node (LangGraph compatible)
    """

    # Check if already resolved to prevent re-execution
    if state.get("resolved_query") and state.get("needs_retrieval") is not None:
        print(f"\n[Context Resolution] ⏭️ Already resolved, skipping")
        return {
            "current_node": "context_resolution",
            "skipped": True
        }

    parsed_query = state.get("parsed_query")
    current_query = (
        parsed_query.raw_query
        if parsed_query
        else state.get("user_query")
    )

    conversation_history = state.get("conversation_history", [])

    # No history → must retrieve
    if not conversation_history:
        print(f"\n[Context Resolution] No conversation history → needs_retrieval=True")
        return {
            "resolved_query": current_query,
            "needs_context": False,
            "needs_retrieval": True,
            "context_type": "none",
            "current_node": "context_resolution",
        }

    # Build compact history (last 5 turns)
    context = ""
    for turn in conversation_history[-5:]:
        # Handle cases where turn objects might be dicts or objects
        u_query = turn.user_query if hasattr(turn, 'user_query') else turn.get('user_query', '')
        a_response = turn.assistant_response if hasattr(turn, 'assistant_response') else turn.get('assistant_response', '')
        
        context += f"User: {u_query}\n"
        context += f"Assistant: {str(a_response)}\n\n"

    prompt = f"""
Conversation context:
{context}

Current user query:
"{current_query}"

Task: Make TWO separate decisions:

1. **Context Resolution**: Does this query contain pronouns or references that need the conversation history to be understood?
   - "it", "that", "this", "the code above", "explain more" → needs_context=true
   - Standalone questions with no references → needs_context=false

2. **Retrieval Decision**: Does this query ask for NEW information that requires searching the knowledge base?
   - Asking for code examples, implementations, tutorials, new concepts → needs_retrieval=true
   - Asking to clarify/elaborate on something ALREADY explained in history → needs_retrieval=false
   - Meta questions about the conversation itself → needs_retrieval=false

Return ONLY valid JSON:
{{
  "needs_context": true/false,
  "standalone_query": "rewritten query with references resolved",
  "needs_retrieval": true/false,
  "context_type": "previous_code | previous_explanation | follow_up | clarification | new_question",
  "reason": "brief explanation of both decisions"
}}
"""

    try:
        print(f"\n[Context Resolution] Analyzing query: '{current_query[:60]}...'")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        text = extract_text_from_response(response.content).strip()

        # Clean JSON extraction
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        analysis = json.loads(text)

        needs_context = analysis.get("needs_context", False)
        needs_retrieval = analysis.get("needs_retrieval", True)
        context_type = analysis.get("context_type", "new_question")
        standalone_query = analysis.get("standalone_query", current_query)
        reason = analysis.get("reason", "")

        return {
            "resolved_query": standalone_query,
            "needs_context": needs_context,
            "needs_retrieval": needs_retrieval,
            "context_type": context_type,
            "current_node": "context_resolution",
            "skipped": False
        }

    except Exception as e:
        print(f"  ⚠️ Context resolution failed: {e}")
        
        # Fail-safe: When in doubt, retrieve
        return {
            "resolved_query": current_query,
            "needs_context": False,
            "needs_retrieval": True,
            "context_type": "error",
            "current_node": "context_resolution",
            "error": str(e)
        }


async def conversation_search_node(state: AgentState) -> Dict:
    """
    Semantic search over conversation history
    """
    conversation_history = state.get("conversation_history", [])
    
    if not conversation_history:
        return {"relevant_past_turns": []}
    
    # Could implement session-based vector search here if needed
    
    return {
        "relevant_past_turns": [],
        "current_node": "conversation_search"
    }


async def answer_from_history_node(state: AgentState) -> Dict:
    """
    Answer directly from conversation history
    """
    
    # Idempotency check
    if state.get("response") is not None:
        return {
            "response": state.get("response"),
            "pipeline_snapshots": state.get("pipeline_snapshots", []),
            "current_node": "answer_from_history",
            "skipped": True
        }
    
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
                
                # Handle dict vs object
                u_query = turn.user_query if hasattr(turn, 'user_query') else turn.get('user_query', '')
                a_response = turn.assistant_response if hasattr(turn, 'assistant_response') else turn.get('assistant_response', '')

                history_context = f"""**Referenced Turn:**
Question: {u_query}
Answer: {a_response}
"""
        else:
            # Use last 6 turns
            for i, turn in enumerate(conversation_history[-6:], 1):
                u_query = turn.user_query if hasattr(turn, 'user_query') else turn.get('user_query', '')
                a_response = turn.assistant_response if hasattr(turn, 'assistant_response') else turn.get('assistant_response', '')
                
                history_context += f"**Turn {i}:**\n"
                history_context += f"Q: {u_query}\n"
                history_context += f"A: {a_response}\n\n"
        
        prompt = f"""{history_context}

**Current Question:** {resolved_query}

Extract or synthesize the answer from the conversation above. 

**Important:**
- Only use information from the conversation history
- Don't make up new information
- If the conversation doesn't fully answer the question, say so
- Explain workflow of code in detail and elaborate as much as possible so user can understand

Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        llm_response = LLMResponse(
            answer=extract_text_from_response(response.content),
            sources=[],
            confidence=0.75,
            code_snippets=[]
        )
        
        new_snapshot = {
            "stage": "answer_from_history",
            "chunk_count": 0,
            "chunks": [],
            "answered_from_memory": True,
            "timestamp": time.time()
        }
        
        return {
            "response": llm_response,
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot],
            "current_node": "answer_from_history",
            "skipped": False
        }
        
    except Exception as e:
        print(f"  ❌ Failed to answer from history: {e}")
        return {
            "errors": state.get("errors", []) + [f"Failed to answer from history: {str(e)}"],
            "current_node": "answer_from_history"
        }