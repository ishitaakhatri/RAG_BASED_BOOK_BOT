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
    - Decide whether retrieval is required (SEPARATE from context needs)
    """

    # 🔥 NEW: Check if already resolved to prevent re-execution
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
        context += f"User: {turn.user_query}\n"
        assistant_text = str(turn.assistant_response or "")
        context += f"Assistant: {assistant_text}\n\n"

    # 🔥 NEW IMPROVED PROMPT - Two separate decisions
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

Examples:
- "how can i implement it" → needs_context=true (resolve "it"), needs_retrieval=true (wants implementation)
- "explain that in simpler terms" → needs_context=true, needs_retrieval=false (already explained)
- "what did you mean by X?" → needs_context=true, needs_retrieval=false (clarification)
- "show me code examples for transformers" → needs_context=false, needs_retrieval=true (new info)
- "can you elaborate on that?" → needs_context=true, needs_retrieval=false (elaboration)

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
        text = response.content.strip()

        # Clean JSON extraction
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        analysis = json.loads(text)

        needs_context = analysis.get("needs_context", False)
        needs_retrieval = analysis.get("needs_retrieval", True)  # 🔥 Default to True for safety
        context_type = analysis.get("context_type", "new_question")
        standalone_query = analysis.get("standalone_query", current_query)
        reason = analysis.get("reason", "")

        print(f"  → needs_context: {needs_context}")
        print(f"  → needs_retrieval: {needs_retrieval}")
        print(f"  → context_type: {context_type}")
        print(f"  → reason: {reason}")

        return {
            "resolved_query": standalone_query,
            "needs_context": needs_context,
            "needs_retrieval": needs_retrieval,  # 🔥 NOW INDEPENDENT
            "context_type": context_type,
            "current_node": "context_resolution",
            "skipped": False
        }

    except Exception as e:
        print(f"  ⚠️ Context resolution failed: {e}")
        import traceback
        print(f"  📍 Traceback:\n{traceback.format_exc()}")
        
        # Fail-safe: When in doubt, retrieve
        return {
            "resolved_query": current_query,
            "needs_context": False,
            "needs_retrieval": True,  # 🔥 Safe default
            "context_type": "error",
            "current_node": "context_resolution",
            "error": str(e)
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
    With idempotency check to prevent duplicate executions
    
    Returns dict with:
    - response: LLMResponse with answer from history
    - pipeline_snapshots: Updated snapshots
    - current_node: Node identifier
    """
    
    # 🔥 NEW: Idempotency check - prevent re-execution
    if state.get("response") is not None:
        print(f"\n[Answer from History] ⏭️ Response already exists, skipping re-execution")
        return {
            "response": state.get("response"),
            "pipeline_snapshots": state.get("pipeline_snapshots", []),
            "current_node": "answer_from_history",
            "skipped": True
        }
    
    # 🔥 NEW: Check if this node was already executed
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    if any(snap.get("stage") == "answer_from_history" for snap in pipeline_snapshots):
        print(f"\n[Answer from History] ⏭️ Already executed, skipping")
        return {
            "pipeline_snapshots": pipeline_snapshots,
            "current_node": "answer_from_history",
            "skipped": True
        }
    
    conversation_history = state.get("conversation_history", [])
    resolved_query = state.get("resolved_query") or state.get("user_query")
    referenced_turn = state.get("referenced_turn")
    
    if not conversation_history:
        print(f"\n[Answer from History] ❌ No conversation history available")
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
            for i, turn in enumerate(conversation_history[-6:], 1):
                history_context += f"**Turn {i}:**\n"
                history_context += f"Q: {turn.user_query}\n"
                history_context += f"A: {turn.assistant_response}\n\n"
            print(f"  → Using last 3 turns")
        
        # 🔥 NEW: Add execution marker to prevent race conditions
        print(f"  🔄 Invoking LLM for history-based answer...")
        
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
        
        print(f"  ✅ LLM response received")
        
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
            "answered_from_memory": True,
            "timestamp": time.time()  # 🔥 NEW: Add timestamp for tracking
        }
        
        print(f"  ✅ Answer generated from conversation history")
        print(f"     Answer length: {len(response.content)} characters")
        
        return {
            "response": llm_response,
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot],
            "current_node": "answer_from_history",
            "skipped": False
        }
        
    except Exception as e:
        print(f"  ❌ Failed to answer from history: {e}")
        import traceback
        print(f"  📍 Traceback:\n{traceback.format_exc()}")
        return {
            "errors": state.get("errors", []) + [f"Failed to answer from history: {str(e)}"],
            "current_node": "answer_from_history"
        }