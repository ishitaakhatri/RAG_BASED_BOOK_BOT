"""
Memory and context resolution nodes for the RAG pipeline

These nodes handle:
1. Context resolution - Resolving ambiguous queries using conversation history
2. Answering from history - Responding without retrieval when possible
3. Conversation search - Finding relevant past context

ALL NODES NOW INCLUDE LANGSMITH TRACING
"""

import json
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langsmith import traceable  # ADDED

from rag_based_book_bot.agents.states import AgentState, ConversationTurn, LLMResponse
from rag_based_book_bot.memory.conversation_store import search_conversation_context

load_dotenv()

# Initialize Ollama LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)


@traceable(
    name="query_context_resolution_node",
    run_type="chain",
    metadata={"purpose": "resolve_query_with_history"}
)
def query_context_resolution_node(state: AgentState) -> AgentState:
    """
    LLM-based context resolution node
    
    Analyzes the current query with conversation history to:
    1. Detect if it references previous context (pronouns, implicit references)
    2. Identify which previous turn it references
    3. Determine if it can be answered from history alone
    4. Generate a standalone query that doesn't need conversation context
    
    This is the KEY node that makes follow-up questions work!
    
    Example:
        History: "What is CNN?" → "CNNs are neural networks..."
        Query: "Show me code for it"
        
        → needs_conversation_context: true
        → references_turn: 1
        → can_answer_from_history: false
        → standalone_query: "Show me CNN implementation code"
    """
    state.current_node = "context_resolution"
    
    current_query = state.parsed_query.raw_query if state.parsed_query else state.user_query
    
    # No history? Skip resolution
    if not state.conversation_history:
        print(f"\n[Context Resolution] No conversation history - using query as-is")
        state.resolved_query = current_query
        state.needs_retrieval = True
        state.referenced_turn = None
        return state
    
    try:
        print(f"\n[Context Resolution] Analyzing query with conversation history...")
        print(f"  Current query: '{current_query}'")
        print(f"  History turns: {len(state.conversation_history)}")
        
        # Build conversation context (last 5 turns max for efficiency)
        conversation_context = ""
        recent_history = state.conversation_history[-5:]
        
        for i, turn in enumerate(recent_history, 1):
            conversation_context += f"[Turn {i}]\n"
            conversation_context += f"Q: {turn.user_query}\n"
            # Truncate long responses
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
6. If referring to "the first one", "earlier", "before" → set references_turn to that turn number

**Examples:**

Query: "Show me code for it" (after discussing CNN)
→ {{"needs_conversation_context": true, "references_turn": 1, "can_answer_from_history": false, "standalone_query": "Show me CNN implementation code"}}

Query: "What did you say about advantages?" 
→ {{"needs_conversation_context": true, "references_turn": 2, "can_answer_from_history": true, "standalone_query": "What are the advantages of CNN?"}}

Query: "How do I implement LSTM?"
→ {{"needs_conversation_context": false, "references_turn": null, "can_answer_from_history": false, "standalone_query": "How do I implement LSTM?"}}

Return ONLY valid JSON, no markdown formatting."""

        # Call LLM with tracing
        response = _analyze_context_with_llm(analysis_prompt)
        response_text = response.content.strip()
        
        # Clean up markdown formatting if present
        if response_text.startswith("```json"):
            response_text = (
                response_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        
        # Parse JSON
        analysis = json.loads(response_text)
        
        # Update state with analysis results
        state.resolved_query = analysis.get('standalone_query', current_query)
        state.needs_retrieval = not analysis.get('can_answer_from_history', True)
        state.referenced_turn = analysis.get('references_turn')
        
        # Log results
        print(f"  ✅ Analysis complete:")
        print(f"     Original: '{current_query}'")
        print(f"     Resolved: '{state.resolved_query}'")
        print(f"     Needs retrieval: {state.needs_retrieval}")
        print(f"     References turn: {state.referenced_turn}")
        print(f"     Reasoning: {analysis.get('reasoning', 'N/A')}")
        
    except json.JSONDecodeError as e:
        print(f"  ⚠️ Failed to parse LLM response as JSON: {e}")
        print(f"     Response was: {response_text[:200]}...")
        print(f"  → Using fallback (treat as new query)")
        state.resolved_query = current_query
        state.needs_retrieval = True
        state.referenced_turn = None
        
    except Exception as e:
        print(f"  ⚠️ Context resolution failed: {e}")
        print(f"  → Using fallback (treat as new query)")
        state.resolved_query = current_query
        state.needs_retrieval = True
        state.referenced_turn = None
    
    return state


@traceable(name="analyze_context_with_llm", run_type="llm")
def _analyze_context_with_llm(prompt: str):
    """Helper function to trace LLM call for context analysis"""
    return llm.invoke([HumanMessage(content=prompt)])


@traceable(
    name="conversation_search_node",
    run_type="retriever",
    metadata={"backend": "pinecone", "search_type": "conversation_history"}
)
def conversation_search_node(state: AgentState) -> AgentState:
    """
    Semantic search over conversation history
    
    Uses vector similarity to find relevant past turns.
    This is called BEFORE retrieval to inject relevant conversation context.
    
    Example:
        Query: "What were the advantages you mentioned?"
        → Searches conversation vectors
        → Finds turn where advantages were discussed
        → Stores in state.relevant_past_turns
    """
    state.current_node = "conversation_search"
    
    # Only search if we have history and a session ID
    if not state.conversation_history:
        state.relevant_past_turns = []
        return state
    
    # Get session_id from first turn (all turns have same session_id)
    session_id = None
    if state.conversation_history:
        # Session ID should be passed in state, but we can infer from metadata
        # For now, we'll skip this search if we don't have explicit session tracking
        print(f"\n[Conversation Search] Skipping - no explicit session_id in state")
        state.relevant_past_turns = []
        return state
    
    try:
        print(f"\n[Conversation Search] Searching conversation history...")
        
        query_to_search = state.resolved_query or state.user_query
        
        # Search for relevant past turns
        relevant_turns = search_conversation_context(
            session_id=session_id,
            query=query_to_search,
            top_k=3
        )
        
        # Convert to ConversationTurn objects
        state.relevant_past_turns = [
            ConversationTurn(
                user_query=turn['user_query'],
                assistant_response=turn['assistant_response'],
                timestamp=turn.get('timestamp', 0),
                sources_used=turn.get('sources_used', [])
            )
            for turn in relevant_turns
        ]
        
        print(f"  ✅ Found {len(state.relevant_past_turns)} relevant past turns")
        
    except Exception as e:
        print(f"  ⚠️ Conversation search failed: {e}")
        state.relevant_past_turns = []
    
    return state


@traceable(
    name="answer_from_history_node",
    run_type="llm",
    metadata={"source": "conversation_memory"}
)
def answer_from_history_node(state: AgentState) -> AgentState:
    """
    Answer directly from conversation history without retrieval
    
    Used when context_resolution determines the answer already exists
    in conversation history (can_answer_from_history=True).
    
    Example:
        User: "What did you say about CNN advantages?"
        → LLM extracts answer from previous turn where advantages were discussed
        → No document retrieval needed
    """
    state.current_node = "answer_from_history"
    
    if not state.conversation_history:
        state.errors.append("No conversation history available to answer from")
        return state
    
    try:
        print(f"\n[Answer from History] Generating answer from previous conversation...")
        
        # Build context from conversation history
        history_context = ""
        
        if state.referenced_turn is not None:
            # Use specific referenced turn
            turn_idx = state.referenced_turn - 1
            if 0 <= turn_idx < len(state.conversation_history):
                turn = state.conversation_history[turn_idx]
                history_context = f"""**Referenced Turn:**
Question: {turn.user_query}
Answer: {turn.assistant_response}

"""
                print(f"  → Using referenced turn #{state.referenced_turn}")
        else:
            # Use last 3 turns
            for i, turn in enumerate(state.conversation_history[-3:], 1):
                history_context += f"**Turn {i}:**\n"
                history_context += f"Q: {turn.user_query}\n"
                history_context += f"A: {turn.assistant_response}\n\n"
            print(f"  → Using last 3 turns")
        
        # Generate answer from history using LLM
        prompt = f"""{history_context}

**Current Question:** {state.resolved_query or state.user_query}

Extract or synthesize the answer from the conversation above. 

**Important:**
- Only use information from the conversation history
- Don't make up new information
- If the conversation doesn't fully answer the question, say so
- Be concise and direct

Answer:"""

        response = _generate_answer_from_history(prompt)
        
        state.response = LLMResponse(
            answer=response.content,
            sources=[],  # No document sources, answered from memory
            confidence=0.75,  # Lower confidence for history-based answers
            code_snippets=[]
        )
        
        # Track that we answered from history
        state.pipeline_snapshots.append({
            "stage": "answer_from_history",
            "chunk_count": 0,
            "chunks": [],
            "answered_from_memory": True
        })
        
        print(f"  ✅ Answer generated from conversation history")
        print(f"     Answer length: {len(response.content)} characters")
        
    except Exception as e:
        print(f"  ❌ Failed to answer from history: {e}")
        state.errors.append(f"Failed to answer from history: {str(e)}")
    
    return state


@traceable(name="generate_answer_from_history", run_type="llm")
def _generate_answer_from_history(prompt: str):
    """Helper function to trace LLM call for generating answer from history"""
    return llm.invoke([HumanMessage(content=prompt)])
