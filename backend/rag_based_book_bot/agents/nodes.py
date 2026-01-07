"""
Updated Node implementations with LangChain 0.3, Pinecone v5, and Manual JSON Parsing.
"""

import re
import os
import json
import traceback
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Class based import for Pinecone v5
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent, LLMResponse
)

from rag_based_book_bot.retrieval.retriever import create_retriever
from rag_based_book_bot.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag_based_book_bot.retrieval.multi_hop_expander import MultiHopExpander
from rag_based_book_bot.retrieval.cluster_manager import ClusterManager
from rag_based_book_bot.retrieval.context_compressor import EnhancedContextCompressor

from app_config import get_config

load_dotenv()
settings = get_config()

# ============================================================================
# GLOBAL SINGLETONS & INITIALIZATION
# ============================================================================

_pc = None
_index = None
_model = None
_search_engine = None 
_cross_encoder = None
_multi_hop = None
_cluster_manager = None
_compressor = None

def get_pinecone_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.vector_db.api_key)
        _index = _pc.Index(settings.vector_db.index_name)
    return _index

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.vector_db.embedding_model)
    return _model

def get_search_engine():
    global _search_engine
    if _search_engine is None:
        _search_engine = create_retriever(get_pinecone_index(), get_embedding_model())
    return _search_engine

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker()
    return _cross_encoder

def get_multi_hop_expander():
    global _multi_hop
    if _multi_hop is None:
        _multi_hop = MultiHopExpander()
    return _multi_hop

def get_cluster_manager():
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager(n_clusters=100)
    return _cluster_manager

def get_compressor(target_tokens=None, max_tokens=None):
    t_tokens = target_tokens or settings.retrieval.target_context_tokens
    m_tokens = max_tokens or settings.retrieval.max_context_tokens
    return EnhancedContextCompressor(target_tokens=t_tokens, max_tokens=m_tokens)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.llm.model_name,
    google_api_key=settings.llm.google_api_key,
    temperature=settings.llm.temperature,
    max_retries=1, 
    convert_system_message_to_human=True 
)

# ============================================================================
# NODE 1: RELEVANCE CHECK
# ============================================================================

async def relevance_check_node(state: AgentState) -> Dict:
    user_query = state.get("user_query")
    print(f"\n[RELEVANCE CHECK] Entering node...")
    print(f"[RELEVANCE CHECK] Input query: '{user_query}'")

    if not user_query:
        print(f"[RELEVANCE CHECK] ❌ Empty query detected → intent='reject'")
        return {
            "intent": "reject",
            "current_node": "relevance_check"
        }

    relevance_result = await _check_query_relevance(user_query)
    intent = relevance_result["intent"]
    confidence = relevance_result["confidence"]
    
    print(f"[RELEVANCE CHECK] ✓ Classification complete: intent='{intent}', confidence={confidence:.2f}")

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "current_node": "relevance_check"
    }

async def _check_query_relevance(query: str) -> dict:
    print(f"  [CLASSIFY] Starting LLM classification...")
    
    system_prompt = """
You are an INTENT classifier for a technical book-based assistant.

Classify the user's query into ONE of the following intents:

rag:
- Clear technical or academic questions
- Programming, AI, ML, systems, debugging
- Requests for explanations, concepts, or code

chat:
- Greetings
- Small talk
- Personal statements (e.g., "my name is bob")
- Conversational messages

reject:
- Empty input
- Gibberish
- Meaningless symbols

Return ONLY valid JSON in this exact format:
{
  "intent": "rag" | "chat" | "reject",
  "confidence": 0.0 to 1.0
}
"""
    user_prompt = f'Query: "{query}"'
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        
        intent = result.get("intent", "chat")
        confidence = float(result.get("confidence", 0))
        
        return {
            "intent": intent,
            "confidence": confidence
        }

    except Exception as e:
        print(f"  [CLASSIFY] ❌ Exception: {e}")
        return {
            "intent": "chat",
            "confidence": 0.0
        }

# ============================================================================
# NODE 2: IRRELEVANT HANDLER
# ============================================================================

def irrelevant_query_handler_node(state: AgentState) -> Dict:
    query = state["user_query"]
    print(f"\n[IRRELEVANT HANDLER] Entering node...")
    
    try:
        response = llm.invoke([
            SystemMessage(
                content="You are a friendly conversational assistant. Respond naturally and politely. If the user asks technical questions, politely suggest they ask about programming or technical topics."
            ),
            HumanMessage(content=query)
        ])
        
        answer = response.content
        
        return {
            "response": LLMResponse(
                answer=answer,
                code_snippets=[],
                sources=[],
                confidence=1.0,
                search_summary=f"Chit-chat interaction: {query[:50]}"
            ),
            "current_node": "chat_handler"
        }
    except Exception as e:
        fallback = "I appreciate your message! I'm designed to help with technical questions."
        return {
            "response": LLMResponse(
                answer=fallback,
                code_snippets=[],
                sources=[],
                confidence=0.5,
                search_summary="Fallback chit-chat response"
            ),
            "current_node": "chat_handler"
        }

# ============================================================================
# NODE 3: QUERY PARSING
# ============================================================================

async def user_query_node(state: AgentState) -> Dict:
    user_query = state.get("user_query")
    if not user_query:
        return {"errors": state.get("errors", []) + ["No user query provided"], "current_node": "user_query"}
    
    try:
        print(f"\n[Query Parsing] Analyzing: '{user_query[:60]}...'")
        parsed_data = await _parse_query_with_llm(user_query)
        
        parsed_query = ParsedQuery(
            raw_query=user_query,
            intent=QueryIntent[parsed_data['intent']],
            topics=parsed_data['topics'],
            keywords=parsed_data['keywords'],
            code_language=parsed_data['code_language'],
            complexity_hint=parsed_data['complexity_hint']
        )
        
        return {"parsed_query": parsed_query, "current_node": "user_query"}
        
    except Exception as e:
        print(f"  ⚠️ LLM parsing failed: {e} -> Using fallback")
        return {"parsed_query": _fallback_parse_query(user_query), "current_node": "user_query"}

async def _parse_query_with_llm(query: str) -> dict:
    system_prompt = """You are an expert query analyzer. Analyze user queries to help retrieve the most relevant content.
Extract:
1. intent (CONCEPTUAL, CODE_REQUEST, DEBUGGING, COMPARISON, TUTORIAL)
2. topics (list of strings)
3. keywords (list of strings)
4. code_language (string or null)
5. complexity_hint (beginner, intermediate, advanced)

Respond with ONLY valid JSON in this exact format:
{
  "intent": "CONCEPTUAL",
  "topics": ["topic1", "topic2"],
  "keywords": ["keyword1", "keyword2"],
  "code_language": null,
  "complexity_hint": "intermediate"
}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Analyze this query: "{query}"')
        ])
        
        text = response.content.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        
        parsed = json.loads(text)
        
        valid_intents = ['CONCEPTUAL', 'CODE_REQUEST', 'DEBUGGING', 'COMPARISON', 'TUTORIAL']
        intent = parsed.get('intent', 'CONCEPTUAL').upper()
        if intent not in valid_intents: intent = 'CONCEPTUAL'
        
        return {
            'intent': intent,
            'topics': parsed.get('topics', [])[:10],
            'keywords': parsed.get('keywords', [])[:10],
            'code_language': parsed.get('code_language'),
            'complexity_hint': parsed.get('complexity_hint', 'intermediate')
        }
    except Exception:
        return _get_fallback_parse()

def _get_fallback_parse() -> dict:
    return {
        'intent': 'CONCEPTUAL',
        'topics': [],
        'keywords': [],
        'code_language': None,
        'complexity_hint': 'intermediate'
    }

def _fallback_parse_query(query: str) -> ParsedQuery:
    data = _get_fallback_parse()
    return ParsedQuery(
        raw_query=query, 
        intent=QueryIntent.CONCEPTUAL, 
        topics=data['topics'], 
        keywords=data['keywords'], 
        code_language=data['code_language'], 
        complexity_hint=data['complexity_hint']
    )

# ============================================================================
# NODE 4: QUERY REWRITING
# ============================================================================

async def query_rewriter_node(state: AgentState, num_variations: int = 3) -> Dict:
    parsed_query = state.get("parsed_query")
    resolved_query = state.get("resolved_query")
    
    if not parsed_query:
        return {"errors": ["Missing parsed query"], "current_node": "query_rewriter"}
    
    try:
        query_to_expand = resolved_query or parsed_query.raw_query
        print(f"\n[Query Rewriting] Expanding: '{query_to_expand}'")
        
        system_prompt = """Generate 3 closely related sub-queries that explore different aspects of the topic.
Return ONLY a valid JSON array of strings."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original: {query_to_expand}")
        ])
        
        text = response.content.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        variations = json.loads(text)
        if not isinstance(variations, list): variations = []
        
        rewritten = [query_to_expand] + variations[:num_variations]
        return {"rewritten_queries": rewritten, "current_node": "query_rewriter"}
        
    except Exception as e:
        print(f"  ⚠️ Rewriting failed: {e}")
        return {"rewritten_queries": [parsed_query.raw_query], "current_node": "query_rewriter"}

# ============================================================================
# NODE 5: VECTOR SEARCH
# ============================================================================

async def vector_search_node(state: AgentState) -> Dict:
    parsed_query = state.get("parsed_query")
    resolved_query = state.get("resolved_query")
    rewritten_queries = state.get("rewritten_queries", [])
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    
    if not parsed_query:
        return {"errors": ["Missing query"], "current_node": "vector_search"}
    
    try:
        top_k = state.get("pass1_k", 50)
        main_query = resolved_query or parsed_query.raw_query
        all_queries = [main_query] + rewritten_queries
        
        print(f"\n[PASS 1] Hierarchical Vector Search (top_k={top_k})")
        
        target_namespaces = [settings.vector_db.namespace, "papers_rag"]
        filter_dict = {}
        if state.get("book_filter"):
            filter_dict["book_title"] = state.get("book_filter")
        
        engine = get_search_engine()
        all_section_results = {}
        
        for q_text in all_queries:
            results = engine.search_and_group(
                query=q_text,
                top_k=top_k,
                namespaces=target_namespaces,
                metadata_filter=filter_dict if filter_dict else None
            )
            for res in results:
                if res['id'] not in all_section_results:
                    all_section_results[res['id']] = res
                else:
                    all_section_results[res['id']]['score'] = max(
                        all_section_results[res['id']]['score'], 
                        res['score']
                    )

        sorted_results = list(all_section_results.values())
        sorted_results.sort(key=lambda x: x['score'], reverse=True)
        sorted_results = sorted_results[:top_k]
        
        retrieved_chunks = []
        for res in sorted_results:
            meta = res['metadata']
            chunk = DocumentChunk(
                chunk_id=res['id'],
                content=res['text'],
                chapter=meta.get('display_title', 'Unknown Section'),
                section=meta.get('section_title', ''),
                page_number=meta.get('page_number', 0),
                chunk_type=meta.get('chunk_type', 'text'),
                book_title=meta.get('book_title', 'Unknown Source'),
                author=meta.get('author', 'Unknown Author')
            )
            retrieved_chunks.append(RetrievedChunk(chunk=chunk, similarity_score=res['score']))
            
        new_snapshot = {
            "stage": "vector_search",
            "chunk_count": len(retrieved_chunks),
            "chunks": retrieved_chunks[:10]
        }
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot],
            "current_node": "vector_search"
        }
    except Exception as e:
        return {"errors": [f"Vector search failed: {e}"], "current_node": "vector_search"}

# ============================================================================
# NODE 6: RERANKING
# ============================================================================

async def reranking_node(state: AgentState) -> Dict:
    retrieved_chunks = state.get("retrieved_chunks", [])
    parsed_query = state.get("parsed_query")
    pass2_k = state.get("pass2_k", 10)
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    
    if not retrieved_chunks or not parsed_query:
        return {"errors": ["Missing chunks/query"], "current_node": "reranking"}
    
    try:
        print(f"\n[PASS 2] Cross-Encoder Reranking (top_k={pass2_k})")
        cross_encoder = get_cross_encoder()
        
        chunks_data = [{
            'text': rc.chunk.content,
            'metadata': {'chunk_id': rc.chunk.chunk_id},
            'similarity_score': rc.similarity_score
        } for rc in retrieved_chunks]
        
        reranked = cross_encoder.rerank_with_metadata(
            parsed_query.raw_query, chunks_data, top_k=pass2_k
        )
        
        final_reranked = []
        for item in reranked:
            original = next(rc for rc in retrieved_chunks if rc.chunk.chunk_id == item['metadata']['chunk_id'])
            final_reranked.append(RetrievedChunk(
                chunk=original.chunk,
                similarity_score=item['similarity_score'],
                rerank_score=item['cross_encoder_score'],
                relevance_percentage=round(item['final_score'] * 100, 1)
            ))
            
        new_snapshot = {
            "stage": "reranking",
            "chunk_count": len(final_reranked),
            "chunks": final_reranked[:10]
        }
        
        return {
            "reranked_chunks": final_reranked, 
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot], 
            "current_node": "reranking"
        }
    except Exception as e:
        return {"errors": [f"Reranking failed: {e}"], "current_node": "reranking"}

# ============================================================================
# NODE 7: MULTI-HOP EXPANSION
# ============================================================================

async def multi_hop_expansion_node(state: AgentState, max_hops: int = 2) -> AgentState:
    state["current_node"] = "multi_hop_expansion"

    if not state.get("pass3_enabled", True) or not state["reranked_chunks"]:
        return state
    
    try:
        print(f"\n[PASS 3] Multi-Hop Expansion")
        expander = get_multi_hop_expander()
        
        initial_results = []
        for rc in state["reranked_chunks"][:5]: 
            initial_results.append({
                'id': rc.chunk.chunk_id,
                'text': rc.chunk.content,
                'score': rc.rerank_score,
                'book_title': rc.chunk.book_title,
                'author': rc.chunk.author
            })
        
        def retrieval_fn(query_text: str, top_k: int = 3):
            engine = get_search_engine()
            namespaces = [settings.vector_db.namespace, "papers_rag"]
            results = engine.search_and_group(query_text, top_k, namespaces)
            return [{
                'id': r['id'], 'text': r['text'], 'score': r['score'],
                'book_title': r['metadata']['book_title'], 'author': r['metadata']['author']
            } for r in results]
        
        expanded_results = expander.multi_hop_retrieve(
            state["parsed_query"].raw_query,
            initial_results,
            retrieval_fn,
            max_hops=max_hops,
            top_k_per_hop=2
        )
        
        existing_ids = {rc.chunk.chunk_id for rc in state["reranked_chunks"]}
        added_count = 0
        
        for exp_result in expanded_results:
            if exp_result['id'] not in existing_ids:
                chunk = DocumentChunk(
                    chunk_id=exp_result['id'],
                    content=exp_result['text'],
                    chapter="Multi-Hop Result", section="", page_number=0, chunk_type="text",
                    book_title=exp_result.get('book_title', 'Unknown'),
                    author=exp_result.get('author', 'Unknown')
                )
                state["reranked_chunks"].append(RetrievedChunk(
                    chunk=chunk,
                    similarity_score=exp_result['score'],
                    rerank_score=exp_result['score'] * 0.9
                ))
                added_count += 1
                existing_ids.add(exp_result['id'])

        state["pipeline_snapshots"].append({
            "stage": "multi_hop_expansion",
            "chunk_count": len(state["reranked_chunks"]),
            "chunks": state["reranked_chunks"][:10],
            "new_chunks_added": added_count
        })
        
    except Exception as e:
        print(f"  ⚠️ Multi-hop failed: {e}")
    
    return state

# ============================================================================
# NODE 8: CLUSTER EXPANSION
# ============================================================================

async def cluster_expansion_node(state: AgentState) -> AgentState:
    state["current_node"] = "cluster_expansion"
    try:
        print(f"\n[PASS 4] Cluster Expansion")
        cluster_manager = get_cluster_manager()
        
        if not cluster_manager.chunk_to_cluster:
            return state
        
        chunk_ids = [rc.chunk.chunk_id for rc in state["reranked_chunks"][:10]]
        neighbor_ids = cluster_manager.get_cluster_neighbors(chunk_ids, max_neighbors=3)
        
        state["pipeline_snapshots"].append({
            "stage": "cluster_expansion",
            "chunk_count": len(state["reranked_chunks"]),
            "chunks": state["reranked_chunks"][:10],
            "neighbors_found": len(neighbor_ids)
        })
        
    except Exception as e:
        print(f"  ⚠️ Cluster expansion failed: {e}")
    
    return state

# ============================================================================
# NODE 9: CONTEXT ASSEMBLY
# ============================================================================

async def context_assembly_node(state: AgentState) -> AgentState:
    state["current_node"] = "context_assembly"
    
    try:
        max_tokens = state.get("max_tokens", settings.retrieval.max_context_tokens)
        print(f"\n[PASS 5] Context Assembly (max_tokens={max_tokens})")
        
        compressor = get_compressor(target_tokens=int(max_tokens * 0.9), max_tokens=max_tokens)
        
        chunks_for_compression = []
        for rc in state["reranked_chunks"]:
            chunks_for_compression.append({
                'text': rc.chunk.content,
                'metadata': {
                    'chapter_title': rc.chunk.chapter,
                    'book_title': rc.chunk.book_title,
                    'author': rc.chunk.author
                },
                'score': rc.rerank_score,
                'chunk_type': rc.chunk.chunk_type
            })
            
        if compressor:
            compressed_context = compressor.compress_context(
                chunks_for_compression,
                state["parsed_query"].raw_query,
                preserve_code=True
            )
        else:
            compressed_context = "\n---\n".join([c['text'] for c in chunks_for_compression[:5]])
            
        state["assembled_context"] = compressed_context
        state["system_prompt"] = _build_system_prompt(state["parsed_query"])
        
        state["pipeline_snapshots"].append({
            "stage": "context_assembly",
            "chunk_count": len(state.get("reranked_chunks", [])),
            "chunks": state.get("reranked_chunks", [])[:10],
            "tokens": len(compressed_context.split())
        })
        
    except Exception as e:
        state["errors"].append(f"Assembly failed: {e}")
        
    return state

def _build_system_prompt(query: ParsedQuery) -> str:
    base = """You are an expert programming tutor with deep knowledge of coding books.

Your role:
- Guide learners through programming concepts
- Provide clear explanations with relevant examples from the books
- Generate new and accurate code based on book examples
- ALWAYS mention the book title when referencing examples

"""
    return base

# ============================================================================
# NODE 10: LLM REASONING (MANUAL JSON)
# ============================================================================

async def llm_reasoning_node(state: AgentState) -> Dict:
    parsed_query = state.get("parsed_query")
    assembled_context = state.get("assembled_context", "")
    system_prompt = state.get("system_prompt", "")
    
    if not assembled_context:
        return {"errors": ["No context"], "current_node": "llm_reasoning"}
        
    try:
        print(f"\n[FINAL] LLM Reasoning (Manual JSON Prompting)")
        
        # 🔥 FIXED: Use explicit prompting instead of structured output API
        # This makes it compatible with Gemma 3
        
        messages = [
            SystemMessage(content=system_prompt + "\n\nIMPORTANT: You must return your answer in valid JSON format."),
            HumanMessage(content=f"""
Context:
{assembled_context}

Question: 
{parsed_query.raw_query}

Provide your answer in the following JSON format ONLY:
{{
  "answer": "Your detailed Markdown response here...",
  "search_summary": "A short 1-sentence summary of the topic",
  "confidence_score": 0.95
}}
""")
        ]
        
        # Plain invoke (no strict schema binding)
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        
        # Manually parse JSON
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback if model fails to output JSON (rare with Gemma 3 IT)
            print("⚠️ Failed to parse JSON, using raw content")
            data = {
                "answer": content,
                "search_summary": "Response generated",
                "confidence_score": 0.5
            }
        
        sources = [c.chunk.chunk_id for c in state.get("reranked_chunks", [])[:3]]
        
        return {
            "response": LLMResponse(
                answer=data.get("answer", ""),
                code_snippets=[], 
                sources=sources,
                confidence=float(data.get("confidence_score", 0.0)),
                search_summary=data.get("search_summary", "") 
            ),
            "current_node": "llm_reasoning"
        }
        
    except Exception as e:
        print(f"❌ Generation Failed: {e}")
        return {"errors": [str(e)], "current_node": "llm_reasoning"}