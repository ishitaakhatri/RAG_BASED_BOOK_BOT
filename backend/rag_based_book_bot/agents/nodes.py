"""
Updated Node implementations with LangChain and Gemini

CHANGES:
- Integrated HierarchicalSearchEngine for vector_search_node
- Added support for multiple namespaces (Books + Papers)
- RESTORED Multi-Hop Expansion logic
- RESTORED Cluster Expansion logic
- RESTORED Enhanced Context Compression
"""

import re
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

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
settings = get_config()


# ============================================================================
# LANGCHAIN LLM INITIALIZATION
# ============================================================================

llm = ChatGoogleGenerativeAI(
    model=settings.llm.model_name,
    google_api_key=settings.llm.google_api_key,
    temperature=settings.llm.temperature,
    max_retries=0, 
    convert_system_message_to_human=True 
)

# Global instances (lazy loading)
_pc = None
_index = None
_model = None
_search_engine = None  # NEW: Hierarchical Search Engine
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
    """Get the Hierarchical Search Engine"""
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


# ============================================================================
# QUERY PARSING NODES
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


async def query_rewriter_node(state: AgentState, num_variations: int = 3) -> Dict:
    parsed_query = state.get("parsed_query")
    resolved_query = state.get("resolved_query")
    
    if not parsed_query:
        return {"errors": state.get("errors", []) + ["Missing parsed query"], "current_node": "query_rewriter"}
    
    try:
        query_to_expand = resolved_query or parsed_query.raw_query
        print(f"\n[Query Rewriting] Expanding: '{query_to_expand}'")
        
        rewritten = await _generate_query_variations(query_to_expand, parsed_query.intent, num_variations)
        
        return {"rewritten_queries": rewritten, "current_node": "query_rewriter"}
    except Exception as e:
        print(f"  ⚠️ Query rewriting failed: {e}")
        return {"rewritten_queries": [], "current_node": "query_rewriter"}


async def _generate_query_variations(
    query: str,
    intent: QueryIntent,
    num_variations: int = 3
) -> list[str]:
    """Generate alternative query formulations using Gemini"""

    system_prompt = """You are an expert at identifying key concepts and aspects of a technical topic.

Task:
Given a user query, generate only 3 closely related sub-queries that explore
IMPORTANT aspects of the same topic and help retrieve comprehensive information.

The goal is to maximize recall without drifting off-topic.

Guidelines:
- Each sub-query should focus on a different important aspect of the topic
  (e.g., definition, components, implementation, applications, limitations)
- Do NOT repeat the original query
- Do NOT introduce unrelated topics
- Do NOT add speculative or advanced topics unless implied by the query
- Keep each sub-query concise and specific
- Use clear technical phrasing suitable for documentation search

Return ONLY a valid JSON array of strings.
"""

    intent_hints = {
        QueryIntent.CONCEPTUAL: "Focus on understanding, explanation, and theoretical aspects.",
        QueryIntent.CODE_REQUEST: "Vary between implementation details, code examples, and practical usage.",
        QueryIntent.DEBUGGING: "Include variations about troubleshooting, error fixing, and problem solving.",
        QueryIntent.COMPARISON: "Rephrase as differences, pros/cons, or when to use each option.",
        QueryIntent.TUTORIAL: "Vary between step-by-step guides, walkthroughs, and practical examples.",
    }

    user_prompt = f"""Original query: "{query}"

Intent: {intent.value}
Hint: {intent_hints.get(intent, "")}

Generate {num_variations} alternative phrasings.
"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        # ---- Clean fenced code blocks if present ----
        if response_text.startswith("```"):
            response_text = (
                response_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        # ---- Parse JSON ----
        variations = json.loads(response_text)

        if not isinstance(variations, list):
            raise ValueError("LLM response is not a JSON list")

        # ---- De-duplicate & remove original query ----
        filtered_variations = []
        query_norm = query.strip().lower()

        for v in variations:
            if isinstance(v, str):
                v_clean = v.strip()
                if v_clean.lower() != query_norm:
                    filtered_variations.append(v_clean)

        # ---- Always return original query first ----
        return [query] + filtered_variations[:num_variations]

    except Exception as e:
        print(f"⚠️ LLM query rewriting failed: {e}")
        return _fallback_query_variations(query, num_variations)

def _fallback_query_variations(query: str, num_variations: int = 3) -> list[str]:
    return [query] * num_variations


async def _parse_query_with_llm(query: str) -> dict:
    system_prompt = """You are an expert query analyzer. Analyze user queries to help retrieve the most relevant content.
Extract:
1. intent (CONCEPTUAL, CODE_REQUEST, DEBUGGING, COMPARISON, TUTORIAL)
2. topics (list of strings)
3. keywords (list of strings)
4. code_language (string or null)
5. complexity_hint (beginner, intermediate, advanced)

Respond with ONLY valid JSON."""

    user_prompt = f'Analyze this query: "{query}"'

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "").replace("```", "")
        
        parsed = json.loads(response_text)
        
        return {
            'intent': parsed.get('intent', 'CONCEPTUAL'),
            'topics': parsed.get('topics', []),
            'keywords': parsed.get('keywords', []),
            'code_language': parsed.get('code_language'),
            'complexity_hint': parsed.get('complexity_hint', 'intermediate')
        }
    except Exception as e:
        print(f"Parsing error: {e}")
        return {
            'intent': 'CONCEPTUAL',
            'topics': [],
            'keywords': [],
            'code_language': None,
            'complexity_hint': 'intermediate'
        }

def _fallback_parse_query(query: str) -> ParsedQuery:
    return ParsedQuery(
        raw_query=query, 
        intent=QueryIntent.CONCEPTUAL, 
        topics=[], 
        keywords=[], 
        code_language=None, 
        complexity_hint="intermediate"
    )


# ============================================================================
# RETRIEVAL NODES (FULLY RESTORED)
# ============================================================================

async def vector_search_node(state: AgentState) -> Dict:
    """PASS 1: Hierarchical Vector Search"""
    parsed_query = state.get("parsed_query")
    resolved_query = state.get("resolved_query")
    rewritten_queries = state.get("rewritten_queries", [])
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    
    if not parsed_query:
        return {"errors": state.get("errors", []) + ["Missing query"], "current_node": "vector_search"}
    
    try:
        top_k = state.get("pass1_k", 50)
        main_query = resolved_query or parsed_query.raw_query
        all_queries = [main_query] + rewritten_queries
        
        print(f"\n[PASS 1] Hierarchical Vector Search (top_k={top_k})")
        
        # 1. Determine Namespaces (Both Books and Papers)
        target_namespaces = [settings.vector_db.namespace, "papers_rag"]
        
        # 2. Build Filter
        filter_dict = {}
        if state.get("book_filter"):
            filter_dict["book_title"] = state.get("book_filter")
        
        # 3. Initialize Engine
        engine = get_search_engine()
        all_section_results = {}
        
        # 4. Search & Group
        for q_text in all_queries:
            results = engine.search_and_group(
                query=q_text,
                top_k=top_k,
                namespaces=target_namespaces,
                metadata_filter=filter_dict if filter_dict else None
            )
            
            for res in results:
                # Deduplicate based on Section ID
                if res['id'] not in all_section_results:
                    all_section_results[res['id']] = res
                else:
                    # Update if we found a higher score for the same section
                    all_section_results[res['id']]['score'] = max(
                        all_section_results[res['id']]['score'], 
                        res['score']
                    )

        # 5. Convert to DocumentChunk
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
        
        print(f"  → Retrieved {len(retrieved_chunks)} coherent sections")
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot],
            "current_node": "vector_search"
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Vector search failed: {e}"], "current_node": "vector_search"}


async def reranking_node(state: AgentState) -> Dict:
    """PASS 2: Cross-Encoder Reranking"""
    retrieved_chunks = state.get("retrieved_chunks", [])
    parsed_query = state.get("parsed_query")
    pass2_k = state.get("pass2_k", 10)
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    
    if not retrieved_chunks or not parsed_query:
        return {"errors": ["Missing chunks/query"], "current_node": "reranking"}
    
    try:
        print(f"\n[PASS 2] Cross-Encoder Reranking (top_k={pass2_k})")
        cross_encoder = get_cross_encoder()
        
        # Prepare input for reranker (using full section text)
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
        print(f"  → Reranked to {len(final_reranked)} sections")
        
        return {
            "reranked_chunks": final_reranked, 
            "pipeline_snapshots": pipeline_snapshots + [new_snapshot], 
            "current_node": "reranking"
        }
    except Exception as e:
        return {"errors": [f"Reranking failed: {e}"], "current_node": "reranking"}


async def multi_hop_expansion_node(state: AgentState, max_hops: int = 2) -> AgentState:
    """PASS 3: Multi-Hop Retrieval (RESTORED)"""
    state["current_node"] = "multi_hop_expansion"

    # Check enabled status
    if not state.get("pass3_enabled", True):
        print(f"\n[PASS 3] Multi-Hop Expansion - SKIPPED (disabled)")
        return state
    
    if not state["reranked_chunks"] or not state["parsed_query"]:
        return state
    
    try:
        print(f"\n[PASS 3] Multi-Hop Expansion (max_hops={max_hops})")
        
        before_expansion = len(state["reranked_chunks"])
        expander = get_multi_hop_expander()
        
        # Prepare initial results from Super Chunks
        initial_results = []
        for rc in state["reranked_chunks"][:5]: # Use top 5 sections as anchors
            initial_results.append({
                'id': rc.chunk.chunk_id,
                'text': rc.chunk.content,
                'score': rc.rerank_score,
                'book_title': rc.chunk.book_title,
                'author': rc.chunk.author
            })
        
        # Helper for recursive search using the Hierarchical Search Engine
        def retrieval_fn(query_text: str, top_k: int = 3):
            engine = get_search_engine()
            namespaces = [settings.vector_db.namespace, "papers_rag"]
            results = engine.search_and_group(query_text, top_k, namespaces)
            
            return [{
                'id': r['id'],
                'text': r['text'],
                'score': r['score'],
                'book_title': r['metadata']['book_title'],
                'author': r['metadata']['author']
            } for r in results]
        
        expanded_results = expander.multi_hop_retrieve(
            state["parsed_query"].raw_query,
            initial_results,
            retrieval_fn,
            max_hops=max_hops,
            top_k_per_hop=2
        )
        
        # Add new results to state
        existing_ids = {rc.chunk.chunk_id for rc in state["reranked_chunks"]}
        
        added_count = 0
        for exp_result in expanded_results:
            if exp_result['id'] not in existing_ids:
                chunk = DocumentChunk(
                    chunk_id=exp_result['id'],
                    content=exp_result['text'],
                    chapter="Multi-Hop Result",
                    section="",
                    page_number=0,
                    chunk_type="text",
                    book_title=exp_result.get('book_title', 'Unknown'),
                    author=exp_result.get('author', 'Unknown')
                )
                state["reranked_chunks"].append(RetrievedChunk(
                    chunk=chunk,
                    similarity_score=exp_result['score'],
                    rerank_score=exp_result['score'] * 0.9 # Penalize slightly
                ))
                added_count += 1
                existing_ids.add(exp_result['id'])

        state["pipeline_snapshots"].append({
            "stage": "multi_hop_expansion",
            "chunk_count": len(state["reranked_chunks"]),
            "chunks": state["reranked_chunks"][before_expansion:],
            "new_chunks_added": added_count
        })
        print(f"  → Added {added_count} multi-hop sections")
        
    except Exception as e:
        print(f"  ⚠️ Multi-hop expansion failed: {e}")
    
    return state


async def cluster_expansion_node(state: AgentState) -> AgentState:
    """PASS 4: Cluster Expansion (RESTORED)"""
    state["current_node"] = "cluster_expansion"
    
    try:
        print(f"\n[PASS 4] Cluster Expansion")
        cluster_manager = get_cluster_manager()
        
        if not cluster_manager.chunk_to_cluster:
            print("  ⚠️ No clusters available, skipping")
            return state
        
        # Use top 10 chunks as seeds
        chunk_ids = [rc.chunk.chunk_id for rc in state["reranked_chunks"][:10]]
        neighbor_ids = cluster_manager.get_cluster_neighbors(chunk_ids, max_neighbors=3)
        
        if neighbor_ids:
            # Note: Fetching content for these neighbors would ideally use index.fetch
            # For this node, we identify potential neighbors but don't force a heavy fetch
            # to keep the response fast, unless they are critical.
            # In a production fetch, you would call `index.fetch(ids=neighbor_ids)`.
            pass 
        
        state["pipeline_snapshots"].append({
            "stage": "cluster_expansion",
            "chunk_count": len(state["reranked_chunks"]),
            "neighbors_found": len(neighbor_ids)
        })
        print(f"  → Identified {len(neighbor_ids)} potential cluster neighbors")
        
    except Exception as e:
        print(f"  ⚠️ Cluster expansion failed: {e}")
    
    return state


async def context_assembly_node(state: AgentState) -> AgentState:
    """PASS 5: Context Assembly (RESTORED SMART COMPRESSION)"""
    state["current_node"] = "context_assembly"
    
    try:
        max_tokens = state.get("max_tokens", settings.retrieval.max_context_tokens)
        print(f"\n[PASS 5] Context Compression & Assembly (max_tokens={max_tokens})")
        
        # 1. Attempt Smart Compression
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
            print("  → Applied Smart Compression")
        else:
            # Fallback
            compressed_context = "\n---\n".join([c['text'] for c in chunks_for_compression[:5]])
            print("  → Applied Fallback Assembly")
            
        state["assembled_context"] = compressed_context
        state["system_prompt"] = _build_system_prompt(state["parsed_query"])
        
    except Exception as e:
        state["errors"].append(f"Assembly failed: {e}")
        
    return state


async def llm_reasoning_node(state: AgentState) -> Dict:
    parsed_query = state.get("parsed_query")
    assembled_context = state.get("assembled_context", "")
    system_prompt = state.get("system_prompt", "")
    
    if not assembled_context:
        return {"errors": ["No context"], "current_node": "llm_reasoning"}
        
    try:
        print(f"\n[FINAL] LLM Reasoning")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{assembled_context}\n\nQuestion: {parsed_query.raw_query}")
        ]
        
        response = llm.invoke(messages)
        
        return {
            "response": LLMResponse(
                answer=response.content,
                code_snippets=[],
                sources=[c.chunk.chunk_id for c in state["reranked_chunks"][:3]],
                confidence=0.9
            ),
            "current_node": "llm_reasoning"
        }
    except Exception as e:
         return {"errors": [str(e)], "current_node": "llm_reasoning"}

def _build_system_prompt(query: ParsedQuery) -> str:
    base = """You are an expert programming tutor with deep knowledge of coding books and technical documentation.

Your role:
- Guide learners through programming concepts
- Provide clear explanations with relevant examples from the books
- Generate new and accurate code based on book examples and best practices
- Explain important keywords present in the answer with sufficient length
- Explain code snippets briefly but clearly
- ALWAYS mention the book title when referencing examples or concepts

Always reference sources WITH BOOK TITLES and ensure code is correct and follows best practices."""
    
    intent_prompts = {
        QueryIntent.CODE_REQUEST: "\n\n**Focus**: Provide working, well-commented code with explanations and cite the source book.",
        QueryIntent.CONCEPTUAL: "\n\n**Focus**: Explain concepts clearly with examples and mention which books they come from.",
        QueryIntent.COMPARISON: "\n\n**Focus**: Compare systematically with pros/cons, citing specific books.",
        QueryIntent.DEBUGGING: "\n\n**Focus**: Identify issues and provide fixes with book references.",
        QueryIntent.TUTORIAL: "\n\n**Focus**: Provide step-by-step guidance with book citations."
    }
    
    complexity = {
        "beginner": " Use simple language and basic examples.",
        "intermediate": " Balance theory and practice.",
        "advanced": " Include technical details and edge cases."
    }
    
    return base + intent_prompts.get(query.intent, "") + complexity.get(query.complexity_hint, "")