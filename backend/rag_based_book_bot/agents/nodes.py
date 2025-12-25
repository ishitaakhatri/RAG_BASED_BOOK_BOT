"""
Node implementations for the LangGraph RAG pipeline.
Combines logic from previous nodes.py and memory_nodes.py.
Uses latest LangChain and Pydantic v2 features.
"""

import os
import json
import re
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Import State definitions
from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent, LLMResponse
)

# Retrieval components
from rag_based_book_bot.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag_based_book_bot.retrieval.multi_hop_expander import MultiHopExpander
from rag_based_book_bot.retrieval.cluster_manager import ClusterManager
from rag_based_book_bot.retrieval.context_compressor import EnhancedContextCompressor

load_dotenv()

# ============================================================================
# INITIALIZATION (Lazy Loading)
# ============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = "BAAI/bge-m3"

# Globals for lazy loading
_pc = None
_index = None
_model = None
_cross_encoder = None
_multi_hop = None
_cluster_manager = None

def get_pinecone_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(INDEX_NAME)
    return _index

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

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

def get_llm():
    """Returns the configured Google Generative AI model."""
    return ChatGoogleGenerativeAI(
        model="models/gemma-3-27b-it",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        max_retries=2
    )

# ============================================================================
# STRUCTURED OUTPUT SCHEMAS (Pydantic v2)
# ============================================================================

class QueryAnalysis(BaseModel):
    intent: QueryIntent = Field(..., description="The intent of the user query")
    topics: List[str] = Field(..., description="Key technical concepts (3-5 items)")
    keywords: List[str] = Field(..., description="Search terms for retrieval (5-10 words)")
    code_language: Optional[str] = Field(None, description="Programming language referenced, if any")
    complexity_hint: str = Field("intermediate", description="beginner, intermediate, or advanced")

class ContextAnalysis(BaseModel):
    needs_conversation_context: bool = Field(..., description="Does query use pronouns or implicit refs?")
    references_turn: Optional[int] = Field(None, description="Which turn number (1-based) is referenced?")
    can_answer_from_history: bool = Field(..., description="Can this be answered solely from conversation history?")
    standalone_query: str = Field(..., description="Rewritten query that is standalone")
    reasoning: str

class RewrittenQueries(BaseModel):
    variations: List[str] = Field(..., description="List of 3 alternative query phrasings")

# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def query_parser_node(state: AgentState) -> dict:
    """Parses the user query using PydanticOutputParser."""
    print(f"\n[Query Parsing] Analyzing: '{state.user_query[:60]}...'")
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
    
    # NOTE: Gemma 3 does not support 'system' role. Merged instructions into 'human'.
    prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an expert query analyzer for a coding book learning assistant.
        Analyze the user query to help retrieve relevant content.
        
        Intents:
        - conceptual: Theory, "what is X?"
        - code_request: Implementations, "show me code"
        - debugging: Troubleshooting, "why isn't this working?"
        - comparison: "difference between X and Y"
        - tutorial: Step-by-step guides

        Return the result in strict JSON format matching the schema below.
        {format_instructions}
        
        User Query: {query}
        """)
    ])
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "query": state.user_query,
            "format_instructions": parser.get_format_instructions()
        })
        
        parsed_query = ParsedQuery(
            raw_query=state.user_query,
            intent=result.intent,
            topics=result.topics,
            keywords=result.keywords,
            code_language=result.code_language,
            complexity_hint=result.complexity_hint
        )
        
        return {"parsed_query": parsed_query}
        
    except Exception as e:
        print(f"  ⚠️ LLM parsing failed: {e}. Using fallback.")
        return {
            "parsed_query": ParsedQuery(
                raw_query=state.user_query, 
                intent=QueryIntent.CONCEPTUAL,
                topics=[], keywords=state.user_query.split()
            ),
            "errors": [f"Query parsing failed: {str(e)}"]
        }

def context_resolution_node(state: AgentState) -> dict:
    """Resolves context from conversation history."""
    current_query = state.parsed_query.raw_query if state.parsed_query else state.user_query
    
    if not state.conversation_history:
        print(f"\n[Context Resolution] No history, skipping.")
        return {
            "resolved_query": current_query,
            "needs_retrieval": True,
            "referenced_turn": None
        }
        
    print(f"\n[Context Resolution] Analyzing history for: '{current_query}'")
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=ContextAnalysis)
    
    history_str = "\n".join([
        f"Turn {i+1}: Q: {t.user_query} A: {t.assistant_response[:100]}..." 
        for i, t in enumerate(state.conversation_history[-5:])
    ])
    
    # NOTE: Merged instructions into 'human' message
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Analyze the current query given the conversation history.
        Determine if the query refers to previous context.
        If the answer exists in history, set can_answer_from_history=True.
        Always provide a standalone_query.

        Return strict JSON.
        {format_instructions}
        
        History:
        {history}

        Current Query: {query}
        """)
    ])
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "history": history_str,
            "query": current_query,
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"  → Resolved: {result.standalone_query}")
        print(f"  → Needs Retrieval: {not result.can_answer_from_history}")
        
        return {
            "resolved_query": result.standalone_query,
            "needs_retrieval": not result.can_answer_from_history,
            "referenced_turn": result.references_turn
        }
    except Exception as e:
        print(f"  ⚠️ Context resolution failed: {e}")
        return {
            "resolved_query": current_query,
            "needs_retrieval": True,
            "errors": [f"Context resolution failed: {str(e)}"]
        }

def answer_from_history_node(state: AgentState) -> dict:
    """Answers directly from conversation history."""
    print(f"\n[Answer from History]")
    llm = get_llm()
    
    turns_to_use = state.conversation_history[-3:]
    if state.referenced_turn and 0 <= state.referenced_turn - 1 < len(state.conversation_history):
        turns_to_use = [state.conversation_history[state.referenced_turn - 1]]
        
    history_context = "\n\n".join([f"Q: {t.user_query}\nA: {t.assistant_response}" for t in turns_to_use])
    
    # NOTE: Merged instructions into 'human' message
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Answer the user's question solely based on the conversation history provided below.
        
        History:
        {history}
        
        Question: {question}""")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "history": history_context,
            "question": state.resolved_query or state.user_query
        })
        
        return {
            "response": LLMResponse(
                answer=response.content,
                sources=[],
                confidence=0.8
            ),
            "pipeline_snapshots": [{
                "stage": "answer_from_history",
                "chunk_count": 0,
                "answered_from_memory": True
            }]
        }
    except Exception as e:
        return {"errors": [f"History answer failed: {str(e)}"]}

def query_rewriter_node(state: AgentState) -> dict:
    """Generates query variations."""
    query_to_expand = state.resolved_query or state.parsed_query.raw_query
    print(f"\n[Query Rewriting] Expanding: '{query_to_expand}'")
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=RewrittenQueries)
    
    # NOTE: Merged instructions into 'human' message
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Generate 3 alternative search queries for the user's request, preserving intent.
        Return strict JSON.
        {format_instructions}
        
        Original Query: {query}""")
    ])
    
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "query": query_to_expand, 
            "intent": state.parsed_query.intent.value if state.parsed_query else "conceptual",
            "format_instructions": parser.get_format_instructions()
        })
        return {"rewritten_queries": result.variations}
    except Exception as e:
        return {"rewritten_queries": [query_to_expand]}

def vector_search_node(state: AgentState) -> dict:
    """Pass 1: Vector Search."""
    print(f"\n[PASS 1] Vector Search")
    try:
        index = get_pinecone_index()
        model = get_embedding_model()
        
        queries = [state.resolved_query] + state.rewritten_queries
        all_results = {}
        
        filter_dict = {}
        if state.book_filter:
            filter_dict["book_title"] = state.book_filter
        if state.chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [state.chapter_filter]}

        for q in queries:
            if not q: continue
            emb = model.encode(q).tolist()
            res = index.query(
                vector=emb, 
                top_k=state.pass1_k, 
                namespace=NAMESPACE, 
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            for m in res.get('matches', []):
                if m['id'] not in all_results or m['score'] > all_results[m['id']]['score']:
                    all_results[m['id']] = m
        
        chunks = []
        sorted_matches = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:state.pass1_k]
        
        for m in sorted_matches:
            md = m['metadata']
            chapter_title = md.get('chapter_title', md.get('chapter', 'Unknown'))
            
            chunks.append(RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=m['id'],
                    content=md.get('text', ''),
                    chapter=chapter_title,
                    book_title=md.get('book_title', 'Unknown Book'),
                    author=md.get('author', 'Unknown Author'),
                    page_number=int(md.get('page_number', 0)),
                    chunk_type=md.get('chunk_type', 'text'),
                    chapter_title=chapter_title,
                    preview=md.get('preview', '')
                ),
                similarity_score=m['score']
            ))
            
        print(f"  → Found {len(chunks)} unique chunks")
        
        return {
            "retrieved_chunks": chunks,
            "pipeline_snapshots": [{
                "stage": "vector_search", 
                "chunk_count": len(chunks),
                "chunks": chunks[:10]
            }]
        }
    except Exception as e:
        return {"errors": [f"Vector search failed: {e}"]}

def reranking_node(state: AgentState) -> dict:
    """Pass 2: Reranking."""
    print(f"\n[PASS 2] Reranking")
    if not state.retrieved_chunks:
        return {}
        
    try:
        reranker = get_cross_encoder()
        query = state.resolved_query
        
        to_rank = [{
            'text': rc.chunk.content, 
            'metadata': {'chunk_id': rc.chunk.chunk_id, 'book_title': rc.chunk.book_title}
        } for rc in state.retrieved_chunks]
        
        results = reranker.rerank_with_metadata(query, to_rank, top_k=state.pass2_k)
        
        reranked = []
        chunk_map = {rc.chunk.chunk_id: rc.chunk for rc in state.retrieved_chunks}
        
        for res in results:
            cid = res['metadata']['chunk_id']
            if cid in chunk_map:
                reranked.append(RetrievedChunk(
                    chunk=chunk_map[cid],
                    similarity_score=res['similarity_score'],
                    rerank_score=res['cross_encoder_score'],
                    relevance_percentage=res['final_score'] * 100
                ))
        
        print(f"  → Reranked top {len(reranked)}")
        
        return {
            "reranked_chunks": reranked,
            "pipeline_snapshots": [{
                "stage": "reranking", 
                "chunk_count": len(reranked),
                "chunks": reranked[:10]
            }]
        }
    except Exception as e:
         return {"errors": [f"Reranking failed: {e}"]}

def context_assembly_node(state: AgentState) -> dict:
    """Pass 5: Context Assembly."""
    print(f"\n[PASS 5] Assembly")
    
    context = ""
    try:
        for rc in state.reranked_chunks:
            context += f"Source: {rc.chunk.book_title} (Ch {rc.chunk.chapter}, Page {rc.chunk.page_number})\n{rc.chunk.content}\n---\n"
    except Exception as e:
        print(f"Assembly warning: {e}")
    
    return {
        "assembled_context": context,
        "pipeline_snapshots": [{
            "stage": "context_assembly", 
            "chunk_count": len(state.reranked_chunks)
        }]
    }

def llm_reasoning_node(state: AgentState) -> dict:
    """Final LLM Answer Generation."""
    print(f"\n[FINAL] LLM Reasoning")
    llm = get_llm()
    
    if not state.assembled_context:
        return {"response": LLMResponse(answer="I couldn't find relevant information.", confidence=0.0)}

    prompt = ChatPromptTemplate.from_template("""You are an expert programming tutor.
    Answer based on the context below. Cite books.
    
    Context:
    {context}
    
    Question: {question}""")
    
    try:
        chain = prompt | llm
        res = chain.invoke({
            "context": state.assembled_context,
            "question": state.resolved_query or state.user_query
        })
        
        code_snippets = re.findall(r'```(?:\w+)?\n(.*?)```', res.content, re.DOTALL)
        
        return {
            "response": LLMResponse(
                answer=res.content,
                sources=[rc.chunk.chunk_id for rc in state.reranked_chunks[:5]],
                confidence=0.9,
                code_snippets=code_snippets
            )
        }
    except Exception as e:
        return {"errors": [f"LLM reasoning failed: {e}"]}