"""
Query Expansion / Multi-Hop Retrieval Node
Extracts key concepts and performs secondary searches
"""
import os
import logging
from typing import List, Set
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from rag_based_book_bot.agents.states import (
    AgentState, RetrievedChunk, DocumentChunk
)

logger = logging.getLogger("query_expansion")

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global resources (lazy init)
_openai_client = None
_pinecone_index = None
_embedding_model = None


def get_openai_client():
    """Lazy load OpenAI client"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_pinecone_index():
    """Lazy load Pinecone index"""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def extract_key_concepts(query: str, top_chunks: List[RetrievedChunk]) -> List[str]:
    """
    Use LLM to extract key concepts from top retrieved chunks.
    
    Args:
        query: Original user query
        top_chunks: Top 3-5 chunks from initial retrieval
    
    Returns:
        List of key concepts to search for
    """
    client = get_openai_client()
    
    # Build context from top chunks
    context = "\n\n".join([
        f"[CHUNK {i+1}]\n{chunk.chunk.content[:500]}"
        for i, chunk in enumerate(top_chunks[:3])
    ])
    
    # Prompt for concept extraction
    prompt = f"""Given this query and retrieved content, extract 3-5 KEY CONCEPTS or TERMS that would help find related information in other chapters.

Query: {query}

Retrieved Content:
{context}

Extract concepts that are:
1. Technical terms (e.g., "convolutional layer", "gradient descent")
2. Prerequisites (e.g., if discussing CNNs, mention "neural networks basics")
3. Related topics (e.g., if discussing training, mention "optimization", "loss functions")

Return ONLY a comma-separated list of concepts, nothing else.
Example: "neural networks, backpropagation, activation functions, gradient descent"
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical concept extractor. Return only comma-separated terms."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        concepts_text = response.choices[0].message.content.strip()
        concepts = [c.strip() for c in concepts_text.split(",")]
        
        logger.info(f"   Extracted concepts: {concepts}")
        return concepts[:5]  # Max 5 concepts
        
    except Exception as e:
        logger.warning(f"Concept extraction failed: {e}")
        # Fallback: use query keywords
        return query.split()[:3]


def search_with_expanded_query(
    expanded_queries: List[str],
    existing_chunk_ids: Set[str],
    top_k: int = 10
) -> List[RetrievedChunk]:
    """
    Perform secondary searches with expanded queries.
    
    Args:
        expanded_queries: List of concept-based queries
        existing_chunk_ids: IDs of already retrieved chunks (avoid duplicates)
        top_k: Number of chunks to retrieve per query
    
    Returns:
        List of newly retrieved chunks
    """
    index = get_pinecone_index()
    model = get_embedding_model()
    
    new_chunks = []
    seen_ids = existing_chunk_ids.copy()
    
    for concept in expanded_queries:
        try:
            # Generate embedding
            query_embedding = model.encode(concept).tolist()
            
            # Search Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=NAMESPACE,
                include_metadata=True
            )
            
            # Convert to RetrievedChunk objects
            for match in results.get("matches", []):
                chunk_id = match["id"]
                
                # Skip if already retrieved
                if chunk_id in seen_ids:
                    continue
                
                metadata = match.get("metadata", {})
                
                # Handle metadata lists
                chapter_titles = metadata.get("chapter_titles", [])
                chapter_numbers = metadata.get("chapter_numbers", [])
                section_titles = metadata.get("section_titles", [])
                
                chapter_title = chapter_titles[0] if chapter_titles else ""
                chapter_number = chapter_numbers[0] if chapter_numbers else ""
                
                # Create DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=metadata.get("text", ""),
                    chapter=f"{chapter_number}: {chapter_title}" if chapter_number else chapter_title,
                    section=", ".join(section_titles) if section_titles else "",
                    page_number=metadata.get("page_start"),
                    chunk_type="code" if metadata.get("contains_code") else "text"
                )
                
                # Create RetrievedChunk
                retrieved = RetrievedChunk(
                    chunk=chunk,
                    similarity_score=match.get("score", 0.0),
                    rerank_score=match.get("score", 0.0) * 0.8  # Slight penalty for secondary retrieval
                )
                
                new_chunks.append(retrieved)
                seen_ids.add(chunk_id)
                
        except Exception as e:
            logger.warning(f"Secondary search for '{concept}' failed: {e}")
            continue
    
    return new_chunks


def query_expansion_node(state: AgentState, top_k_expansion: int = 10) -> AgentState:
    """
    PASS 3: Query Expansion / Multi-Hop Retrieval
    
    Extracts key concepts from top chunks and performs secondary searches
    to find cross-chapter supporting information.
    
    Args:
        state: Current agent state
        top_k_expansion: Max new chunks to retrieve
    
    Returns:
        Updated state with expanded retrieved chunks
    """
    state.current_node = "query_expansion"
    
    if not state.retrieved_chunks or not state.parsed_query:
        state.errors.append("Missing chunks or query for query expansion")
        return state
    
    try:
        logger.info(f"🔍 Query expansion: analyzing top {len(state.retrieved_chunks[:5])} chunks...")
        
        # Step 1: Extract key concepts from top chunks
        concepts = extract_key_concepts(
            state.parsed_query.raw_query,
            state.retrieved_chunks[:5]
        )
        
        if not concepts:
            logger.info("   No concepts extracted, skipping expansion")
            return state
        
        # Step 2: Get IDs of existing chunks
        existing_ids = {chunk.chunk.chunk_id for chunk in state.retrieved_chunks}
        
        # Step 3: Search with expanded queries
        logger.info(f"   Searching for: {', '.join(concepts)}")
        new_chunks = search_with_expanded_query(
            concepts,
            existing_ids,
            top_k=top_k_expansion
        )
        
        if new_chunks:
            logger.info(f"✅ Query expansion found {len(new_chunks)} additional relevant chunks")
            
            # Merge with existing chunks
            state.retrieved_chunks.extend(new_chunks)
            
            # Re-sort by rerank_score
            state.retrieved_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        else:
            logger.info("   No additional chunks found")
        
    except Exception as e:
        state.errors.append(f"Query expansion failed: {str(e)}")
        logger.error(f"❌ Query expansion error: {e}")
    
    return state