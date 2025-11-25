"""
Compression & Deduplication Node
Removes redundancy and compresses context for LLM
"""
import logging
import tiktoken
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from rag_based_book_bot.agents.states import AgentState, RetrievedChunk

logger = logging.getLogger("compression")

# Global resources
_embedding_model = None
_tokenizer = None


def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_tokenizer():
    """Lazy load tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text"""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def detect_duplicates(chunks: List[RetrievedChunk], similarity_threshold: float = 0.95) -> List[int]:
    """
    Detect near-duplicate chunks using embedding similarity.
    
    Args:
        chunks: List of retrieved chunks
        similarity_threshold: Cosine similarity threshold for duplicates
    
    Returns:
        List of indices to remove (lower-scoring duplicates)
    """
    if len(chunks) <= 1:
        return []
    
    # Generate embeddings for all chunks
    model = get_embedding_model()
    texts = [chunk.chunk.content for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)
    
    # Find duplicates (compare each pair)
    to_remove = set()
    
    for i in range(len(chunks)):
        if i in to_remove:
            continue
        
        for j in range(i + 1, len(chunks)):
            if j in to_remove:
                continue
            
            if similarities[i, j] >= similarity_threshold:
                # Found duplicate - keep higher scoring one
                if chunks[i].rerank_score >= chunks[j].rerank_score:
                    to_remove.add(j)
                    logger.debug(f"   Duplicate detected: keeping chunk {i}, removing {j} (similarity: {similarities[i,j]:.3f})")
                else:
                    to_remove.add(i)
                    logger.debug(f"   Duplicate detected: keeping chunk {j}, removing {i} (similarity: {similarities[i,j]:.3f})")
                    break  # Don't compare i with others anymore
    
    return sorted(list(to_remove), reverse=True)


def merge_overlapping_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Merge chunks from same chapter/section that are sequential.
    
    Args:
        chunks: List of retrieved chunks
    
    Returns:
        List with merged chunks
    """
    if len(chunks) <= 1:
        return chunks
    
    merged = []
    skip_next = set()
    
    for i, chunk in enumerate(chunks):
        if i in skip_next:
            continue
        
        # Check if next chunk is from same chapter and has similar content
        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            
            # Same chapter and sequential pages?
            same_chapter = chunk.chunk.chapter == next_chunk.chunk.chapter
            sequential_pages = (
                chunk.chunk.page_number is not None and
                next_chunk.chunk.page_number is not None and
                abs(chunk.chunk.page_number - next_chunk.chunk.page_number) <= 2
            )
            
            if same_chapter and sequential_pages:
                # Merge content
                merged_content = chunk.chunk.content + "\n\n" + next_chunk.chunk.content
                
                # Check if merged content is reasonable size
                merged_tokens = count_tokens(merged_content)
                if merged_tokens <= 2000:  # Don't create too large chunks
                    chunk.chunk.content = merged_content
                    chunk.rerank_score = max(chunk.rerank_score, next_chunk.rerank_score)
                    skip_next.add(i + 1)
                    logger.debug(f"   Merged chunks {i} and {i+1} from {chunk.chunk.chapter}")
        
        merged.append(chunk)
    
    return merged


def truncate_chunks_to_budget(
    chunks: List[RetrievedChunk],
    max_tokens: int = 2500
) -> List[RetrievedChunk]:
    """
    Truncate chunks to fit within token budget while preserving most important ones.
    
    Args:
        chunks: Sorted list of chunks (best first)
        max_tokens: Maximum total tokens
    
    Returns:
        List of chunks that fit within budget
    """
    selected = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk.chunk.content)
        
        if current_tokens + chunk_tokens <= max_tokens:
            selected.append(chunk)
            current_tokens += chunk_tokens
        else:
            # Check if we can fit a truncated version
            remaining = max_tokens - current_tokens
            if remaining >= 200:  # At least 200 tokens
                # Truncate this chunk
                tokenizer = get_tokenizer()
                tokens = tokenizer.encode(chunk.chunk.content)
                truncated_tokens = tokens[:remaining]
                chunk.chunk.content = tokenizer.decode(truncated_tokens) + "..."
                selected.append(chunk)
                logger.debug(f"   Truncated chunk to fit budget: {remaining} tokens")
            break
    
    return selected


def compression_node(
    state: AgentState,
    max_total_tokens: int = 2500,
    remove_duplicates: bool = True,
    merge_sequential: bool = True
) -> AgentState:
    """
    PASS 5: Compression & Deduplication
    
    Removes duplicate chunks, merges overlapping content, and ensures
    final context fits within token budget.
    
    Args:
        state: Current agent state
        max_total_tokens: Maximum tokens for final context
        remove_duplicates: Whether to remove near-duplicate chunks
        merge_sequential: Whether to merge sequential chunks from same chapter
    
    Returns:
        Updated state with compressed chunks
    """
    state.current_node = "compression"
    
    if not state.retrieved_chunks:
        state.errors.append("No chunks to compress")
        return state
    
    try:
        initial_count = len(state.retrieved_chunks)
        logger.info(f"🗜️  Compression: processing {initial_count} chunks...")
        
        # Step 1: Remove duplicates
        if remove_duplicates:
            duplicates = detect_duplicates(state.retrieved_chunks)
            if duplicates:
                logger.info(f"   Removing {len(duplicates)} duplicate chunks")
                for idx in duplicates:
                    state.retrieved_chunks.pop(idx)
        
        # Step 2: Merge sequential chunks from same chapter
        if merge_sequential:
            before_merge = len(state.retrieved_chunks)
            state.retrieved_chunks = merge_overlapping_chunks(state.retrieved_chunks)
            merged_count = before_merge - len(state.retrieved_chunks)
            if merged_count > 0:
                logger.info(f"   Merged {merged_count} sequential chunks")
        
        # Step 3: Calculate current token usage
        total_tokens = sum(count_tokens(c.chunk.content) for c in state.retrieved_chunks)
        logger.info(f"   Current token usage: {total_tokens}")
        
        # Step 4: Truncate if needed
        if total_tokens > max_total_tokens:
            logger.info(f"   Truncating to fit {max_total_tokens} token budget...")
            state.retrieved_chunks = truncate_chunks_to_budget(
                state.retrieved_chunks,
                max_total_tokens
            )
            
            final_tokens = sum(count_tokens(c.chunk.content) for c in state.retrieved_chunks)
            logger.info(f"   Final token usage: {final_tokens}")
        
        final_count = len(state.retrieved_chunks)
        logger.info(f"✅ Compression complete: {initial_count} → {final_count} chunks")
        
        # Update reranked_chunks for context assembly
        state.reranked_chunks = state.retrieved_chunks
        
    except Exception as e:
        state.errors.append(f"Compression failed: {str(e)}")
        logger.error(f"❌ Compression error: {e}")
    
    return state