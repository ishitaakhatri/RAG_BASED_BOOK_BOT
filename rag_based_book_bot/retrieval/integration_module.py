"""
Integration module to use 5-pass retrieval in existing app.py

Drop-in replacement for the current query_pinecone function.
"""

import os
from typing import List, Dict, Optional, Tuple
from rag_based_book_bot.agents.states import AgentState, RetrievedChunk, DocumentChunk, ParsedQuery


def integrate_five_pass_retrieval():
    """
    Factory function to create a retriever instance with proper configuration
    """
    from five_pass_retrieval import FivePassRetriever, RetrievalConfig
    
    # Configure for production use
    config = RetrievalConfig(
        coarse_top_k=60,              # Slightly smaller for speed
        rerank_top_k=12,              # Good balance
        enable_multihop=True,
        multihop_iterations=1,        # Single hop for faster response
        concepts_per_iteration=2,     # Fewer concepts
        enable_graph_expansion=True,
        cluster_epsilon=0.3,
        expand_per_cluster=2,
        final_token_budget=1800,      # Leave room for system prompt
        similarity_dedup_threshold=0.85,
        enable_summarization=False    # Disable for now (can add later)
    )
    
    return FivePassRetriever(config)


# Global retriever instance (cached)
_retriever_instance = None


def get_retriever():
    """Get or create retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = integrate_five_pass_retrieval()
    return _retriever_instance


def query_with_five_pass(
    query_text: str,
    top_k: int = 5,
    book_filter: Optional[str] = None,
    chapter_filter: Optional[str] = None,
    user_context: Optional[str] = None,
    enable_multihop: bool = True,
    enable_expansion: bool = True
) -> Tuple[str, List[Dict], Dict]:
    """
    Enhanced query function using 5-pass retrieval.
    
    This is a DROP-IN replacement for the query_pinecone function.
    
    Args:
        query_text: User's query
        top_k: Number of final results (used for display, actual retrieval is optimized)
        book_filter: Optional book title filter
        chapter_filter: Optional chapter number filter
        user_context: Optional user context
        enable_multihop: Enable multi-hop retrieval
        enable_expansion: Enable graph expansion
    
    Returns:
        - final_context: Assembled context for LLM
        - matches: List of match dicts (compatible with existing code)
        - stats: Statistics about retrieval
    """
    retriever = get_retriever()
    
    # Temporarily override config if needed
    original_multihop = retriever.config.enable_multihop
    original_expansion = retriever.config.enable_graph_expansion
    
    retriever.config.enable_multihop = enable_multihop
    retriever.config.enable_graph_expansion = enable_expansion
    
    try:
        # Run 5-pass retrieval
        final_context, results, stats = retriever.retrieve(
            query=query_text,
            book_filter=book_filter,
            chapter_filter=chapter_filter,
            user_context=user_context
        )
        
        # Convert RetrievalResult objects to match format expected by app.py
        matches = []
        for result in results[:top_k]:  # Limit to top_k for display
            match = {
                "id": result.chunk_id,
                "score": result.score,
                "metadata": result.metadata
            }
            matches.append(match)
        
        return final_context, matches, stats
    
    finally:
        # Restore original config
        retriever.config.enable_multihop = original_multihop
        retriever.config.enable_graph_expansion = original_expansion


def convert_to_agent_state(
    query_text: str,
    matches: List[Dict],
    parsed_query: ParsedQuery
) -> AgentState:
    """
    Convert matches to AgentState format for LLM reasoning
    
    This maintains compatibility with existing nodes.py
    """
    retrieved_chunks = []
    
    for match in matches:
        metadata = match.get("metadata", {})
        
        # Handle metadata lists safely
        chapter_titles = metadata.get("chapter_titles", [])
        chapter_numbers = metadata.get("chapter_numbers", [])
        section_titles = metadata.get("section_titles", [])
        
        chapter_title = chapter_titles[0] if chapter_titles else ""
        chapter_number = chapter_numbers[0] if chapter_numbers else ""
        
        # Create DocumentChunk
        chunk = DocumentChunk(
            chunk_id=match["id"],
            content=metadata.get("text", ""),
            chapter=f"{chapter_number}: {chapter_title}" if chapter_number else chapter_title,
            section=", ".join(section_titles) if section_titles else "",
            page_number=metadata.get("page_start"),
            chunk_type="code" if metadata.get("contains_code") else "text"
        )
        
        # Create RetrievedChunk
        retrieved_chunks.append(RetrievedChunk(
            chunk=chunk,
            similarity_score=match.get("score", 0.0),
            rerank_score=match.get("score", 0.0),
            relevance_percentage=round(match.get("score", 0.0) * 100, 1)
        ))
    
    # Create agent state
    state = AgentState(
        user_query=query_text,
        parsed_query=parsed_query,
        retrieved_chunks=retrieved_chunks,
        reranked_chunks=retrieved_chunks
    )
    
    return state


# =============================================================================
# STREAMLIT APP MODIFICATIONS
# =============================================================================

def get_app_modifications():
    """
    Returns the modifications needed for app.py
    
    This shows what changes to make in your existing app.py
    """
    
    modifications = """
# =============================================================================
# MODIFICATIONS FOR app.py
# =============================================================================

# 1. Add import at the top (after other imports)
from integration_module import query_with_five_pass, convert_to_agent_state, get_retriever

# 2. Add toggle in sidebar for advanced features
with st.sidebar:
    st.divider()
    with st.expander("🔬 Advanced Retrieval Settings"):
        enable_multihop = st.toggle(
            "Multi-Hop Retrieval",
            value=True,
            help="Find cross-chapter connections"
        )
        enable_expansion = st.toggle(
            "Cluster Expansion",
            value=True,
            help="Expand semantic neighborhoods"
        )
        show_retrieval_stats = st.toggle(
            "Show Retrieval Statistics",
            value=True
        )

# 3. Replace the query_pinecone call in the main search button handler
if st.button("🔍 Search", type="primary", disabled=not query):
    if query:
        with st.spinner("🔎 Searching with 5-pass retrieval..."):
            # OLD CODE:
            # matches = query_pinecone(query, top_k, book_filter, chapter_filter)
            
            # NEW CODE:
            final_context, matches, stats = query_with_five_pass(
                query_text=query,
                top_k=top_k,
                book_filter=book_filter if book_filter != "All Books" else None,
                chapter_filter=chapter_filter if chapter_filter else None,
                enable_multihop=enable_multihop,
                enable_expansion=enable_expansion
            )
        
        # Show retrieval statistics if enabled
        if show_retrieval_stats:
            with st.expander("📊 Retrieval Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pass 1 (Coarse)", stats.get("pass1_count", 0))
                    st.metric("Pass 2 (Rerank)", stats.get("pass2_count", 0))
                with col2:
                    st.metric("Pass 3 (Multi-hop)", stats.get("pass3_count", 0))
                    st.metric("Pass 4 (Expansion)", stats.get("pass4_count", 0))
                with col3:
                    st.metric("Final Chunks", stats.get("pass5_count", 0))
                    st.metric("Final Tokens", stats.get("final_tokens", 0))
                
                st.info(f"🗑️ Removed {stats.get('duplicate_removed', 0)} duplicate chunks")
        
        # Rest of the code remains the same...
        # Add to history, display results, etc.

# 4. (Optional) Add a "Explain Retrieval" button
if matches:
    if st.button("🔍 Explain How Results Were Found"):
        retriever = get_retriever()
        
        st.markdown("### Retrieval Process Breakdown")
        
        st.markdown(\"\"\"
        **Pass 1: Coarse Search**
        - Retrieved {} broad candidates using semantic embeddings
        - Fast vector search across entire knowledge base
        
        **Pass 2: Cross-Encoder Reranking**
        - Reranked candidates using precise cross-encoder
        - Selected top {} most relevant chunks
        
        **Pass 3: Multi-Hop Discovery** {}
        - Extracted key concepts from top results
        - Searched for related content in other chapters
        - Found {} additional cross-chapter connections
        
        **Pass 4: Cluster Expansion** {}
        - Grouped results into semantic clusters
        - Expanded each cluster to find nearby content
        - Added {} semantically related chunks
        
        **Pass 5: Final Assembly**
        - Removed {} duplicate chunks
        - Assembled final context within token budget
        - Final output: {} chunks, {} tokens
        \"\"\".format(
            stats.get("pass1_count", 0),
            stats.get("pass2_count", 0),
            "✅ Enabled" if enable_multihop else "❌ Disabled",
            stats.get("pass3_count", 0),
            "✅ Enabled" if enable_expansion else "❌ Disabled",
            stats.get("pass4_count", 0),
            stats.get("duplicate_removed", 0),
            stats.get("pass5_count", 0),
            stats.get("final_tokens", 0)
        ))
    """
    
    return modifications


# =============================================================================
# INSTALLATION INSTRUCTIONS
# =============================================================================

def get_installation_instructions():
    """Installation guide for the 5-pass retrieval system"""
    
    instructions = """
# =============================================================================
# INSTALLATION INSTRUCTIONS
# =============================================================================

## Step 1: Install Additional Dependencies

Add to requirements.txt:
```
# 5-Pass Retrieval
sentence-transformers>=2.7.0
scikit-learn>=1.5.0
```

## Step 2: Add Files to Project

1. Save `five_pass_retrieval.py` to:
   `rag_based_book_bot/retrieval/five_pass_retrieval.py`

2. Save `integration_module.py` to:
   `rag_based_book_bot/retrieval/integration_module.py`

3. Create `__init__.py`:
   `rag_based_book_bot/retrieval/__init__.py`

## Step 3: Modify app.py

Follow the modifications shown in get_app_modifications()

## Step 4: Test

Run:
```bash
streamlit run app.py
```

The retrieval will now use the 5-pass pipeline automatically!

## Performance Notes

- First query may be slower (model loading)
- Subsequent queries are fast (cached models)
- Multi-hop adds ~2-3 seconds per query
- Cluster expansion adds ~1-2 seconds
- Total: 5-10 seconds for complex queries

## Configuration Tuning

Edit RetrievalConfig in integration_module.py:

- coarse_top_k: Higher = better recall, slower
- rerank_top_k: Balance precision/speed
- enable_multihop: Enable for complex queries
- multihop_iterations: 1-2 recommended
- enable_graph_expansion: Enable for book collections
- final_token_budget: 1500-2500 recommended
    """
    
    return instructions


if __name__ == "__main__":
    print(get_app_modifications())
    print("\n" + "="*60 + "\n")
    print(get_installation_instructions())