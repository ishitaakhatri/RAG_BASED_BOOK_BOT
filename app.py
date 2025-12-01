"""
Streamlit App for RAG-Based Book Bot - WITH COMPARISON MODE
"""
import streamlit as st
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
from rag_based_book_bot.agents.graph import build_query_graph, build_query_graph_baseline
from rag_based_book_bot.agents.states import AgentState
from rag_based_book_bot.agents.comparison_utils import compare_pipeline_results

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="RAG Book Bot - Comparison Mode",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Custom CSS (same as before)
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metadata-badge {
        display: inline-block;
        background-color: #e1f5ff;
        color: #01579b;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .baseline-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .comparison-header {
        background: linear-gradient(90deg, #ffc107 0%, #28a745 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'ingestion_complete' not in st.session_state:
    st.session_state.ingestion_complete = False
if 'last_ingestion_stats' not in st.session_state:
    st.session_state.last_ingestion_stats = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = True  # Default to comparison mode


# =============================================================================
# HELPER FUNCTIONS (same as before)
# =============================================================================

@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone connection"""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            st.error("❌ PINECONE_API_KEY not found in .env file")
            return None, None
        
        pc = Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "coding-books")
        
        try:
            index = pc.Index(index_name)
            return pc, index
        except Exception as e:
            st.error(f"❌ Pinecone index '{index_name}' not found")
            return None, None
    except Exception as e:
        st.error(f"❌ Failed to connect to Pinecone: {str(e)}")
        return None, None


@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_enhanced_ingestor():
    """Get cached semantic ingestor instance"""
    config = IngestorConfig(
        similarity_threshold=0.75, 
        min_chunk_size=200,
        max_chunk_size=1500,
        debug=False
    )
    return EnhancedBookIngestorPaddle(config=config)


def get_available_books():
    """Get list of books in Pinecone"""
    pc, index = initialize_pinecone()
    if index is None:
        return []
    
    namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
    try:
        results = index.query(
            vector=[0.0] * 384,
            top_k=100,
            namespace=namespace,
            include_metadata=True
        )
        
        books = set()
        for match in results.get("matches", []):
            book_title = match.get("metadata", {}).get("book_title")
            if book_title:
                books.add(book_title)
        
        return sorted(list(books))
    except Exception as e:
        st.warning(f"Could not retrieve books: {str(e)}")
        return []


def ingest_book_enhanced(pdf_path, book_title, author):
    """Ingest a book into Pinecone using Semantic Chunking"""
    try:
        ingestor = get_enhanced_ingestor()
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("📄 Step 1/3: Extracting text from PDF...")
        progress_bar.progress(0.33)
        
        progress_text.text("🧠 Step 2/3: Semantic chunking (grouping related content)...")
        progress_bar.progress(0.66)
        
        result = ingestor.ingest_book(
            pdf_path=pdf_path,
            book_title=book_title,
            author=author
        )
        
        progress_text.text("✅ Step 3/3: Ingestion complete!")
        progress_bar.progress(1.0)
        
        return True, result
        
    except Exception as e:
        st.error(f"❌ Semantic ingestion error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, str(e)


def run_both_pipelines(
    query_text: str,
    pass1_k: int = 50,
    pass2_k: int = 15,
    pass3_enabled: bool = True,
    pass3_max_hops: int = 2,
    max_tokens: int = 2500,
    book_filter: str = None
):
    """
    Run BOTH pipelines and return comparison results.
    
    Returns:
        tuple: (state_baseline, state_reranked, comparison_metrics)
    """
    print(f"\n{'='*70}")
    print(f"RUNNING COMPARISON: BASELINE vs RERANKED")
    print(f"{'='*70}")
    
    # ========== PIPELINE A: WITH RERANKING ==========
    print(f"\n[PIPELINE A] WITH Reranking (50 → 15)")
    state_reranked = AgentState(
        user_query=query_text,
        vector_search_top_k=pass1_k,  # 50
        pass2_k=pass2_k,  # 15
        pass3_enabled=pass3_enabled,
        pass3_max_hops=pass3_max_hops,
        max_tokens=max_tokens,
        book_filter=book_filter if book_filter and book_filter != "All Books" else None
    )
    
    graph_reranked = build_query_graph()
    result_reranked = graph_reranked.execute(state_reranked)
    
    if not result_reranked.success:
        return None, None, {"error": f"Reranked pipeline failed: {result_reranked.error_message}"}
    
    state_reranked = result_reranked.final_state
    
    # ========== PIPELINE B: WITHOUT RERANKING ==========
    print(f"\n[PIPELINE B] WITHOUT Reranking (15 only)")
    state_baseline = AgentState(
        user_query=query_text,
        vector_search_top_k=15,  # Only 15
        pass2_k=15,  # Not used
        pass3_enabled=pass3_enabled,
        pass3_max_hops=pass3_max_hops,
        max_tokens=max_tokens,
        book_filter=book_filter if book_filter and book_filter != "All Books" else None
    )
    
    graph_baseline = build_query_graph_baseline()
    result_baseline = graph_baseline.execute(state_baseline)
    
    if not result_baseline.success:
        return None, None, {"error": f"Baseline pipeline failed: {result_baseline.error_message}"}
    
    state_baseline = result_baseline.final_state
    
    # ========== COMPARE RESULTS ==========
    comparison = compare_pipeline_results(state_baseline, state_reranked)
    
    return state_baseline, state_reranked, comparison


# =============================================================================
# SIDEBAR (same as before)
# =============================================================================

with st.sidebar:
    st.title("📚 Book Management")
    
    with st.expander("➕ Ingest New Book (Semantic Chunking)", expanded=False):
        st.markdown("**Semantic Chunking Features:**")
        st.markdown("- 🧠 AI-powered topic boundary detection")
        st.markdown("- 📊 Groups semantically related content")
        st.markdown("- 💻 Keeps code + explanations together")
        st.markdown("- ⚡ No complex hierarchy detection needed")
        
        st.divider()
        
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Upload a technical book in PDF format"
        )
        
        book_title = st.text_input(
            "Book Title",
            placeholder="e.g., Hands-On Machine Learning"
        )
        
        author = st.text_input(
            "Author",
            placeholder="e.g., Aurélien Géron"
        )
        
        if st.button("🚀 Ingest Book (Enhanced)", type="primary", disabled=not uploaded_file):
            if uploaded_file and book_title:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                success, result = ingest_book_enhanced(temp_path, book_title, author or "Unknown")
                
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                if success:
                    st.success(f"✅ Successfully ingested '{book_title}'!")
                    
                    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                    st.markdown("**📊 Ingestion Statistics:**")
                    st.markdown(f"- **Title:** {result.get('title', 'N/A')}")
                    st.markdown(f"- **Author:** {result.get('author', 'N/A')}")
                    st.markdown(f"- **Total Pages:** {result.get('total_pages', 'N/A')}")
                    st.markdown(f"- **Total Chunks:** {result.get('total_chunks', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.ingestion_complete = True
                    st.session_state.last_ingestion_stats = result
                else:
                    st.error(f"❌ Ingestion failed: {result}")
            else:
                st.warning("Please provide both PDF and book title")
    
    st.divider()
    
    if st.session_state.last_ingestion_stats:
        with st.expander("📊 Last Ingestion Stats", expanded=False):
            stats = st.session_state.last_ingestion_stats
            st.json({
                "title": stats.get("title"),
                "author": stats.get("author"),
                "total_pages": stats.get("total_pages"),
                "total_chunks": stats.get("total_chunks")
            })
    
    st.divider()
    
    st.subheader("📖 Available Books")
    books = get_available_books()
    
    if books:
        for book in books:
            st.markdown(f"• {book}")
    else:
        st.info("No books ingested yet. Upload a book above to get started!")
    
    st.divider()
    
    # Comparison Mode Toggle
    st.subheader("🔬 Comparison Mode")
    comparison_mode = st.checkbox(
        "Enable Pipeline Comparison",
        value=True,
        help="Compare WITH vs WITHOUT cross-encoder reranking"
    )
    st.session_state.comparison_mode = comparison_mode
    
    if comparison_mode:
        st.info("📊 Will run both pipelines:\n- Baseline (15 chunks)\n- Reranked (50→15 chunks)")
    
    st.divider()
    
    with st.expander("⚙️ Pipeline Settings"):
        st.markdown("**Pass 1:** Vector Search")
        pass1_k = st.slider("Initial candidates (for reranked)", 30, 100, 50)
        
        st.markdown("**Pass 2:** Cross-Encoder Reranking")
        pass2_k = st.slider("After reranking", 10, 30, 15)
        
        st.markdown("**Pass 3:** Query Expansion")
        pass3_enabled = st.checkbox("Enable multi-hop", value=True)
        pass3_max_hops = st.slider("Max hops", 1, 3, 2)
        
        st.markdown("**Pass 5:** Compression")
        max_tokens = st.slider("Max context tokens", 1500, 4000, 2500)
        
        pc, index = initialize_pinecone()
        if index:
            st.success("✅ Connected to Pinecone")
            try:
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                st.info(f"📊 Total chunks: {total_vectors}")
            except:
                pass
        else:
            st.error("❌ Not connected to Pinecone")


# =============================================================================
# MAIN INTERFACE
# =============================================================================

st.title("🤖 RAG-Based Book Bot")

if st.session_state.comparison_mode:
    st.markdown("**🔬 COMPARISON MODE** • See how cross-encoder reranking improves results!")
else:
    st.markdown("**Standard Mode** • Using reranked pipeline")

books = get_available_books()

if not books:
    st.warning("⚠️ No books found in the knowledge base. Please ingest a book using the sidebar first.")
    st.stop()

# Query interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Ask a question:",
        placeholder="e.g., How do I implement a CNN in Keras?",
        help="Ask any question about the ingested books"
    )

with col2:
    top_k = st.slider("Results", 3, 10, 5, help="Number of chunks to display")

# Filters
col1, col2 = st.columns(2)

with col1:
    book_filter = st.selectbox(
        "Filter by Book",
        ["All Books"] + books,
        help="Optionally filter results to a specific book"
    )

# Query button
if st.button("🔍 Search & Compare", type="primary", disabled=not query):
    if query:
        if st.session_state.comparison_mode:
            # RUN COMPARISON
            with st.spinner("🔬 Running comparison (2 pipelines)..."):
                state_baseline, state_reranked, comparison = run_both_pipelines(
                    query_text=query,
                    pass1_k=pass1_k,
                    pass2_k=pass2_k,
                    pass3_enabled=pass3_enabled,
                    pass3_max_hops=pass3_max_hops,
                    max_tokens=max_tokens,
                    book_filter=book_filter
                )
            
            if 'error' in comparison:
                st.error(f"❌ {comparison['error']}")
            else:
                # ========== COMPARISON HEADER ==========
                st.markdown('<div class="comparison-header">📊 PIPELINE COMPARISON RESULTS</div>', unsafe_allow_html=True)
                
                # ========== SUMMARY METRICS ==========
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Chunk Overlap",
                        f"{comparison['overlap']['overlap_count']}/15",
                        delta=f"{comparison['overlap']['overlap_percentage']}%"
                    )
                
                with col2:
                    st.metric(
                        "Different Chunks",
                        comparison['overlap']['unique_to_b_count'],
                        delta="Reranked found new"
                    )
                
                with col3:
                    st.metric(
                        "Avg Score Improvement",
                        f"{comparison['stats_reranked']['avg_relevance']:.1f}%",
                        delta=f"+{comparison['summary']['avg_score_improvement']}%"
                    )
                
                with col4:
                    biggest_change = comparison['summary']['biggest_rank_change']
                    st.metric(
                        "Biggest Rank Jump",
                        f"±{abs(biggest_change)}",
                        delta="positions moved"
                    )
                
                st.divider()
                
                # ========== SIDE-BY-SIDE CHUNKS ==========
                st.subheader("📄 Top 5 Chunks Comparison")
                
                col_baseline, col_reranked = st.columns(2)
                
                with col_baseline:
                    st.markdown("### 📋 WITHOUT Reranking (Baseline)")
                    st.caption("Top 15 by vector similarity only")
                    
                    if state_baseline and state_baseline.reranked_chunks:
                        for i, rc in enumerate(state_baseline.reranked_chunks[:5], 1):
                            with st.expander(f"Chunk {i} - Score: {rc.relevance_percentage:.1f}%"):
                                st.markdown(f"**Chapter:** {rc.chunk.chapter}")
                                st.markdown(f"**Page:** {rc.chunk.page_number}")
                                st.markdown(f"**Type:** {rc.chunk.chunk_type}")
                                st.markdown(f"**Similarity:** {rc.similarity_score:.3f}")
                                st.text_area(
                                    "Content",
                                    rc.chunk.content[:400] + "...",
                                    height=150,
                                    key=f"baseline_{i}"
                                )
                    else:
                        st.warning("No baseline chunks")
                
                with col_reranked:
                    st.markdown("### 🎯 WITH Reranking")
                    st.caption("Top 15 from 50 candidates, reranked")
                    
                    if state_reranked and state_reranked.reranked_chunks:
                        for i, rc in enumerate(state_reranked.reranked_chunks[:5], 1):
                            # Check if this chunk is new (not in baseline top 15)
                            is_new = rc.chunk.chunk_id in comparison['overlap']['unique_to_b_ids']
                            badge = "🆕" if is_new else ""
                            
                            with st.expander(f"Chunk {i} {badge} - Score: {rc.relevance_percentage:.1f}%"):
                                st.markdown(f"**Chapter:** {rc.chunk.chapter}")
                                st.markdown(f"**Page:** {rc.chunk.page_number}")
                                st.markdown(f"**Type:** {rc.chunk.chunk_type}")
                                st.markdown(f"**Rerank Score:** {rc.rerank_score:.3f}")
                                if is_new:
                                    st.success("✨ This chunk was NOT in baseline top 15!")
                                st.text_area(
                                    "Content",
                                    rc.chunk.content[:400] + "...",
                                    height=150,
                                    key=f"reranked_{i}"
                                )
                    else:
                        st.warning("No reranked chunks")
                
                st.divider()
                
                # ========== RANK CHANGES ==========
                with st.expander("📊 Detailed Rank Changes", expanded=False):
                    if comparison['rank_changes']:
                        st.markdown("**Chunks that appear in both pipelines:**")
                        
                        for change_info in comparison['rank_changes'][:10]:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.text(f"Chunk: {change_info['chunk_id'][:20]}...")
                            
                            with col2:
                                st.text(f"Baseline: #{change_info['rank_baseline'] + 1}")
                            
                            with col3:
                                change = change_info['change']
                                if change > 0:
                                    st.success(f"Reranked: #{change_info['rank_reranked'] + 1} (⬆️ +{change})")
                                elif change < 0:
                                    st.error(f"Reranked: #{change_info['rank_reranked'] + 1} (⬇️ {change})")
                                else:
                                    st.info(f"Reranked: #{change_info['rank_reranked'] + 1} (→)")
                    else:
                        st.info("No common chunks to compare")
                
                st.divider()
                
                # ========== ANSWERS COMPARISON ==========
                st.subheader("💬 Generated Answers Comparison")
                
                col_baseline, col_reranked = st.columns(2)
                
                with col_baseline:
                    st.markdown("### 📋 Baseline Answer")
                    if state_baseline.response:
                        st.markdown('<div class="baseline-box">', unsafe_allow_html=True)
                        st.write(state_baseline.response.answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.caption(f"Length: {comparison['answers']['baseline_length']} chars")
                    else:
                        st.warning("No baseline answer generated")
                
                with col_reranked:
                    st.markdown("### 🎯 Reranked Answer")
                    if state_reranked.response:
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.write(state_reranked.response.answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.caption(f"Length: {comparison['answers']['reranked_length']} chars")
                    else:
                        st.warning("No reranked answer generated")
        
        else:
            # STANDARD MODE (original functionality)
            st.info("Standard mode - use comparison mode to see the difference!")

# Query History
if st.session_state.query_history:
    st.divider()
    st.subheader("🕐 Query History")
    
    for i, item in enumerate(st.session_state.query_history[:5]):
        st.markdown(f"{i+1}. **{item['query']}** ({item['book_filter']})")

# Footer
st.divider()
st.caption("Built with Streamlit • Comparison Mode for Research • Powered by Pinecone & OpenAI")