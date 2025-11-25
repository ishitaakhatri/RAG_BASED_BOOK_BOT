"""
Streamlit App for RAG-Based Book Bot
✅ NOW WITH 5-PASS RETRIEVAL INTEGRATED
"""
import streamlit as st
import os
import sys
from dotenv import load_dotenv

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="RAG Book Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# ✅ IMPORTS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rag_based_book_bot.document_ingestion.book_ingestion import BookIngestionService

# Import agent components
from rag_based_book_bot.agents.nodes import llm_reasoning_node
from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent
)

# ✅ NEW: Import 5-pass retrieval
try:
    from rag_based_book_bot.retrieval.integration_module import (
        query_with_five_pass,
        get_retriever
    )
    FIVE_PASS_AVAILABLE = True
except ImportError:
    FIVE_PASS_AVAILABLE = False
    print("⚠️ 5-Pass retrieval not available. Using standard retrieval.")

# Custom CSS
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
    .stats-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'ingestion_complete' not in st.session_state:
    st.session_state.ingestion_complete = False
if 'use_five_pass' not in st.session_state:
    st.session_state.use_five_pass = FIVE_PASS_AVAILABLE
if 'retrieval_stats' not in st.session_state:
    st.session_state.retrieval_stats = None


# =============================================================================
# HELPER FUNCTIONS
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


def query_pinecone_standard(query_text, top_k=5, book_filter=None, chapter_filter=None):
    """Standard Pinecone query (fallback)"""
    pc, index = initialize_pinecone()
    if index is None:
        return []
    
    model = load_embedding_model()
    namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
    
    # Generate embedding
    query_embedding = model.encode(query_text).tolist()
    
    # Build filter
    filter_dict = {}
    if book_filter and book_filter != "All Books":
        filter_dict["book_title"] = book_filter
    if chapter_filter:
        filter_dict["chapter_numbers"] = {"$in": [chapter_filter]}
    
    # Query
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        return results.get("matches", [])
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return []


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


def ingest_book(pdf_path, book_title, author):
    """Ingest a book into Pinecone using BookIngestionService"""
    try:
        with st.spinner(f"🔄 Ingesting '{book_title}'... This may take several minutes."):
            service = BookIngestionService()
            metadata = service.ingest_book(pdf_path)
            
            return True, {
                "title": metadata.title,
                "author": metadata.author,
                "source": metadata.source_file
            }
    except Exception as e:
        return False, str(e)


def parse_query_simple(query_text):
    """Simple query parsing"""
    query_lower = query_text.lower()
    
    # Detect intent
    if any(w in query_lower for w in ['implement', 'code', 'write', 'build', 'create']):
        intent = QueryIntent.CODE_REQUEST
    elif any(w in query_lower for w in ['difference', 'compare', 'vs', 'versus']):
        intent = QueryIntent.COMPARISON
    elif any(w in query_lower for w in ['error', 'bug', 'fix', 'wrong', 'not working']):
        intent = QueryIntent.DEBUGGING
    elif any(w in query_lower for w in ['tutorial', 'walk through', 'step by step', 'guide']):
        intent = QueryIntent.TUTORIAL
    else:
        intent = QueryIntent.CONCEPTUAL
    
    return ParsedQuery(
        raw_query=query_text,
        intent=intent,
        topics=[],
        keywords=query_text.split(),
        complexity_hint="intermediate"
    )


def convert_pinecone_to_chunks(matches):
    """Convert Pinecone matches to RetrievedChunk objects"""
    chunks = []
    
    for match in matches:
        metadata = match.get("metadata", {})
        
        # Handle metadata lists safely
        chapter_numbers = metadata.get("chapter_numbers", [])
        chapter_titles = metadata.get("chapter_titles", [])
        section_titles = metadata.get("section_titles", [])
        
        chapter_num = chapter_numbers[0] if chapter_numbers else ""
        chapter_title = chapter_titles[0] if chapter_titles else ""
        
        # Create DocumentChunk
        chunk = DocumentChunk(
            chunk_id=match["id"],
            content=metadata.get("text", ""),
            chapter=f"{chapter_num}: {chapter_title}" if chapter_num else chapter_title,
            section=", ".join(section_titles) if section_titles else "",
            page_number=metadata.get("page_start"),
            chunk_type="code" if metadata.get("contains_code") else "text"
        )
        
        # Create RetrievedChunk
        retrieved_chunk = RetrievedChunk(
            chunk=chunk,
            similarity_score=match.get("score", 0.0),
            rerank_score=match.get("score", 0.0),
            relevance_percentage=round(match.get("score", 0.0) * 100, 1)
        )
        
        chunks.append(retrieved_chunk)
    
    return chunks


def generate_answer(query_text, matches):
    """Generate AI answer using LLM reasoning node"""
    try:
        # Parse query
        parsed_query = parse_query_simple(query_text)
        
        # Convert Pinecone matches to RetrievedChunk objects
        retrieved_chunks = convert_pinecone_to_chunks(matches)
        
        # Create agent state
        state = AgentState(
            user_query=query_text,
            parsed_query=parsed_query,
            reranked_chunks=retrieved_chunks
        )
        
        # Run LLM reasoning node
        state = llm_reasoning_node(state)
        
        # Check for errors
        if state.errors:
            return None, f"Error generating answer: {', '.join(state.errors)}"
        
        return state.response, None
        
    except Exception as e:
        return None, f"Answer generation failed: {str(e)}"


# =============================================================================
# SIDEBAR - INGESTION & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("📚 Book Management")
    
    with st.expander("➕ Ingest New Book", expanded=False):
        st.markdown("Upload a PDF book to add it to the knowledge base.")
        
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
        
        if st.button("🚀 Ingest Book", type="primary", disabled=not uploaded_file):
            if uploaded_file and book_title:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest using BookIngestionService
                success, result = ingest_book(temp_path, book_title, author or "Unknown")
                
                # Cleanup
                os.remove(temp_path)
                
                if success:
                    st.success(f"✅ Successfully ingested '{book_title}'!")
                    st.info(f"📖 Book: {result['title']}")
                    st.session_state.ingestion_complete = True
                else:
                    st.error(f"❌ Ingestion failed: {result}")
            else:
                st.warning("Please provide both PDF and book title")
    
    st.divider()
    
    # ✅ NEW: Advanced Retrieval Settings
    if FIVE_PASS_AVAILABLE:
        with st.expander("🔬 Advanced Retrieval Settings", expanded=True):
            st.markdown("**5-Pass Retrieval Pipeline**")
            
            use_five_pass = st.toggle(
                "Enable 5-Pass Retrieval",
                value=True,
                help="Use advanced multi-pass retrieval for better results"
            )
            st.session_state.use_five_pass = use_five_pass
            
            if use_five_pass:
                st.markdown("---")
                
                enable_multihop = st.toggle(
                    "Multi-Hop Discovery",
                    value=True,
                    help="Find cross-chapter connections automatically"
                )
                
                enable_expansion = st.toggle(
                    "Cluster Expansion",
                    value=True,
                    help="Expand semantic neighborhoods for comprehensive results"
                )
                
                show_retrieval_stats = st.toggle(
                    "Show Statistics",
                    value=True,
                    help="Display detailed retrieval metrics"
                )
                
                # Store in session state
                st.session_state.enable_multihop = enable_multihop
                st.session_state.enable_expansion = enable_expansion
                st.session_state.show_retrieval_stats = show_retrieval_stats
                
                st.info("💡 5-Pass Retrieval provides:\n- Cross-chapter discovery\n- Reduced noise\n- Optimized token usage")
            else:
                st.warning("Using standard retrieval (faster but less comprehensive)")
    else:
        st.warning("⚠️ 5-Pass retrieval not available. Install dependencies:\n```\npip install sentence-transformers scikit-learn\n```")
    
    st.divider()
    
    # Available books
    st.subheader("📖 Available Books")
    books = get_available_books()
    
    if books:
        for book in books:
            st.markdown(f"• {book}")
    else:
        st.info("No books ingested yet. Upload a book above to get started!")
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        st.markdown("**Pinecone Configuration**")
        st.text(f"Index: {os.getenv('PINECONE_INDEX_NAME', 'coding-books')}")
        st.text(f"Namespace: {os.getenv('PINECONE_NAMESPACE', 'books_rag')}")
        
        # Check connection
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
# MAIN INTERFACE - QUERY SYSTEM
# =============================================================================

st.title("🤖 RAG-Based Book Bot")

if FIVE_PASS_AVAILABLE and st.session_state.get('use_five_pass', False):
    st.markdown("Ask questions and get AI-powered answers with **5-Pass Retrieval** 🚀")
else:
    st.markdown("Ask questions about your ingested books and get AI-powered answers with citations!")

# Check if any books are available
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
    top_k = st.slider("Results", 3, 10, 5, help="Number of chunks to retrieve")

# Filters
col1, col2 = st.columns(2)

with col1:
    book_filter = st.selectbox(
        "Filter by Book",
        ["All Books"] + books,
        help="Optionally filter results to a specific book"
    )

with col2:
    chapter_filter = st.text_input(
        "Filter by Chapter",
        placeholder="e.g., 1, 10, etc.",
        help="Optionally filter to a specific chapter number"
    )

# ✅ MODIFIED: Query button with 5-pass integration
if st.button("🔍 Search", type="primary", disabled=not query):
    if query:
        # Determine which retrieval method to use
        use_five_pass = FIVE_PASS_AVAILABLE and st.session_state.get('use_five_pass', False)
        
        if use_five_pass:
            # ✅ NEW: Use 5-pass retrieval
            with st.spinner("🔎 Searching with 5-pass retrieval..."):
                try:
                    final_context, matches, stats = query_with_five_pass(
                        query_text=query,
                        top_k=top_k,
                        book_filter=book_filter if book_filter != "All Books" else None,
                        chapter_filter=chapter_filter if chapter_filter else None,
                        enable_multihop=st.session_state.get('enable_multihop', True),
                        enable_expansion=st.session_state.get('enable_expansion', True)
                    )
                    st.session_state.retrieval_stats = stats
                except Exception as e:
                    st.error(f"❌ 5-Pass retrieval failed: {str(e)}\nFalling back to standard retrieval...")
                    matches = query_pinecone_standard(
                        query,
                        top_k=top_k,
                        book_filter=book_filter if book_filter != "All Books" else None,
                        chapter_filter=chapter_filter if chapter_filter else None
                    )
                    st.session_state.retrieval_stats = None
        else:
            # Standard retrieval
            with st.spinner("🔎 Searching knowledge base..."):
                matches = query_pinecone_standard(
                    query,
                    top_k=top_k,
                    book_filter=book_filter if book_filter != "All Books" else None,
                    chapter_filter=chapter_filter if chapter_filter else None
                )
                st.session_state.retrieval_stats = None
        
        # Add to history
        st.session_state.query_history.insert(0, {
            'query': query,
            'results': len(matches),
            'book_filter': book_filter,
            'used_five_pass': use_five_pass
        })
        
        if not matches:
            st.warning("🤷 No relevant results found. Try rephrasing your question or removing filters.")
        else:
            st.success(f"✅ Found {len(matches)} relevant chunks")
            
            # ✅ NEW: Show retrieval statistics
            if use_five_pass and st.session_state.retrieval_stats and st.session_state.get('show_retrieval_stats', True):
                stats = st.session_state.retrieval_stats
                
                with st.expander("📊 5-Pass Retrieval Statistics", expanded=False):
                    st.markdown("### Pipeline Breakdown")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Pass 1: Coarse Search", stats.get("pass1_count", 0))
                        st.caption("Broad semantic search")
                        
                        st.metric("Pass 2: Reranking", stats.get("pass2_count", 0))
                        st.caption("Cross-encoder precision")
                    
                    with col2:
                        st.metric("Pass 3: Multi-Hop", stats.get("pass3_count", 0))
                        st.caption("Cross-chapter discovery")
                        
                        st.metric("Pass 4: Expansion", stats.get("pass4_count", 0))
                        st.caption("Cluster neighborhoods")
                    
                    with col3:
                        st.metric("Final Chunks", stats.get("pass5_count", 0))
                        st.caption("After deduplication")
                        
                        st.metric("Final Tokens", stats.get("final_tokens", 0))
                        st.caption("Optimized context")
                    
                    # Summary
                    st.markdown("---")
                    dedup = stats.get("duplicate_removed", 0)
                    if dedup > 0:
                        st.info(f"🗑️ Removed {dedup} duplicate/similar chunks")
                    
                    if stats.get("pass3_count", 0) > 0:
                        st.success(f"🔗 Found {stats['pass3_count']} cross-chapter connections")
            
            # Generate AI answer
            st.divider()
            st.subheader("💬 AI-Generated Answer")
            
            with st.spinner("🤖 Generating answer..."):
                response, error = generate_answer(query, matches)
            
            if error:
                st.error(f"❌ {error}")
            elif response:
                # Display answer in a nice box
                st.markdown(
                    f'<div class="answer-box">'
                    f'<h4>Answer:</h4>'
                    f'<p>{response.answer}</p>'
                    f'<br><small>Confidence: {response.confidence:.1%}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Show code snippets if available
                if response.code_snippets:
                    st.markdown("#### 💻 Code Examples")
                    for i, code in enumerate(response.code_snippets, 1):
                        st.code(code, language="python")
            
            # Display sources
            st.divider()
            st.subheader("📋 Retrieved Sources")
            
            for i, match in enumerate(matches):
                metadata = match.get("metadata", {})
                score = match.get("score", 0.0)
                
                # Extract metadata
                book_title = metadata.get("book_title", "Unknown")
                chapter_titles = metadata.get("chapter_titles", [])
                chapter_numbers = metadata.get("chapter_numbers", [])
                section_titles = metadata.get("section_titles", [])
                
                chapter_title = chapter_titles[0] if chapter_titles else "Unknown"
                chapter_number = chapter_numbers[0] if chapter_numbers else "N/A"
                
                page_start = metadata.get("page_start", "?")
                page_end = metadata.get("page_end", "?")
                contains_code = metadata.get("contains_code", False)
                
                # Display source
                with st.expander(
                    f"**Source {i+1}** | {book_title} - Ch.{chapter_number} | Relevance: {score:.2%}",
                    expanded=(i == 0)
                ):
                    # Metadata badges
                    st.markdown(
                        f'<span class="metadata-badge">📚 {book_title}</span>'
                        f'<span class="metadata-badge">📖 Ch.{chapter_number}: {chapter_title}</span>'
                        f'<span class="metadata-badge">📄 pp.{page_start}-{page_end}</span>'
                        f'<span class="metadata-badge">{"💻 Code" if contains_code else "📝 Text"}</span>'
                        f'<span class="metadata-badge">🎯 {score:.2%}</span>',
                        unsafe_allow_html=True
                    )
                    
                    if section_titles:
                        section_str = " → ".join(section_titles)
                        st.caption(f"Sections: {section_str}")
                    
                    st.divider()
                    
                    # Show content preview
                    text_preview = metadata.get("text", "")[:500]
                    if text_preview:
                        st.text_area("Content Preview", text_preview, height=150, disabled=True)

# Query History
if st.session_state.query_history:
    st.divider()
    st.subheader("🕐 Query History")
    
    for i, item in enumerate(st.session_state.query_history[:5]):
        method = "🚀 5-Pass" if item.get('used_five_pass', False) else "📍 Standard"
        st.markdown(f"{i+1}. **{item['query']}** - {item['results']} results ({item['book_filter']}) {method}")

# Footer
st.divider()
footer_text = "Built with Streamlit • Powered by Pinecone & OpenAI"
if FIVE_PASS_AVAILABLE and st.session_state.get('use_five_pass', False):
    footer_text += " • 5-Pass Retrieval System 🚀"
st.caption(footer_text)