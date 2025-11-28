"""
Streamlit App for RAG-Based Book Bot
Using enhanced_ingestion.py for PDF processing
WITH INTEGRATED ANSWER GENERATION
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
from rag_based_book_bot.agents.nodes import llm_reasoning_node
from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent
)

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="RAG Book Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

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
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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


def query_pinecone(query_text, top_k=5, book_filter=None, chapter_filter=None):
    """Query Pinecone for relevant chunks"""
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


def ingest_book_enhanced(pdf_path, book_title, author):
    """Ingest a book into Pinecone using Semantic Chunking"""
    try:
        # Get the ingestor (now returns SemanticBookIngestor)
        ingestor = get_enhanced_ingestor()
        
        # Create progress container
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Step 1: Extract text
        progress_text.text("📄 Step 1/3: Extracting text from PDF...")
        progress_bar.progress(0.33)
        
        # Step 2: Semantic chunking
        progress_text.text("🧠 Step 2/3: Semantic chunking (grouping related content)...")
        progress_bar.progress(0.66)
        
        # Ingest the book
        result = ingestor.ingest_book(
            pdf_path=pdf_path,
            book_title=book_title,
            author=author
        )
        
        # Step 3: Complete
        progress_text.text("✅ Step 3/3: Ingestion complete!")
        progress_bar.progress(1.0)
        
        return True, result
        
    except Exception as e:
        st.error(f"❌ Semantic ingestion error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
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


def generate_answer(query_text, matches, pass2_k=15, pass3_enabled=True, max_tokens=2500):
    """
    Generate AI answer using FULL 5-pass retrieval pipeline
    
    Returns:
        tuple: (response, error, final_state)
    """
    try:
        from rag_based_book_bot.agents.nodes import (
            user_query_node,
            reranking_node,
            multi_hop_expansion_node,
            cluster_expansion_node,
            context_assembly_node,
            llm_reasoning_node
        )
        
        # Initialize state
        state = AgentState(user_query=query_text)
        
        # Parse query
        state = user_query_node(state)
        
        # PASS 1 - Already done via UI
        state.retrieved_chunks = convert_pinecone_to_chunks(matches)
        print(f"\n[PASS 1] Vector Search")
        print(f"  → Retrieved {len(state.retrieved_chunks)} candidates")
        
        # PASS 2 - Cross-Encoder Reranking
        state = reranking_node(state, top_k=pass2_k)
        
        # PASS 3 - Multi-Hop Expansion (optional)
        if pass3_enabled:
            state = multi_hop_expansion_node(state, max_hops=2)
        
        # PASS 4 - Cluster Expansion
        state = cluster_expansion_node(state)
        
        # PASS 5 - Context Compression
        state = context_assembly_node(state, max_tokens=max_tokens)
        
        # FINAL - LLM Reasoning
        state = llm_reasoning_node(state)
        
        if state.errors:
            return None, f"Error: {', '.join(state.errors)}", state  # ← Return state
        
        return state.response, None, state  # ← Return state
        
    except Exception as e:
        import traceback
        return None, f"Pipeline failed: {str(e)}\n{traceback.format_exc()}", None





# =============================================================================
# SIDEBAR - INGESTION INTERFACE
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
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest using EnhancedBookIngestorPaddle
                success, result = ingest_book_enhanced(temp_path, book_title, author or "Unknown")
                
                # Cleanup
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                if success:
                    st.success(f"✅ Successfully ingested '{book_title}'!")
                    
                    # Display ingestion stats
                    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                    st.markdown("**📊 Ingestion Statistics:**")
                    st.markdown(f"- **Title:** {result.get('title', 'N/A')}")
                    st.markdown(f"- **Author:** {result.get('author', 'N/A')}")
                    st.markdown(f"- **Total Pages:** {result.get('total_pages', 'N/A')}")
                    st.markdown(f"- **Chapters Detected:** {result.get('total_chapters', 'N/A')}")
                    st.markdown(f"- **Total Chunks:** {result.get('total_chunks', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.ingestion_complete = True
                    st.session_state.last_ingestion_stats = result
                else:
                    st.error(f"❌ Ingestion failed: {result}")
            else:
                st.warning("Please provide both PDF and book title")
    
    st.divider()
    
    # Show last ingestion stats if available
    if st.session_state.last_ingestion_stats:
        with st.expander("📊 Last Ingestion Stats", expanded=False):
            stats = st.session_state.last_ingestion_stats
            st.json({
                "title": stats.get("title"),
                "author": stats.get("author"),
                "total_pages": stats.get("total_pages"),
                "total_chapters": stats.get("total_chapters"),
                "total_chunks": stats.get("total_chunks")
            })
    
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
    with st.expander("⚙️ 5-Pass Retrieval Settings"):
        st.markdown("**Pass 1:** Broad Vector Search")
        pass1_k = st.slider("Initial candidates", 30, 100, 50)
        
        st.markdown("**Pass 2:** Cross-Encoder Reranking")
        pass2_k = st.slider("After reranking", 10, 30, 15)
        
        st.markdown("**Pass 3:** Query Expansion")
        pass3_enabled = st.checkbox("Enable multi-hop", value=True)
        
        st.markdown("**Pass 5:** Compression")
        max_tokens = st.slider("Max context tokens", 1500, 4000, 2500)
        
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
st.markdown("**Enhanced Pipeline** • Ask questions about your ingested books and get AI-powered answers with citations!")

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


# Query button
if st.button("🔍 Search", type="primary", disabled=not query):
    if query:
        with st.spinner("🔎 Running 5-pass retrieval pipeline..."):
            matches = query_pinecone(
                query,
                top_k=pass1_k,
                book_filter=book_filter if book_filter != "All Books" else None,
                chapter_filter=chapter_filter if chapter_filter else None
            )
        
        # Add to history
        st.session_state.query_history.insert(0, {
            'query': query,
            'results': len(matches),
            'book_filter': book_filter
        })
        
        if not matches:
            st.warning("🤷 No relevant results found. Try rephrasing your question or removing filters.")
        else:
            st.success(f"✅ Found {len(matches)} relevant chunks")
            
            # Generate answer with user settings
            with st.spinner("🤖 Generating answer..."):
                response, error, final_state = generate_answer(
                    query, 
                    matches,
                    pass2_k=pass2_k,
                    pass3_enabled=pass3_enabled,
                    max_tokens=max_tokens
                )
            
            # ==================== FINAL CHUNKS EXPANDER ====================
            with st.expander("🔍 Final Chunks (After 5-Pass Pipeline)", expanded=False):
                if final_state and final_state.reranked_chunks:
                    st.markdown(f"**Showing final {len(final_state.reranked_chunks)} chunks after:**")
                    st.markdown("- ✅ Cross-encoder reranking")
                    st.markdown("- ✅ Multi-hop expansion")
                    st.markdown("- ✅ Cluster expansion")
                    st.markdown("- ✅ Deduplication & compression")
                    st.markdown("---")
                    
                    for i, retrieved_chunk in enumerate(final_state.reranked_chunks[:5], 1):  # Show top 5
                        chunk = retrieved_chunk.chunk
                        
                        st.markdown(f"### 🔹 Chunk {i}")
                        
                        # Show relevance scores
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Similarity", f"{retrieved_chunk.similarity_score:.2%}")
                        with col2:
                            st.metric("Rerank Score", f"{retrieved_chunk.rerank_score:.2%}")
                        with col3:
                            st.metric("Final Relevance", f"{retrieved_chunk.relevance_percentage:.1f}%")
                        
                        # Show metadata
                        st.markdown("**Metadata:**")
                        metadata_dict = {
                            "chunk_id": chunk.chunk_id,
                            "chapter": chunk.chapter,
                            "section": chunk.section if chunk.section else "N/A",
                            "page": chunk.page_number if chunk.page_number else "N/A",
                            "type": chunk.chunk_type,
                        }
                        st.json(metadata_dict)
                        
                        # Show content
                        st.markdown("**Content:**")
                        content_preview = chunk.content[:800] + ("..." if len(chunk.content) > 800 else "")
                        st.text_area(
                            f"Chunk {i} content",
                            content_preview,
                            height=200,
                            key=f"final_chunk_{i}",
                            label_visibility="collapsed"
                        )
                        
                        if i < len(final_state.reranked_chunks[:5]):  # Don't add divider after last chunk
                            st.divider()
                else:
                    st.warning("No final chunks available")

            if final_state and final_state.reranked_chunks:
                st.divider()
                
                st.markdown("### 📊 Pipeline Statistics")
                
                # Pipeline Statistics in 4 columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Pass 1 → Pass 2",
                        f"{len(matches)} → {pass2_k}",
                        delta=f"-{len(matches) - pass2_k}",
                        delta_color="normal"
                    )
                
                with col2:
                    st.metric(
                        "After Multi-Hop",
                        f"{len(final_state.reranked_chunks)}",
                        delta=f"+{len(final_state.reranked_chunks) - pass2_k}",
                        delta_color="normal"
                    )
                
                with col3:
                    # Get token count from assembled context
                    token_count = len(final_state.assembled_context.split()) if final_state.assembled_context else 0
                    st.metric(
                        "Final Tokens",
                        f"{token_count}",
                        delta=f"Max: {max_tokens}",
                        delta_color="off"
                    )
                
                with col4:
                    # Average relevance of final chunks
                    avg_relevance = sum(c.relevance_percentage for c in final_state.reranked_chunks) / len(final_state.reranked_chunks)
                    st.metric(
                        "Avg Relevance",
                        f"{avg_relevance:.1f}%",
                        delta_color="off"
                    )
            
            # ==================== ANSWER DISPLAY SECTION ====================
            st.divider()
            st.subheader("💬 AI-Generated Answer")
            
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
            else:
                st.warning("⚠️ No response generated")



# Query History
if st.session_state.query_history:
    st.divider()
    st.subheader("🕐 Query History")
    
    for i, item in enumerate(st.session_state.query_history[:5]):
        st.markdown(f"{i+1}. **{item['query']}** - {item['results']} results ({item['book_filter']})")

# Footer
st.divider()
st.caption("Built with Streamlit • Powered by Pinecone & OpenAI • Using Semantic Chunking with Sentence Transformers")