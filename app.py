"""
Streamlit App for RAG-Based Book Bot
Using book_ingestion.py for PDF processing
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

# ✅ CORRECT IMPORTS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rag_based_book_bot.document_ingestion.book_ingestion import BookIngestionService

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'ingestion_complete' not in st.session_state:
    st.session_state.ingestion_complete = False


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
        # Convert to list format to match storage
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
            # Initialize the service
            service = BookIngestionService()
            
            # Ingest the book (book_ingestion.py handles everything)
            metadata = service.ingest_book(pdf_path)
            
            return True, {
                "title": metadata.title,
                "author": metadata.author,
                "source": metadata.source_file
            }
    except Exception as e:
        return False, str(e)


# =============================================================================
# SIDEBAR - INGESTION INTERFACE
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

# Query button
if st.button("🔍 Search", type="primary", disabled=not query):
    if query:
        with st.spinner("🔎 Searching knowledge base..."):
            matches = query_pinecone(
                query,
                top_k=top_k,
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
            
            # Display results
            st.divider()
            st.subheader("📋 Retrieved Sources")
            
            for i, match in enumerate(matches):
                metadata = match.get("metadata", {})
                score = match.get("score", 0.0)
                
                # Extract metadata (book_ingestion.py format)
                book_title = metadata.get("book_title", "Unknown")
                
                # Handle lists for chapters/sections
                chapter_titles = metadata.get("chapter_titles", [])
                chapter_numbers = metadata.get("chapter_numbers", [])
                section_titles = metadata.get("section_titles", [])
                
                chapter_title = chapter_titles[0] if chapter_titles else "Unknown"
                chapter_number = chapter_numbers[0] if chapter_numbers else "N/A"
                
                page_start = metadata.get("page_start", "?")
                page_end = metadata.get("page_end", "?")
                contains_code = metadata.get("contains_code", False)
                
                # Get content - book_ingestion.py doesn't store text in metadata
                # So we'll show a preview message
                content_preview = f"[Chunk {i+1} from {book_title}]"
                
                # Display source
                with st.expander(f"**Source {i+1}** | {book_title} - Ch.{chapter_number} | Relevance: {score:.2%}", expanded=(i == 0)):
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
                    
                    st.info("💡 Note: Text content is stored as embeddings. To see full text, enable text storage in book_ingestion.py")
                    
                    # Show metadata details
                    st.json({
                        "book": book_title,
                        "chapter": f"{chapter_number}: {chapter_title}",
                        "pages": f"{page_start}-{page_end}",
                        "type": "code" if contains_code else "text",
                        "relevance": f"{score:.2%}"
                    })
            
            # Generate answer section
            st.divider()
            st.subheader("💬 Generated Answer")
            
            st.info("""
            **🔧 To Enable Answer Generation:**
            
            1. The retrieved chunks above show relevant sections
            2. To get AI-generated answers, integrate nodes.py pipeline:
               - Use `llm_reasoning_node()` from agents/nodes.py
               - Or add LLM call directly in this section
            
            3. Example integration:
```python
            from rag_based_book_bot.agents.nodes import llm_reasoning_node
            from rag_based_book_bot.agents.states import AgentState
            
            state = AgentState(user_query=query, reranked_chunks=matches)
            state = llm_reasoning_node(state)
            answer = state.response.answer
```
            """)

# Query History
if st.session_state.query_history:
    st.divider()
    st.subheader("🕐 Query History")
    
    for i, item in enumerate(st.session_state.query_history[:5]):
        st.markdown(f"{i+1}. **{item['query']}** - {item['results']} results ({item['book_filter']})")

# Footer
st.divider()
st.caption("Built with Streamlit • Powered by Pinecone & Sentence Transformers • Using book_ingestion.py")