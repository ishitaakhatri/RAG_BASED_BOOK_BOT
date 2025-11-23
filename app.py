"""
Streamlit App for RAG-Based Book Bot
Test your ingestion and query system interactively
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

# Now the debug messages
st.write("✅ Step 1: Basic imports successful")

load_dotenv()
st.write("✅ Step 2: .env loaded")

# Check env variables
if os.getenv("PINECONE_API_KEY"):
    st.write("✅ Step 3: Pinecone API key found")
else:
    st.error("❌ Step 3: Pinecone API key NOT found")
    st.stop()

st.write("✅ Step 4: Starting Pinecone import...")
from pinecone import Pinecone
st.write("✅ Step 5: Pinecone imported")

st.write("✅ Step 6: Starting sentence-transformers import...")
from sentence_transformers import SentenceTransformer
st.write("✅ Step 7: sentence-transformers imported")

st.write("✅ Step 8: Starting enhanced_ingestion import...")
from rag_based_book_bot.document_ingestion.enhanced_ingestion import EnhancedBookIngestor
st.write("✅ Step 9: ALL IMPORTS SUCCESSFUL!")

# Rest of your app code...


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
    .code-snippet {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
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
        
        # Check if index exists
        try:
            index = pc.Index(index_name)
            return pc, index
        except Exception as e:
            st.error(f"❌ Pinecone index '{index_name}' not found")
            st.info("""
            **To create the index, run:**
            ```
            from pinecone import Pinecone, ServerlessSpec
            
            pc = Pinecone(api_key="your-api-key")
            pc.create_index(
                name="coding-books",
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            ```
            """)
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
        filter_dict["chapter_number"] = chapter_filter
    
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
    """Ingest a book into Pinecone"""
    try:
        with st.spinner(f"🔄 Ingesting '{book_title}'... This may take several minutes."):
            ingestor = EnhancedBookIngestor(debug=False)
            metadata = ingestor.ingest_book(
                pdf_path=pdf_path,
                book_title=book_title,
                author=author
            )
            return True, metadata
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
                
                # Ingest
                success, result = ingest_book(temp_path, book_title, author or "Unknown")
                
                # Cleanup
                os.remove(temp_path)
                
                if success:
                    st.success(f"✅ Successfully ingested '{book_title}'!")
                    st.info(f"📖 Chapters: {result.total_chapters}\n📄 Pages: {result.total_pages}")
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
                
                # Extract metadata
                book_title = metadata.get("book_title", "Unknown")
                chapter_title = metadata.get("chapter_title", "Unknown")
                chapter_number = metadata.get("chapter_number", "N/A")
                page_start = metadata.get("page_start", "?")
                page_end = metadata.get("page_end", "?")
                contains_code = metadata.get("contains_code", False)
                content = metadata.get("text", "No content available")
                
                # Build hierarchy
                sections = metadata.get("section_titles", [])
                section_str = " → ".join(sections) if sections else ""
                
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
                    
                    if section_str:
                        st.caption(f"Section: {section_str}")
                    
                    st.divider()
                    
                    # Content
                    if contains_code:
                        st.code(content, language="python")
                    else:
                        st.markdown(content)
                    
                    # Copy button
                    st.download_button(
                        label="📋 Copy Content",
                        data=content,
                        file_name=f"source_{i+1}.txt",
                        mime="text/plain",
                        key=f"download_{i}"
                    )
            
            # Generate answer section (placeholder for LLM integration)
            st.divider()
            st.subheader("💬 Generated Answer")
            
            st.info("""
            **🔧 LLM Integration Required**
            
            To generate answers, integrate your LLM (Azure OpenAI, OpenAI, etc.) in the code.
            
            The retrieved sources above can be passed to your LLM as context to generate comprehensive answers.
            """)
            
            # Show assembled context
            with st.expander("🔍 View Assembled Context (for LLM)"):
                context = ""
                for i, match in enumerate(matches):
                    metadata = match.get("metadata", {})
                    content = metadata.get("text", "")
                    book = metadata.get("book_title", "Unknown")
                    chapter = metadata.get("chapter_title", "Unknown")
                    
                    context += f"[SOURCE {i+1}] {book} - {chapter}\n{content}\n\n---\n\n"
                
                st.text_area("Context for LLM", context, height=300)

# Query History
if st.session_state.query_history:
    st.divider()
    st.subheader("🕐 Query History")
    
    for i, item in enumerate(st.session_state.query_history[:5]):
        st.markdown(f"{i+1}. **{item['query']}** - {item['results']} results ({item['book_filter']})")

# Footer
st.divider()
st.caption("Built with Streamlit • Powered by Pinecone & Sentence Transformers")
