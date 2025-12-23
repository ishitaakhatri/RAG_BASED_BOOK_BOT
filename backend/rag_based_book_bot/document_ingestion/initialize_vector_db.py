"""
Initialize Pinecone Vector Database for RAG Book Bot
Run this once to set up your Pinecone index with proper configuration
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
else:
    print(f"⚠️  Warning: .env file not found at {env_path}")
    print("   Using environment variables or defaults")

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("❌ Error: pinecone-client not installed")
    print("   Run: pip install pinecone-client")
    sys.exit(1)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books-2")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
DIMENSION = 1024  # Qwen/Qwen3-Embedding-0.6B
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def validate_config():
    """Validate configuration before proceeding"""
    print("\n" + "=" * 70)
    print("📋 Configuration Validation")
    print("=" * 70)
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found")
        print("\n💡 To fix this:")
        print("   1. Create a .env file in backend/ directory")
        print("   2. Add: PINECONE_API_KEY=your-api-key-here")
        print("   3. Get your API key from: https://app.pinecone.io/")
        return False
    
    print(f"✅ API Key: {'*' * 20}{PINECONE_API_KEY[-4:]}")
    print(f"✅ Index Name: {INDEX_NAME}")
    print(f"✅ Namespace: {NAMESPACE}")
    print(f"✅ Dimension: {DIMENSION}")
    print(f"✅ Metric: {METRIC}")
    print(f"✅ Cloud: {CLOUD}")
    print(f"✅ Region: {REGION}")
    
    return True


def init_pinecone():
    """Initialize Pinecone index with proper configuration"""
    
    # Validate configuration
    if not validate_config():
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🚀 Initializing Pinecone")
    print("=" * 70)
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Connected to Pinecone")
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        print(f"\n📊 Existing indexes: {existing_indexes if existing_indexes else 'None'}")
        
        if INDEX_NAME in existing_indexes:
            print(f"\n⚠️  Index '{INDEX_NAME}' already exists")
            choice = input("   Delete and recreate? (yes/no): ").lower().strip()
            
            if choice == 'yes':
                print(f"🗑️  Deleting existing index '{INDEX_NAME}'...")
                pc.delete_index(INDEX_NAME)
                print("✅ Deleted")
                
                # Wait for deletion to complete
                import time
                print("⏳ Waiting for deletion to complete...")
                time.sleep(5)
            else:
                print("✅ Keeping existing index")
                index = pc.Index(INDEX_NAME)
                print_index_stats(index)
                return index
        
        # Create new index
        print(f"\n🔨 Creating index '{INDEX_NAME}'...")
        print(f"   Dimension: {DIMENSION}")
        print(f"   Metric: {METRIC}")
        print(f"   Cloud: {CLOUD} ({REGION})")
        
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION
            )
        )
        
        print("✅ Index created successfully!")
        
        # Wait for index to be ready
        import time
        print("⏳ Waiting for index to be ready...")
        time.sleep(10)
        
        # Get index
        index = pc.Index(INDEX_NAME)
        
        # Display statistics
        print_index_stats(index)
        
        # Initialize namespaces by upserting dummy vectors
        print("\n📦 Initializing namespaces...")
        initialize_namespaces(index)
        
        return index
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your PINECONE_API_KEY is correct")
        print("   2. Ensure you have an active Pinecone account")
        print("   3. Verify your account has available indexes")
        print("   4. Check: https://app.pinecone.io/")
        sys.exit(1)


def initialize_namespaces(index):
    """Initialize the required namespaces with dummy vectors"""
    import uuid
    
    try:
        # Initialize main namespace (books_rag)
        print(f"   Creating namespace: {NAMESPACE}")
        dummy_vector = {
            "id": f"init_{uuid.uuid4().hex[:8]}",
            "values": [0.0] * DIMENSION,
            "metadata": {
                "text": "Initialization vector",
                "book_title": "__init__",
                "author": "System",
                "page_start": 0
            }
        }
        index.upsert(vectors=[dummy_vector], namespace=NAMESPACE)
        print(f"   ✅ Namespace '{NAMESPACE}' initialized")
        
        # Initialize metadata namespace (books_metadata)
        metadata_namespace = "books_metadata"
        print(f"   Creating namespace: {metadata_namespace}")
        dummy_metadata = {
            "id": f"init_{uuid.uuid4().hex[:8]}",
            "values": [1.0] * DIMENSION,
            "metadata": {
                "book_title": "__init__",
                "author": "System",
                "total_chunks": 0,
                "indexed_at": 0
            }
        }
        index.upsert(vectors=[dummy_metadata], namespace=metadata_namespace)
        print(f"   ✅ Namespace '{metadata_namespace}' initialized")
        
        print("✅ All namespaces initialized")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize namespaces: {e}")
        print("   This is OK - namespaces will be created on first upsert")


def print_index_stats(index):
    """Print index statistics"""
    try:
        stats = index.describe_index_stats()
        
        print("\n" + "=" * 70)
        print("📊 Index Statistics")
        print("=" * 70)
        
        total_vectors = stats.get('total_vector_count', 0)
        namespaces = stats.get('namespaces', {})
        
        print(f"Total vectors: {total_vectors}")
        
        if namespaces:
            print("\nNamespaces:")
            for ns_name, ns_stats in namespaces.items():
                count = ns_stats.get('vector_count', 0)
                print(f"  • {ns_name}: {count} vectors")
        else:
            print("\nNamespaces: None (will be created on first upsert)")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"⚠️  Could not retrieve stats: {e}")


def verify_connection():
    """Verify connection works"""
    print("\n" + "=" * 70)
    print("🔍 Verifying Connection")
    print("=" * 70)
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        # Try a simple query
        test_vector = [0.1] * DIMENSION
        results = index.query(
            vector=test_vector,
            top_k=1,
            namespace=NAMESPACE,
            include_metadata=True
        )
        
        print("✅ Connection verified - query successful")
        print(f"   Returned {len(results.get('matches', []))} matches")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection verification failed: {e}")
        return False


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("🌲 RAG Book Bot - Pinecone Initialization")
    print("=" * 70)
    
    # Initialize index
    index = init_pinecone()
    
    # Verify connection
    if verify_connection():
        print("\n" + "=" * 70)
        print("✅ Setup Complete!")
        print("=" * 70)
        print("\n🎉 Your Pinecone index is ready!")
        print("\n📝 Next steps:")
        print("   1. Run the backend: cd backend && python main.py")
        print("   2. Upload a book using the /ingest endpoint")
        print("   3. Start querying!")
        print("\n💡 Useful commands:")
        print(f"   • Check index: https://app.pinecone.io/indexes/{INDEX_NAME}")
        print("   • API docs: http://localhost:8000/docs")
        print("=" * 70 + "\n")
    else:
        print("\n⚠️  Setup completed but verification failed")
        print("   The index was created but there might be connection issues")
        print("   Try running your application anyway - it might work!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)