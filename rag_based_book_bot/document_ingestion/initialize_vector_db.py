"""
Initialize Pinecone Vector Database
Run this once to set up your index
"""
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "coding-books"
DIMENSION = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

def init_pinecone():
    """Initialize Pinecone index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' already exists")
        choice = input("Delete and recreate? (yes/no): ")
        if choice.lower() == 'yes':
            pc.delete_index(INDEX_NAME)
            print(f"Deleted existing index '{INDEX_NAME}'")
        else:
            print("Keeping existing index")
            return pc.Index(INDEX_NAME)
    
    # Create new index
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )
    
    print(f"Index '{INDEX_NAME}' created successfully")
    return pc.Index(INDEX_NAME)

if __name__ == "__main__":
    index = init_pinecone()
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")