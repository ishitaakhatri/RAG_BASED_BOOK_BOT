from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("pinecone_api_key")
pc = Pinecone(api_key=api_key)

# Create index
pc.create_index(
    name="coding-books",
    dimension=384,  # Matches sentence-transformers/all-MiniLM-L6-v2
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # Use free tier region
    )
)

print("✅ Index 'coding-books' created successfully!")
