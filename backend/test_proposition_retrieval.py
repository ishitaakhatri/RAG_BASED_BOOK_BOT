# Create a test script: backend/test_proposition_retrieval.py
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "coding-books"))

# Fetch a few random vectors
stats = index.describe_index_stats()
print("Total vectors:", stats['total_vector_count'])

# Query with dummy vector to see metadata
results = index.query(
    vector=[0.1] * 384,
    top_k=3,
    namespace="books_rag",
    include_metadata=True
)

# Inspect the metadata
for i, match in enumerate(results['matches']):
    print(f"\n--- Match {i+1} ---")
    print(f"ID: {match['id']}")
    meta = match['metadata']
    print(f"Heading: {meta.get('heading', 'N/A')}")
    print(f"Full text preview: {meta.get('full_text', 'N/A')[:100]}...")
    print(f"Hierarchy: {meta.get('hierarchy_path', 'N/A')}")
