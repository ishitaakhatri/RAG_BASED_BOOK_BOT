# test_pinecone_now.py
from dotenv import load_dotenv
from pinecone import Pinecone
import os


load_dotenv()

try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("coding-books")
    stats = index.describe_index_stats()
    
    print("✅ Pinecone connection SUCCESS!")
    print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"   Namespaces: {list(stats.get('namespaces', {}).keys())}")
except Exception as e:
    print(f"❌ Pinecone connection FAILED: {e}")
