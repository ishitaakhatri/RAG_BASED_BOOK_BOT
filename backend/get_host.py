# ============================================================================
# STEP 1: Get the correct Pinecone host
# ============================================================================

import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Get all indexes
indexes = pc.list_indexes()

print("\n" + "="*70)
print("YOUR PINECONE INDEXES")
print("="*70)

for index_info in indexes:
    print(f"\nIndex Name: {index_info.name}")
    print(f"Host: {index_info.host}")
    print(f"Dimension: {index_info.dimension}")
    print(f"Metric: {index_info.metric}")
    print(f"Status: {index_info.status}")
    print(f"Region: {index_info.spec.serverless.cloud if hasattr(index_info.spec, 'serverless') else 'N/A'}")

print("\n" + "="*70)

# Find your 'coding-books' index
target_index = None
for idx in indexes:
    if idx.name == PINECONE_INDEX:
        target_index = idx
        break

if target_index:
    print(f"\n✅ Found index: {PINECONE_INDEX}")
    print(f"✅ HOST TO USE: {target_index.host}")
    print(f"\nAdd this to your .env file:")
    print(f"PINECONE_INDEX_HOST={target_index.host}")
else:
    print(f"\n❌ Index '{PINECONE_INDEX}' not found!")
    print(f"Available indexes: {[idx.name for idx in indexes]}")

print("="*70 + "\n")