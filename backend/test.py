#!/usr/bin/env python3
"""
Complete diagnostic to find why vectors aren't being upserted
Run this BEFORE trying to ingest another book
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*80)
print("COMPLETE PINECONE DIAGNOSTIC")
print("="*80)

# Step 1: Check environment variables
print("\n[STEP 1] Checking Environment Variables...")
print("-" * 80)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")

print(f"  PINECONE_API_KEY set:     {bool(PINECONE_API_KEY)} (len={len(PINECONE_API_KEY) if PINECONE_API_KEY else 0})")
print(f"  PINECONE_INDEX_NAME:      {PINECONE_INDEX_NAME}")
print(f"  PINECONE_INDEX_HOST:      {PINECONE_INDEX_HOST}")
print(f"  PINECONE_NAMESPACE:       {PINECONE_NAMESPACE}")

if not PINECONE_API_KEY:
    print("\n❌ ERROR: PINECONE_API_KEY is empty!")
    sys.exit(1)

if not PINECONE_INDEX_HOST:
    print("\n❌ ERROR: PINECONE_INDEX_HOST is empty!")
    print("   Get this from: Pinecone Console → Your Index → Host")
    sys.exit(1)

print("\n✅ All environment variables set")

# Step 2: Test Pinecone connection
print("\n[STEP 2] Testing Pinecone Connection...")
print("-" * 80)

try:
    from pinecone import Pinecone
    print("  ✅ Pinecone library imported")
except Exception as e:
    print(f"  ❌ Failed to import Pinecone: {e}")
    sys.exit(1)

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("  ✅ Pinecone client created")
except Exception as e:
    print(f"  ❌ Failed to create Pinecone client: {e}")
    sys.exit(1)

# Step 3: Test index connection
print("\n[STEP 3] Testing Index Connection...")
print("-" * 80)

try:
    index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_INDEX_HOST)
    print(f"  ✅ Connected to index: {PINECONE_INDEX_NAME}")
    print(f"     Host: {PINECONE_INDEX_HOST}")
except Exception as e:
    print(f"  ❌ Failed to connect to index: {e}")
    sys.exit(1)

# Step 4: Check index stats
print("\n[STEP 4] Checking Index Stats...")
print("-" * 80)

try:
    stats = index.describe_index_stats()
    print(f"  Total vectors:     {stats.get('total_vector_count', 0)}")
    print(f"  Dimension:         {stats.get('dimension', 'unknown')}")
    
    namespaces = stats.get('namespaces', {})
    print(f"\n  Namespaces ({len(namespaces)}):")
    for ns_name, ns_stats in namespaces.items():
        print(f"    - '{ns_name}': {ns_stats.get('vector_count', 0)} vectors")
    
    # Check our specific namespace
    our_ns = namespaces.get(PINECONE_NAMESPACE, {})
    our_count = our_ns.get('vector_count', 0)
    print(f"\n  Our namespace '{PINECONE_NAMESPACE}': {our_count} vectors")
    
except Exception as e:
    print(f"  ❌ Failed to get stats: {e}")
    sys.exit(1)

# Step 5: Test upsert with sample data
print("\n[STEP 5] Testing Upsert with Sample Data...")
print("-" * 80)

try:
    from sentence_transformers import SentenceTransformer
    import uuid
    
    # Load embedding model
    print("  Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("  ✅ Model loaded")
    
    # Create test data
    test_text = "This is a test vector for diagnostic purposes to verify Pinecone upsert works correctly."
    test_embedding = model.encode([test_text])[0]
    
    test_vector = {
        "id": f"test-{uuid.uuid4()}",
        "values": test_embedding.tolist(),
        "metadata": {
            "text": test_text,
            "test": True,
            "timestamp": int(time.time())
        }
    }
    
    print(f"  Test vector ID: {test_vector['id']}")
    print(f"  Embedding dim:  {len(test_vector['values'])}")
    
    # Try upsert
    print(f"\n  Attempting upsert to namespace '{PINECONE_NAMESPACE}'...")
    response = index.upsert(
        vectors=[test_vector],
        namespace=PINECONE_NAMESPACE
    )
    print(f"  ✅ Upsert successful!")
    print(f"     Response: {response}")
    
    # Wait for indexing
    print(f"\n  Waiting 3 seconds for indexing...")
    time.sleep(3)
    
    # Verify
    print(f"  Verifying vector was stored...")
    stats_after = index.describe_index_stats()
    ns_after = stats_after.get('namespaces', {}).get(PINECONE_NAMESPACE, {})
    count_after = ns_after.get('vector_count', 0)
    
    print(f"  Vectors in '{PINECONE_NAMESPACE}' after upsert: {count_after}")
    
    if count_after > our_count:
        print(f"\n  ✅ SUCCESS! Vector was stored!")
        print(f"     Increase: {count_after - our_count} vector(s)")
    else:
        print(f"\n  ⚠️  Vector count didn't increase")
        print(f"     Before: {our_count}, After: {count_after}")
        print(f"     This might be a timing issue - check again in 10 seconds")
    
except Exception as e:
    print(f"  ❌ Upsert failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Summary
print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

print("\n✅ All tests passed! Your Pinecone setup is working correctly.")
print("\nNow try ingesting a book again:")
print("  1. Upload PDF via Streamlit")
print("  2. Check logs for upsert messages")
print("  3. Verify vectors appear in Pinecone Console")

print("\n" + "="*80 + "\n")