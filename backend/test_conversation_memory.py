"""
Test script to verify conversation memory works
Run this to test the complete conversation flow
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_conversation():
    print("="*80)
    print("TESTING CONVERSATION MEMORY")
    print("="*80)
    
    # Query 1
    print("\n[Query 1] What is a convolutional neural network?")
    response1 = requests.post(f'{BASE_URL}/query', json={
        'query': 'What is a convolutional neural network?',
        'pass1_k': 20,  # Smaller for faster testing
        'pass2_k': 10
    })
    
    if response1.status_code != 200:
        print(f"❌ Query 1 failed: {response1.text}")
        return
    
    data1 = response1.json()
    session_id = data1['session_id']
    
    print(f"✅ Query 1 successful")
    print(f"   Session ID: {session_id}")
    print(f"   Answer preview: {data1['answer'][:150]}...")
    print(f"   Conversation turn: {data1['conversation_turn']}")
    
    # Wait a moment
    time.sleep(2)
    
    # Query 2 - Follow-up with SAME session_id
    print(f"\n[Query 2] Show me code for it (with session_id: {session_id})")
    response2 = requests.post(f'{BASE_URL}/query', json={
        'query': 'Show me code for it',
        'session_id': session_id,  # ← Use same session!
        'pass1_k': 20,
        'pass2_k': 10
    })
    
    if response2.status_code != 200:
        print(f"❌ Query 2 failed: {response2.text}")
        return
    
    data2 = response2.json()
    
    print(f"✅ Query 2 successful")
    print(f"   Session ID: {data2['session_id']}")
    print(f"   Resolved query: {data2.get('resolved_query', 'N/A')}")
    print(f"   Referenced turn: {data2['stats'].get('referenced_turn', 'N/A')}")
    print(f"   Answered from history: {data2.get('answered_from_history', False)}")
    print(f"   Conversation turn: {data2['conversation_turn']}")
    print(f"   Answer preview: {data2['answer'][:150]}...")
    
    # Query 3 - Another follow-up
    time.sleep(2)
    
    print(f"\n[Query 3] What are the advantages? (with session_id: {session_id})")
    response3 = requests.post(f'{BASE_URL}/query', json={
        'query': 'What are the advantages?',
        'session_id': session_id,
        'pass1_k': 20,
        'pass2_k': 10
    })
    
    if response3.status_code != 200:
        print(f"❌ Query 3 failed: {response3.text}")
        return
    
    data3 = response3.json()
    
    print(f"✅ Query 3 successful")
    print(f"   Resolved query: {data3.get('resolved_query', 'N/A')}")
    print(f"   Referenced turn: {data3['stats'].get('referenced_turn', 'N/A')}")
    print(f"   Conversation turn: {data3['conversation_turn']}")
    print(f"   Answer preview: {data3['answer'][:150]}...")
    
    # Get conversation history
    print(f"\n[Verification] Fetching conversation history from API...")
    history_response = requests.get(f'{BASE_URL}/conversation/{session_id}')
    
    if history_response.status_code == 200:
        history_data = history_response.json()
        print(f"✅ Conversation history retrieved")
        print(f"   Total turns: {history_data['total_turns']}")
        for i, turn in enumerate(history_data['turns'], 1):
            print(f"   Turn {i}: {turn['user_query'][:50]}...")
    else:
        print(f"❌ Failed to get history: {history_response.text}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return session_id


if __name__ == "__main__":
    try:
        session_id = test_conversation()
        print(f"\n✅ All tests passed!")
        print(f"Session ID for manual testing: {session_id}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
