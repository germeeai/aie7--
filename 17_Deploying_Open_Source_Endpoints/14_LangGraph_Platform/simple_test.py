#!/usr/bin/env python3
"""Simple test using LangGraph SDK."""

from langgraph_sdk import get_client

def test_simple_app():
    """Test the simple LangGraph application."""
    
    print("🧪 TESTING SIMPLE LANGGRAPH APP")
    print("=" * 50)
    
    try:
        # Connect to local server
        client = get_client(url="http://127.0.0.1:2024")
        
        # List available assistants
        assistants = client.assistants.search()
        print(f"✓ Available assistants: {[a['assistant_id'] for a in assistants]}")
        
        # Create a thread
        thread = client.threads.create()
        print(f"✓ Thread created: {thread['thread_id']}")
        
        # Send a message using the simple assistant
        response = client.runs.create(
            thread_id=thread['thread_id'],
            assistant_id="simple",
            input={"messages": [{"role": "user", "content": "Hello! What is 7+3?"}]}
        )
        
        # Wait for completion
        result = client.runs.join(thread_id=thread['thread_id'], run_id=response['run_id'])
        
        print(f"✓ Run completed: {result['status']}")
        
        # Get the response
        if 'messages' in result and result['messages']:
            last_message = result['messages'][-1]
            print(f"✓ Assistant response: {last_message['content']}")
        else:
            print("⚠️  No response message found")
            
        print("\n✅ SIMPLE APP TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_app()