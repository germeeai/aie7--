#!/usr/bin/env python3
"""Test script for the simple LangGraph server."""

import requests
import json

def test_simple_server():
    """Test the simple LangGraph server API."""
    
    print("🧪 TESTING SIMPLE LANGGRAPH SERVER")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:2024"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/ok")
        print(f"✓ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test assistant creation
    try:
        create_payload = {
            "assistant_id": "simple",
            "graph_id": "simple_chat", 
            "config": {}
        }
        
        response = requests.post(
            f"{base_url}/assistants",
            json=create_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201]:
            assistant_data = response.json()
            assistant_id = assistant_data.get("assistant_id", "simple")
            print(f"✓ Assistant created/found: {assistant_id}")
        else:
            print(f"⚠️  Assistant creation response: {response.status_code}")
            assistant_id = "simple"  # Try with default
        
        # Test thread creation  
        thread_response = requests.post(f"{base_url}/threads")
        if thread_response.status_code == 200:
            thread_data = thread_response.json()
            thread_id = thread_data["thread_id"]
            print(f"✓ Thread created: {thread_id}")
        else:
            print(f"❌ Thread creation failed: {thread_response.status_code}")
            return
            
        # Test message sending
        message_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! What is 3+3?"
                }
            ]
        }
        
        run_response = requests.post(
            f"{base_url}/threads/{thread_id}/runs",
            json=message_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if run_response.status_code == 200:
            run_data = run_response.json()
            print(f"✓ Message sent successfully")
            
            # Get the response
            messages_response = requests.get(f"{base_url}/threads/{thread_id}/messages")
            if messages_response.status_code == 200:
                messages = messages_response.json()
                if messages:
                    last_message = messages[-1]
                    print(f"✓ Response received: {last_message.get('content', 'No content')}")
                else:
                    print("⚠️  No messages found")
            else:
                print(f"❌ Failed to get messages: {messages_response.status_code}")
        else:
            print(f"❌ Message sending failed: {run_response.status_code}")
            
        print("\n✅ SIMPLE SERVER TEST COMPLETED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_simple_server()