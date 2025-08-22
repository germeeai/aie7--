#!/usr/bin/env python3
"""Test script to verify Together AI integration with the LangGraph platform."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../.env")

from app.models import get_chat_model
from app.rag import retrieve_information

def test_chat_model():
    """Test the chat model integration."""
    print("Testing chat model...")
    print(f"TOGETHER_API_KEY set: {bool(os.environ.get('TOGETHER_API_KEY'))}")
    print(f"TOGETHER_MODEL: {os.environ.get('TOGETHER_MODEL')}")
    
    model = get_chat_model()
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    # Test a simple chat
    try:
        result = model.invoke("Hello! How are you?")
        print(f"Chat test successful: {result.content[:100]}...")
    except Exception as e:
        print(f"Chat test failed: {e}")

def test_embeddings():
    """Test embeddings integration."""
    print("\nTesting embeddings in RAG...")
    print(f"TOGETHER_EMBEDDING_MODEL: {os.environ.get('TOGETHER_EMBEDDING_MODEL')}")
    
    try:
        result = retrieve_information("What is a Pell Grant?")
        print(f"RAG test successful: {result[:100]}...")
    except Exception as e:
        print(f"RAG test failed: {e}")

if __name__ == "__main__":
    print("Testing Together AI integration...")
    print(f"Current working directory: {os.getcwd()}")
    
    test_chat_model()
    test_embeddings()
    print("\nIntegration test complete!")
    
    print("\nYour configuration:")
    print(f"TOGETHER_API_KEY: {os.environ.get('TOGETHER_API_KEY', 'Not set')}")
    print(f"TOGETHER_MODEL: {os.environ.get('TOGETHER_MODEL', 'Not set')}")
    print(f"TOGETHER_EMBEDDING_MODEL: {os.environ.get('TOGETHER_EMBEDDING_MODEL', 'Not set')}")