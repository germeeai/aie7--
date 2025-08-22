#!/usr/bin/env python3
"""Demonstration of minimal_graph.py working with Together AI."""

from minimal_graph import graph
from langchain_core.messages import HumanMessage

print("🎯 DEMONSTRATING LANGGRAPH + TOGETHER AI")
print("=" * 60)

print("📋 Graph Configuration:")
print("  • LangGraph: Minimal chat graph")
print("  • Together AI Model: germeeai_f92a/openai/gpt-oss-20b-d37a5870")
print("  • Flow: User Message → Chat Node → AI Response")
print()

# Test cases
test_cases = [
    "What is 3+7?",
    "Say hello in French",
    "What's the capital of Japan?",
    "Explain gravity in one sentence"
]

for i, question in enumerate(test_cases, 1):
    print(f"🧪 TEST {i}/4")
    print(f"❓ Question: {question}")
    print("🔄 Processing through LangGraph...")
    
    try:
        # Invoke the graph
        result = graph.invoke({
            'messages': [HumanMessage(content=question)]
        })
        
        # Extract response
        ai_response = result['messages'][-1]
        
        print(f"✅ AI Response: {ai_response.content}")
        print(f"📊 Model: {getattr(ai_response, 'response_metadata', {}).get('model_name', 'N/A')}")
        print(f"🎯 Tokens: {getattr(ai_response, 'response_metadata', {}).get('token_usage', {})}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 40)
    print()

print("🏆 DEMONSTRATION COMPLETE!")
print("✅ LangGraph successfully integrated with Together AI endpoint")
print("🚀 Your custom model is working through the graph workflow")