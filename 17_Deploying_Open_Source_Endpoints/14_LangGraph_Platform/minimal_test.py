#!/usr/bin/env python3
"""Minimal Together AI test with verbose output."""

import os
from langchain_together import ChatTogether

print("🚀 TESTING TOGETHER AI ENDPOINT")
print("=" * 50)

# Set environment variables directly
api_key = "tgp_v1_8QKZsZVy9pnrlwweGajLJt05U8Y8_CtvKGH6ABE05IE"
model_name = "germeeai_f92a/openai/gpt-oss-20b-d37a5870"

os.environ["TOGETHER_API_KEY"] = api_key

print(f"📡 API Key: {api_key[:20]}...{api_key[-10:]}")
print(f"🤖 Model: {model_name}")
print(f"🌐 Endpoint: Together AI")
print()

print("🔄 Initializing ChatTogether client...")
# Create the model with verbose settings
llm = ChatTogether(
    model=model_name,
    temperature=0,
    max_tokens=50,
    timeout=30
)
print("✅ Client initialized successfully")
print()

print("📤 Sending request to Together AI...")
print("💬 Question: 'What is 2+2? Answer in one sentence.'")
print()

# Test it
try:
    response = llm.invoke("What is 2+2? Answer in one sentence.")
    
    print("📥 RESPONSE RECEIVED:")
    print(f"📄 Content: {response.content}")
    print(f"🔧 Model: {response.response_metadata.get('model_name', 'N/A')}")
    print(f"🎯 Tokens: {response.response_metadata.get('token_usage', {})}")
    print(f"✨ Finish Reason: {response.response_metadata.get('finish_reason', 'N/A')}")
    print()
    print("✅ SUCCESS! Together AI endpoint is working!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("💡 Check your API key and endpoint configuration")

print()
print("🏁 Test completed")