"""Simple model configuration for Together AI only."""

import os
from dotenv import load_dotenv
from langchain_together import ChatTogether

# Load environment variables
load_dotenv(".env")

def get_simple_chat_model():
    """Get Together AI chat model."""
    
    api_key = os.environ.get("TOGETHER_API_KEY")
    model_name = os.environ.get("TOGETHER_MODEL", "germeeai_f92a/openai/gpt-oss-20b-d37a5870")
    
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is required")
    
    print(f"✓ Using Together AI model: {model_name}")
    return ChatTogether(model=model_name, temperature=0)