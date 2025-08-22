"""Model utilities for constructing chat LLM clients.

Centralizes configuration of the default chat model and temperature so graphs can
import a single helper without repeating provider-specific wiring.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

try:
    from langchain_together import ChatTogether
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    ChatTogether = None


def get_chat_model(model_name: str | None = None, *, temperature: float = 0) -> Any:
    """Return a configured LangChain chat model client.

    - model_name: optional override. If not provided, uses TOGETHER_MODEL env var,
      falling back to OPENAI_MODEL, then "germeeai_f92a/openai/gpt-oss-20b-d37a5870".
    - temperature: sampling temperature for the chat model.

    Returns: a LangChain-compatible chat model instance.
    """
    name = model_name or os.environ.get("TOGETHER_MODEL") or os.environ.get("OPENAI_MODEL", "germeeai_f92a/openai/gpt-oss-20b-d37a5870")
    
    # Use Together AI if available and TOGETHER_API_KEY is set or if model name suggests Together/custom endpoints
    if (TOGETHER_AVAILABLE and 
        (os.environ.get("TOGETHER_API_KEY") or 
         any(provider in name.lower() for provider in ["meta-llama", "mistral", "qwen", "germeeai", "baai"]))):
        return ChatTogether(model=name, temperature=temperature)
    else:
        return ChatOpenAI(model=name, temperature=temperature)


