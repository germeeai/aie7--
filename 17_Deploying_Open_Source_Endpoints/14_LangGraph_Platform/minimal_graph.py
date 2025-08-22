#!/usr/bin/env python3
"""Minimal LangGraph app with Together AI."""

import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_together import ChatTogether
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Set environment
os.environ["TOGETHER_API_KEY"] = "tgp_v1_8QKZsZVy9pnrlwweGajLJt05U8Y8_CtvKGH6ABE05IE"

class State(TypedDict):
    messages: List[BaseMessage]

def chat(state: State) -> Dict[str, Any]:
    llm = ChatTogether(model="germeeai_f92a/openai/gpt-oss-20b-d37a5870", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(State)
graph.add_node("chat", chat)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
compiled_graph = graph.compile()

# Export for LangGraph
graph = compiled_graph