"""Simple LangGraph agent using Together AI."""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from simple_models import get_simple_chat_model

class SimpleState(TypedDict):
    messages: List[BaseMessage]

def chat_node(state: SimpleState) -> Dict[str, Any]:
    """Simple chat node that responds to messages."""
    model = get_simple_chat_model()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def build_simple_graph():
    """Build a minimal chat graph."""
    graph = StateGraph(SimpleState)
    
    # Add single chat node
    graph.add_node("chat", chat_node)
    
    # Set entry point and end
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    
    return graph

# Create the compiled graph
graph = build_simple_graph().compile()