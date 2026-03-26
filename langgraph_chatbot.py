import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

# state schema

class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]

# helpers

def to_langchain_messages(
    messages: List[Dict[str, str]]
) -> List[BaseMessage]:
    """Convert dict-based messages to LangChain message objects."""
    converted: List[BaseMessage] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role}")

    return converted

# llm

# initiate your open ai key here 

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
)

# nodes

def input_node(state: ChatState) -> ChatState:
    return state


def llm_node(state: ChatState) -> ChatState:
    langchain_messages = to_langchain_messages(state["messages"])

    response = llm.invoke(langchain_messages)

    return {
        "messages": state["messages"] + [
            {"role": "assistant", "content": response.content}
        ],
        "metadata": state["metadata"],
    }


def memory_update_node(state: ChatState) -> ChatState:
    return state


def output_node(state: ChatState) -> ChatState:
    
    return state

# graph definition

graph = StateGraph(ChatState)

graph.add_node("input", input_node)
graph.add_node("llm", llm_node)
graph.add_node("memory_update", memory_update_node)
graph.add_node("output", output_node)

graph.set_entry_point("input")

graph.add_edge("input", "llm")
graph.add_edge("llm", "memory_update")
graph.add_edge("memory_update", "output")
graph.add_edge("output", END)

chat_workflow = graph.compile()

# demo

if __name__ == "__main__":
    state: ChatState = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, my name is Shiv."},
        ],
        "metadata": {},
    }

    result = chat_workflow.invoke(state)
    print(result["messages"][-1]["content"])

    
    state = {
        "messages": result["messages"] + [
            {"role": "user", "content": "What is my name?"}
        ],
        "metadata": {},
    }

    result = chat_workflow.invoke(state)
    print(result["messages"][-1]["content"])
