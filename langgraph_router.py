import os
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# state schema

class RouterState(TypedDict):
    input: str
    route: str
    output: str

# llm

# initiate open ai key here 
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
)

# Tool Node

def math_tool(expression: str) -> str:
    """
    Very simple and safe math evaluator.
    Supports +, -, *, / and integers.
    """
    if not re.fullmatch(r"[0-9+\-*/ ().]+", expression):
        return "Invalid math expression."

    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "Error while evaluating expression."

# nodes

def decision_node(state: RouterState) -> RouterState:
    text = state["input"]

    # Simple, explicit routing rule
    if re.search(r"\d", text):
        route = "math"
    else:
        route = "llm"

    return {
        "input": state["input"],
        "route": route,
        "output": "",
    }


def math_node(state: RouterState) -> RouterState:
    result = math_tool(state["input"])
    return {
        "input": state["input"],
        "route": state["route"],
        "output": result,
    }


def llm_node(state: RouterState) -> RouterState:
    response = llm.invoke(state["input"])
    return {
        "input": state["input"],
        "route": state["route"],
        "output": response.content,
    }

# graph definition

graph = StateGraph(RouterState)

graph.add_node("decision", decision_node)
graph.add_node("math", math_node)
graph.add_node("llm", llm_node)

graph.set_entry_point("decision")

# Conditional routing
graph.add_conditional_edges(
    "decision",
    lambda state: state["route"],
    {
        "math": "math",
        "llm": "llm",
    },
)

graph.add_edge("math", END)
graph.add_edge("llm", END)

router_workflow = graph.compile()

# demo
if __name__ == "__main__":
    tests = [
        "2 + 3 * 4",
        "What is the capital of France?",
        "10 / (2 + 3)",
        "Explain LangGraph in one sentence",
    ]

    for text in tests:
        result = router_workflow.invoke(
            {"input": text, "route": "", "output": ""}
        )
        print(f"\nInput: {text}")
        print(f"Output: {result['output']}")
