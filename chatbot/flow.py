from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from chatbot.utils import search_faiss
from chatbot.groq_llm import refine_explanation

class State(TypedDict):
    query: str
    retrieved: List[dict]
    response: str

def retrieve_node(state: State) -> dict:
    return {"retrieved": search_faiss(state["query"])}

def route(state: State) -> Literal["respond", "fallback"]:
    return "respond" if state["retrieved"] else "fallback"

def respond_node(state: State) -> dict:
    result = state["retrieved"][0]
    correct = result["correct"]
    options = "  \n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(result["options"])])
    answer = result["options"][correct] if 0 <= correct < len(result["options"]) else "Unknown"

    raw_exp = result.get("explanation", "") or ""
    cleaned_exp = refine_explanation(raw_exp, result["question"]) if raw_exp else ""

    response = (
        f"Q: {result['question']}\n\n"
        f"Options:\n{options}\n\n"
        f"Correct Answer: {chr(65+correct)}: {answer}"
    )
    if cleaned_exp:
        response += f"\n\nExplanation:\n{cleaned_exp}"
    return {"response": response}

def fallback_node(state: State) -> dict:
    return {"response": "Sorry, I couldn't find an answer in the dataset."}

def build_chatbot_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("respond", respond_node)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, {"respond": "respond", "fallback": "fallback"})
    g.add_edge("respond", END)
    g.add_edge("fallback", END)
    return g.compile()
