from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class State(TypedDict):
    papers: List[str]
    graph_nodes: List[str]
    obsidian_notes: List[str]

# Define Nodes
def ingest_step(state):
    # Call the ingestion logic from Phase 2
    # Return updated state
    return state

def graph_build_step(state):
    # Call MCP server to create notes
    return state

def visualize_step(state):
    # Trigger visualization tools
    return state

# Build Graph
workflow = StateGraph(State)
workflow.add_node("ingest", ingest_step)
workflow.add_node("graph", graph_build_step)
workflow.add_node("visualize", visualize_step)
workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "graph")
workflow.add_edge("graph", "visualize")
workflow.add_edge("visualize", END)

app = workflow.compile()