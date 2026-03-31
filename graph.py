from langgraph.graph import StateGraph, END

from state import LeadState
from agents import (
    planner_agent,
    researcher_agent,
    graph_architect_agent,
)


def _route_after_planner(state: LeadState) -> str:
    goal_type = state.get("goal_type", "ingestion")
    if goal_type == "query":
        return "query"
    return "ingestion"


def build_app():
    """
    Build the LangGraph StateGraph wiring together the Planner,
    Researcher, and Graph Architect agents.
    """
    workflow = StateGraph(LeadState)

    workflow.add_node("planner", planner_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("graph_architect", graph_architect_agent)

    workflow.set_entry_point("planner")

    # After planning, branch to ingestion or query path
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "ingestion": "researcher",
            "query": "graph_architect",
        },
    )

    # Ingestion path: researcher -> graph architect -> end
    workflow.add_edge("researcher", "graph_architect")

    # For both ingestion and query, graph architect is the terminal node
    workflow.add_edge("graph_architect", END)

    return workflow.compile()

