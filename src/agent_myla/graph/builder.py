"""Build the LangGraph

Ver 1: START -> Planner -> Researcher -> Librarian -> kb_writer -> END
Research is a stub currently
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_myla.domain.state import TutorState
from agent_myla.graph.nodes import kb_writer_node, librarian_node, planner_node, research_node


def build_graph() -> CompiledStateGraph[TutorState]:
    builder = StateGraph(TutorState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", research_node)
    builder.add_node("librarian", librarian_node)
    builder.add_node("kb_writer", kb_writer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "librarian")
    builder.add_edge("librarian", "kb_writer")
    builder.add_edge("kb_writer", END)

    return builder.compile()
