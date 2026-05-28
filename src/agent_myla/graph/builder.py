"""Build the LangGraph

Ver 1: START -> Planner -> Researcher -> Librarian -> kb_writer -> END
Research is a stub currently
"""

from __future__ import annotations

import logging
from typing import Literal, cast, get_args

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from agent_myla.domain.models import ConceptNode, ResearchLayer
from agent_myla.domain.state import ResearchInput, TutorState
from agent_myla.graph.nodes import kb_writer_node, librarian_node, planner_node, research_node

log = logging.getLogger(__name__)

_RESEARCH_MODES = Literal["intuition", "mechanism", "formalism"]


def distributed_research(state: TutorState) -> list[Send]:
    """Condition edge: emit one Send per (concept, intuition / formalism / mechanism) layer
    for concepts with status != deep
    """
    concepts_to_research: list[ConceptNode] = state.get("concepts_dag", [])
    parent_topic: str = state.get("user_topic") or ""

    if parent_topic == "":
        log.warning("distributed research recevied a empty parent topic")

    sends: list[Send] = []

    for c_idx, c in enumerate(concepts_to_research):
        if c.existing_status == "deep":
            continue
        for r_layer in get_args(ResearchLayer):
            sends.append(
                Send(
                    "researcher",
                    ResearchInput(
                        concept_id=c.concept_id,
                        display_name=c.concept_name,
                        layer=cast(ResearchLayer, r_layer),
                        parent_topic=parent_topic,
                        rationale=c.rationale,
                        prereqs_known=c.prereqs,
                        concept_index=c_idx + 1,
                    ),
                )
            )
    log.info(f"distributed researched successfully created: {len(sends)} layers...")
    return sends


def build_graph() -> CompiledStateGraph[TutorState]:
    builder = StateGraph(TutorState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", research_node)
    builder.add_node("librarian", librarian_node)
    builder.add_node("kb_writer", kb_writer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", distributed_research, ["researcher"])
    builder.add_edge("researcher", "librarian")
    builder.add_edge("librarian", "kb_writer")
    builder.add_edge("kb_writer", END)

    return builder.compile()
