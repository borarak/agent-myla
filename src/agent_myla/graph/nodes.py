"""Graph nodes"""

from __future__ import annotations

import logging

from agent_myla.domain.state import TutorState

log = logging.getLogger(__name__)


async def planner_node(state: TutorState) -> dict[str, object]:
    """Decomposes a given topic into smaller concepts, create a pre-requisites DAG
    after checking the exiting KB
    """

    topic = state["user_topic"]
    log.info("Planner receied a new topic=%s", topic)
    greeting_stub = (
        f"You asked for research into topic: {topic}. "
        "I'm the Planner to help facilitate this research. "
        "However, I'm not completely wired in yet!"
    )

    return {
        "concepts_dag": [],
        "assistant_message": greeting_stub,
        "node_notes": [f"[planner] greeted; topic={topic!r}"],
    }


async def research_node(state: TutorState) -> dict[str, object]:
    """STUB: becomes a parallel fan-out of 3 Researchers (Send API) once the Planner
    emits a real concept DAG to fan out over."""
    log.info("research stub")
    return {"node_notes": ["[research] stub — no sections produced yet"]}


async def librarian_node(state: TutorState) -> dict[str, object]:
    """STUB: will assemble the three layers into one note and reconcile against the KB."""
    log.info("librarian stub")
    return {"node_notes": ["[librarian] stub — no notes assembled yet"]}


async def kb_writer_node(state: TutorState) -> dict[str, object]:
    """STUB: will persist notes to markdown + index.json and git-commit them."""
    log.info("kb_writer stub")
    return {"node_notes": ["[kb_writer] stub — nothing written yet"]}
