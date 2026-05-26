"""Graph nodes"""

from __future__ import annotations

import logging

from agent_myla.domain.state import TutorState

log = logging.getLogger(__name__)


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
