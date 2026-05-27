"""Graph nodes"""

from __future__ import annotations

import logging

from agent_myla.domain.state import TutorState

log = logging.getLogger(__name__)


async def librarian_node(state: TutorState) -> dict[str, object]:
    """STUB: will assemble the three layers into one note and reconcile against the KB."""
    log.info("librarian stub")
    return {"node_notes": ["[librarian] stub — no notes assembled yet"]}


async def kb_writer_node(state: TutorState) -> dict[str, object]:
    """STUB: will persist notes to markdown + index.json and git-commit them."""
    log.info("kb_writer stub")
    return {"node_notes": ["[kb_writer] stub — nothing written yet"]}
