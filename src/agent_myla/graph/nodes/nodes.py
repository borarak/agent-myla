"""Graph nodes"""

from __future__ import annotations

import logging

from agent_myla.domain.state import TutorState

log = logging.getLogger(__name__)


async def kb_writer_node(state: TutorState) -> dict[str, object]:
    """STUB: will persist notes to markdown + index.json and git-commit them."""
    log.info("kb_writer stub")
    return {"node_notes": ["[kb_writer] stub — nothing written yet"]}
