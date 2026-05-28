from __future__ import annotations

import logging

from agent_myla.config.settings import get_settings
from agent_myla.domain.models import LibrarianOutput
from agent_myla.domain.state import TutorState
from agent_myla.kb.store import KBStore

log = logging.getLogger(__name__)


async def kb_writer_node(state: TutorState) -> dict[str, object]:
    """Persist each assembled note to disk and update index.json."""
    notes: list[LibrarianOutput] = state.get("assembled_output", [])
    if not notes:
        log.info("kb_writer: nothing to write")
        return {"kb_writes": [], "node_notes": []}

    store = KBStore(get_settings().kb_path)
    written: list[str] = [store.write_note(n) for n in notes]
    store.flush_index(notes)

    log_lines = [f"[kb_writer] wrote {p}" for p in written]
    log.info("kb_writer wrote %d notes", len(written))
    return {"kb_writes": written, "node_notes": log_lines}
