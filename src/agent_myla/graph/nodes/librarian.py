from __future__ import annotations

import logging
from collections import defaultdict
from typing import cast

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_myla.domain.models import ArticleStatus, LibrarianOutput, ResearchOutput
from agent_myla.domain.state import TutorState
from agent_myla.llm.factory import get_librarian_llm
from agent_myla.prompts.librarian_prompts import LIBRARIAN_SYSTEM_PROMPT, librarian_user_prompt

log = logging.getLogger(__name__)


class _LibrarianAssembly(BaseModel):
    """Internal structired output for the liobrarian - so we need to send known values"""

    frontmatter: dict[str, str] = Field(description="YAML serialisable note metadata")
    markdown_body: str = Field(
        description="The entire body of the explained topic given by the user initially"
    )
    new_status: ArticleStatus
    prereq_edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="(from_concept, to_concept) edges to add to the graph",
    )
    reconciliation_flags: list[str] = Field(
        default_factory=list,
        description="contradictions/updates vs existing notes that need user review",
    )


async def librarian_node(state: TutorState) -> dict[str, object]:
    """Assembles the three research layers into one note and reconcile against the KB."""
    research_outputs: list[ResearchOutput] = state.get("research_outputs", [])

    rop_by_concepts = defaultdict(list)

    # group the different research levels per concept
    for output in research_outputs:
        rop_by_concepts[output.concept_id].append(output)

    model: BaseChatModel = get_librarian_llm()
    chain = model.with_structured_output(_LibrarianAssembly)

    assembled_output: list[LibrarianOutput] = []
    notes: list[str] = []

    for concept_id, layers in rop_by_concepts.items():
        concept_name = layers[0].concept_id  # fallback; ideally pass display_name
        log.info("Librarian assembling concept: %s (%d layers)", concept_id, len(layers))

        messages = [
            SystemMessage(content=LIBRARIAN_SYSTEM_PROMPT),
            HumanMessage(
                content=librarian_user_prompt(
                    concept_name=concept_name,
                    concept_id=concept_id,
                    research_outputs=layers,
                )
            ),
        ]

        raw: _LibrarianAssembly = cast(_LibrarianAssembly, await chain.ainvoke(messages))

        librarian_output = LibrarianOutput(
            concept_id=concept_id,
            frontmatter=raw.frontmatter,
            markdown_body=raw.markdown_body,
            new_status=raw.new_status,
            prereq_edges=raw.prereq_edges,
            reconciliation_flags=raw.reconciliation_flags,
        )
        assembled_output.append(librarian_output)
        notes.append(f"[librarian] {concept_id} → status={raw.new_status}")

    log.info("Librarian assembled %d notes", len(assembled_output))

    return {
        "assembled_output": assembled_output,
        "node_notes": [f"[librarian] assembled, final notes: {assembled_output}"],
    }
