from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import cast

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_myla.domain.models import ArticleStatus, LibrarianOutput, PrereqEdge, ResearchOutput
from agent_myla.domain.state import TutorState
from agent_myla.llm.factory import get_librarian_llm
from agent_myla.prompts.librarian_prompts import LIBRARIAN_SYSTEM_PROMPT, librarian_user_prompt

log = logging.getLogger(__name__)


def _derive_status(r_layer_outputs: list[ResearchOutput]) -> ArticleStatus:
    """Derive KB status from average research confidence — LLM must not decide this."""
    if not r_layer_outputs:
        return "stub"
    avg = sum(r.content_confidence for r in r_layer_outputs) / len(r_layer_outputs)
    if avg < 0.4:
        return "stub"
    if avg < 0.7:
        return "fuzzy"
    if avg < 0.9:
        return "working"
    return "deep"


class _LibrarianAssembly(BaseModel):
    """Internal structured output for the librarian."""

    frontmatter: dict[str, str] = Field(description="YAML serialisable note metadata")
    markdown_body: str = Field(
        description="The entire body of the explained topic given by the user initially"
    )
    prereq_edges: list[PrereqEdge] = Field(
        default_factory=list,
        description="Prerequisite edges to add to the concept graph",
    )
    reconciliation_flags: list[str] = Field(
        default_factory=list,
        description="contradictions/updates vs existing notes that need user review",
    )


async def librarian_node(state: TutorState, llm: BaseChatModel | None = None) -> dict[str, object]:
    """Assembles the three research layers into one note and reconcile against the KB."""
    research_outputs: list[ResearchOutput] = state.get("research_outputs", [])

    rop_by_concepts: dict[str, list[ResearchOutput]] = defaultdict(list)

    # group the different research levels per concept
    for output in research_outputs:
        rop_by_concepts[output.concept_id].append(output)

    model: BaseChatModel = llm if llm is not None else get_librarian_llm()
    chain = model.with_structured_output(_LibrarianAssembly, method="function_calling")

    async def _assemble_one(
        concept_id: str, r_layer_outputs: list[ResearchOutput]
    ) -> LibrarianOutput:
        concept_name = r_layer_outputs[0].concept_name
        log.info("Librarian assembling concept: %s (%d layers)", concept_id, len(r_layer_outputs))
        messages = [
            SystemMessage(content=LIBRARIAN_SYSTEM_PROMPT),
            HumanMessage(
                content=librarian_user_prompt(
                    concept_name=concept_name,
                    concept_id=concept_id,
                    research_outputs=r_layer_outputs,
                )
            ),
        ]
        raw: _LibrarianAssembly = cast(_LibrarianAssembly, await chain.ainvoke(messages))
        return LibrarianOutput(
            concept_id=concept_id,
            frontmatter=raw.frontmatter,
            markdown_body=raw.markdown_body,
            new_status=_derive_status(r_layer_outputs=r_layer_outputs),
            prereq_edges=raw.prereq_edges,
            reconciliation_flags=raw.reconciliation_flags,
        )

    assembled_output: list[LibrarianOutput] = await asyncio.gather(
        *[_assemble_one(cid, layers) for cid, layers in rop_by_concepts.items()]
    )
    notes = [f"[librarian] {o.concept_id} → status={o.new_status}" for o in assembled_output]

    log.info("Librarian assembled %d notes", len(assembled_output))

    return {
        "assembled_output": assembled_output,
        "node_notes": notes,
    }
