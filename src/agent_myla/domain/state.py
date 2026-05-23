from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from agent_myla.domain.models import (
    ConceptNode,
    KBContent,
    LibrarianOutput,
    ResearchLayer,
    ResearchOutput,
)


class TutorState(TypedDict, total=False):
    """The top-level graph state - the tutor itself
    total=False lets a node only retun the keys it writes"""

    # user str input
    user_topic: str

    # Written by Planner
    kb_content: KBContent
    concepts_dag: list[ConceptNode]

    # Written by the Researcher
    research_outputs: Annotated[list[ResearchOutput], operator.add]

    # Written by Librarian
    assembled_output: list[LibrarianOutput]

    # Written by the KB writer
    kb_writes: list[str]

    # Shared across nodes
    assistant_message: str | tuple[str, ...]
    node_notes: Annotated[list[str], operator.add]


class ResearchInput(TypedDict):
    """Per invocarton state sent to Researcher - to avoid sending full TutorState"""

    concept_id: str
    display_name: str
    rationale: str
    prereqs_known: list[str]
    parent_topic: str
    layer: ResearchLayer
