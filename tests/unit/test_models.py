from __future__ import annotations

from agent_myla.domain.models import (
    ConceptNode,
)


def test_concept_node_minimal() -> None:
    node = ConceptNode(
        concept_id="qlora",
        concept_name="QloRA",
        depth=2,
        rationale="Test Concept needed",
        existing_status="missing",
    )

    assert node.prereqs == []
    assert node.existing_status == "missing"
