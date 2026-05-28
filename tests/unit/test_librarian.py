from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_myla.domain.models import ResearchOutput, Source
from agent_myla.graph.nodes.librarian import _LibrarianAssembly, librarian_node


def _make_research_output(
    concept_id: str, layer: str, content_confidence: float = 0.8
) -> ResearchOutput:
    return ResearchOutput(
        concept_id=concept_id,
        concept_name=concept_id,
        research_layer=layer,  # type: ignore[arg-type]
        content_confidence=content_confidence,
        computed_examples=[],
        content_markdown="",
        fuzzy_areas=[],
        sources=[Source(source_title="Test", source_url="https://example.com", source_type="blog")],
    )


@pytest.fixture(scope="module")  # type: ignore[misc, untyped-decorator]
def fake_assembly() -> _LibrarianAssembly:
    return _LibrarianAssembly(
        frontmatter={"title": "Attention", "concept_id": "attention", "status": "working"},
        markdown_body="## Intuition\n...\n## Mechanism\n...\n## Formalism\n...",
        prereq_edges=[],
        reconciliation_flags=[],
    )


@pytest.mark.asyncio  # type: ignore[misc, untyped-decorator]
async def test_librarian_groups_by_concept(fake_assembly: _LibrarianAssembly) -> None:
    """Node groups research outputs by concept_id and calls the LLM once per concept."""
    mock_chain = AsyncMock(return_value=fake_assembly)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain

    state = {
        "research_outputs": [
            _make_research_output("attention", "intuition"),
            _make_research_output("attention", "mechanism"),
            _make_research_output("attention", "formalism"),
        ]
    }

    result: dict[str, Any] = await librarian_node(state, llm=mock_llm)  # type: ignore[arg-type, assignment]

    assert mock_chain.ainvoke.call_count == 1  # one concept → one LLM call
    assert len(result["assembled_output"]) == 1
    assert result["assembled_output"][0].concept_id == "attention"


async def test_librarian_handles_two_concepts(fake_assembly: _LibrarianAssembly) -> None:
    mock_chain = AsyncMock(return_value=fake_assembly)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain

    state = {
        "research_outputs": [
            _make_research_output("qlora", "intuition", content_confidence=0.6),
            _make_research_output("attention", "intuition", content_confidence=0.6),
        ]
    }

    result: dict[str, Any] = await librarian_node(state=state, llm=mock_llm)  # type: ignore[arg-type, assignment]

    assert mock_chain.ainvoke.call_count == 2
    ids = {o.concept_id for o in result["assembled_output"]}
    assert ids == {"attention", "qlora"}


async def test_librarian_empty_research_outputs() -> None:
    """Node handles empty research_outputs gracefully without calling the LLM."""
    mock_llm = MagicMock()
    result = await librarian_node({"research_outputs": []}, llm=mock_llm)  # type: ignore[arg-type]

    mock_llm.with_structured_output.assert_not_called()
    assert result["assembled_output"] == []
    assert result["node_notes"] == []


async def test_librarian_status_derived_from_confidence(fake_assembly: _LibrarianAssembly) -> None:
    """new_status is derived from confidence scores, not from the LLM response."""
    mock_chain = AsyncMock(return_value=fake_assembly)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain

    state = {
        "research_outputs": [
            _make_research_output("attention", "intuition", content_confidence=0.5),
            _make_research_output("attention", "mechanism", content_confidence=0.5),
            _make_research_output("attention", "formalism", content_confidence=0.5),
        ]
    }

    result: dict[str, Any] = await librarian_node(state, llm=mock_llm)  # type: ignore[arg-type, assignment]

    assert result["assembled_output"][0].new_status == "fuzzy"  # avg 0.5 → fuzzy
