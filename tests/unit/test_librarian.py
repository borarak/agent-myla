from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_myla.domain.models import ResearchOutput, Source
from agent_myla.graph.nodes.librarian import _derive_status, _LibrarianAssembly, librarian_node


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


def _mock_llm(assembly: _LibrarianAssembly) -> MagicMock:
    """Build a MagicMock LLM whose chain.ainvoke returns the given assembly."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = assembly
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


@pytest.mark.asyncio  # type: ignore[misc, untyped-decorator]
async def test_librarian_groups_by_concept(fake_assembly: _LibrarianAssembly) -> None:
    """Node groups research outputs by concept_id and calls the LLM once per concept."""
    state = {
        "research_outputs": [
            _make_research_output("attention", "intuition"),
            _make_research_output("attention", "mechanism"),
            _make_research_output("attention", "formalism"),
        ]
    }

    result: dict[str, Any] = await librarian_node(state, llm=_mock_llm(fake_assembly))  # type: ignore[arg-type, assignment]

    # one concept → one LLM call (verify via output count)
    assert len(result["assembled_output"]) == 1
    assert result["assembled_output"][0].concept_id == "attention"


async def test_librarian_handles_two_concepts(fake_assembly: _LibrarianAssembly) -> None:
    state = {
        "research_outputs": [
            _make_research_output("qlora", "intuition", content_confidence=0.6),
            _make_research_output("attention", "intuition", content_confidence=0.6),
        ]
    }

    result: dict[str, Any] = await librarian_node(state=state, llm=_mock_llm(fake_assembly))  # type: ignore[arg-type, assignment]

    ids = {o.concept_id for o in result["assembled_output"]}
    assert ids == {"attention", "qlora"}


async def test_librarian_calls_llm_once_per_concept(fake_assembly: _LibrarianAssembly) -> None:
    """LLM chain.ainvoke is called exactly once per unique concept."""
    mock_llm = _mock_llm(fake_assembly)
    state = {
        "research_outputs": [
            _make_research_output("attention", "intuition"),
            _make_research_output("attention", "mechanism"),
            _make_research_output("attention", "formalism"),
        ]
    }

    await librarian_node(state, llm=mock_llm)  # type: ignore[arg-type]

    mock_llm.with_structured_output.return_value.ainvoke.assert_called_once()


async def test_librarian_empty_research_outputs() -> None:
    """Node handles empty research_outputs gracefully without invoking the chain."""
    mock_llm = MagicMock()
    mock_chain = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_chain

    result = await librarian_node({"research_outputs": []}, llm=mock_llm)  # type: ignore[arg-type]

    mock_chain.ainvoke.assert_not_called()
    assert result["assembled_output"] == []
    assert result["node_notes"] == []


async def test_librarian_status_derived_from_confidence(fake_assembly: _LibrarianAssembly) -> None:
    """new_status is derived from confidence scores, not from the LLM response."""
    state = {
        "research_outputs": [
            _make_research_output("attention", "intuition", content_confidence=0.5),
            _make_research_output("attention", "mechanism", content_confidence=0.5),
            _make_research_output("attention", "formalism", content_confidence=0.5),
        ]
    }

    result: dict[str, Any] = await librarian_node(state, llm=_mock_llm(fake_assembly))  # type: ignore[arg-type, assignment]

    assert result["assembled_output"][0].new_status == "fuzzy"  # avg 0.5 → fuzzy


async def test_librarian_node_notes_report_status(fake_assembly: _LibrarianAssembly) -> None:
    state = {
        "research_outputs": [
            _make_research_output("attention", "intuition"),
        ]
    }

    result: dict[str, Any] = await librarian_node(state, llm=_mock_llm(fake_assembly))  # type: ignore[arg-type, assignment]

    assert any("attention" in note for note in result["node_notes"])


# --------------------------------------------------------------------------- #
# _derive_status unit tests                                                    #
# --------------------------------------------------------------------------- #


def test_derive_status_empty_returns_stub() -> None:
    assert _derive_status([]) == "stub"


def test_derive_status_very_low_confidence_returns_stub() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.2)]
    assert _derive_status(outputs) == "stub"


def test_derive_status_boundary_below_04_is_stub() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.39)]
    assert _derive_status(outputs) == "stub"


def test_derive_status_at_04_is_fuzzy() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.4)]
    assert _derive_status(outputs) == "fuzzy"


def test_derive_status_mid_range_is_fuzzy() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.6)]
    assert _derive_status(outputs) == "fuzzy"


def test_derive_status_at_07_is_working() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.7)]
    assert _derive_status(outputs) == "working"


def test_derive_status_high_confidence_is_working() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.85)]
    assert _derive_status(outputs) == "working"


def test_derive_status_at_09_is_deep() -> None:
    outputs = [_make_research_output("x", "intuition", content_confidence=0.9)]
    assert _derive_status(outputs) == "deep"


def test_derive_status_averages_across_layers() -> None:
    outputs = [
        _make_research_output("x", "intuition", content_confidence=0.6),
        _make_research_output("x", "mechanism", content_confidence=0.8),
    ]
    # avg = 0.7 → working
    assert _derive_status(outputs) == "working"
