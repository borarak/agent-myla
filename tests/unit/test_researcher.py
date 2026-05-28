"""Unit tests for the researcher node and its prompt helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_myla.domain.models import ComputedExample, Source
from agent_myla.graph.nodes.researcher import _ResearchContent, research_node
from agent_myla.prompts.researcher_prompts import get_system_prompt, researcher_user_prompt

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_state(
    layer: str = "intuition",
    prereqs: list[str] | None = None,
    concept_index: int = 0,
) -> dict[str, Any]:
    return {
        "concept_id": "attention",
        "display_name": "Attention Mechanism",
        "rationale": "Core building block of transformers",
        "prereqs_known": prereqs if prereqs is not None else ["linear-algebra", "softmax"],
        "parent_topic": "transformers",
        "layer": layer,
        "concept_index": concept_index,
    }


def _make_content(
    confidence: float = 0.8,
    fuzzy: list[str] | None = None,
    examples: list[ComputedExample] | None = None,
) -> _ResearchContent:
    return _ResearchContent(
        content_markdown="## Attention\nAttention is all you need.",
        sources=[
            Source(
                source_title="Vaswani 2017",
                source_url="https://arxiv.org/abs/1706.03762",
                source_type="arxiv",
            )
        ],
        computed_examples=examples or [],
        content_confidence=confidence,
        fuzzy_areas=fuzzy or [],
    )


def _mock_llm(content: _ResearchContent) -> MagicMock:
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = content
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


# --------------------------------------------------------------------------- #
# research_node tests                                                          #
# --------------------------------------------------------------------------- #


async def test_research_node_returns_required_keys() -> None:
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(_make_content()))

    assert "research_outputs" in result
    assert "node_notes" in result


async def test_research_node_produces_one_output() -> None:
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(_make_content()))

    assert len(result["research_outputs"]) == 1


async def test_research_node_maps_concept_id() -> None:
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(_make_content()))

    assert result["research_outputs"][0].concept_id == "attention"


async def test_research_node_maps_concept_name() -> None:
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(_make_content()))

    assert result["research_outputs"][0].concept_name == "Attention Mechanism"


async def test_research_node_maps_research_layer() -> None:
    result: dict[str, Any] = await research_node(
        _make_state(layer="mechanism"), llm=_mock_llm(_make_content())
    )

    assert result["research_outputs"][0].research_layer == "mechanism"


async def test_research_node_maps_confidence() -> None:
    result: dict[str, Any] = await research_node(
        _make_state(), llm=_mock_llm(_make_content(confidence=0.65))
    )

    assert result["research_outputs"][0].content_confidence == pytest.approx(0.65)


async def test_research_node_maps_content_markdown() -> None:
    content = _make_content()
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(content))

    assert result["research_outputs"][0].content_markdown == content.content_markdown


async def test_research_node_maps_sources() -> None:
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(_make_content()))

    assert len(result["research_outputs"][0].sources) == 1
    assert result["research_outputs"][0].sources[0].source_title == "Vaswani 2017"


async def test_research_node_maps_computed_examples() -> None:
    example = ComputedExample(
        description="Softmax of [1,2,3]",
        inputs={"x": [1, 2, 3]},
        output="[0.09, 0.24, 0.67]",
        tool_used="numpy",
    )
    result: dict[str, Any] = await research_node(
        _make_state(layer="mechanism"), llm=_mock_llm(_make_content(examples=[example]))
    )

    assert len(result["research_outputs"][0].computed_examples) == 1
    assert result["research_outputs"][0].computed_examples[0].description == "Softmax of [1,2,3]"


async def test_research_node_maps_fuzzy_areas() -> None:
    result: dict[str, Any] = await research_node(
        _make_state(), llm=_mock_llm(_make_content(fuzzy=["multi-head details"]))
    )

    assert result["research_outputs"][0].fuzzy_areas == ["multi-head details"]


async def test_research_node_note_mentions_layer() -> None:
    result: dict[str, Any] = await research_node(
        _make_state(layer="formalism"), llm=_mock_llm(_make_content())
    )

    note: str = result["node_notes"][0]
    assert "formalism" in note


async def test_research_node_note_mentions_concept_name() -> None:
    result: dict[str, Any] = await research_node(_make_state(), llm=_mock_llm(_make_content()))

    note: str = result["node_notes"][0]
    assert "Attention Mechanism" in note


async def test_research_node_uses_function_calling_method() -> None:
    """method='function_calling' avoids OpenAI strict-mode schema rejections."""
    mock_llm = _mock_llm(_make_content())
    await research_node(_make_state(), llm=mock_llm)

    mock_llm.with_structured_output.assert_called_once_with(
        _ResearchContent, method="function_calling"
    )


async def test_research_node_sends_two_messages() -> None:
    mock_llm = _mock_llm(_make_content())
    await research_node(_make_state(), llm=mock_llm)

    chain = mock_llm.with_structured_output.return_value
    chain.ainvoke.assert_called_once()
    messages = chain.ainvoke.call_args[0][0]
    assert len(messages) == 2  # SystemMessage + HumanMessage


async def test_research_node_empty_prereqs() -> None:
    result: dict[str, Any] = await research_node(
        _make_state(prereqs=[]), llm=_mock_llm(_make_content())
    )

    assert len(result["research_outputs"]) == 1


async def test_research_node_all_three_layers() -> None:
    for layer in ("intuition", "mechanism", "formalism"):
        result: dict[str, Any] = await research_node(
            _make_state(layer=layer), llm=_mock_llm(_make_content())
        )
        assert result["research_outputs"][0].research_layer == layer


# --------------------------------------------------------------------------- #
# researcher_prompts tests                                                     #
# --------------------------------------------------------------------------- #


def test_get_system_prompt_intuition() -> None:
    prompt = get_system_prompt("intuition")
    assert "intuition" in prompt.lower() or "mental model" in prompt.lower()


def test_get_system_prompt_mechanism() -> None:
    prompt = get_system_prompt("mechanism")
    assert "mechanism" in prompt.lower() or "HOW" in prompt


def test_get_system_prompt_formalism() -> None:
    prompt = get_system_prompt("formalism")
    assert "formalism" in prompt.lower() or "mathematical" in prompt.lower()


def test_researcher_user_prompt_with_prereqs() -> None:
    prompt = researcher_user_prompt(
        concept_name="Attention",
        rationale="Core to transformers",
        prereqs_known=["linear-algebra", "softmax"],
        layer="intuition",
    )
    assert "Attention" in prompt
    assert "linear-algebra" in prompt
    assert "softmax" in prompt
    assert "intuition" in prompt


def test_researcher_user_prompt_no_prereqs() -> None:
    prompt = researcher_user_prompt(
        concept_name="Attention",
        rationale="Core to transformers",
        prereqs_known=[],
        layer="mechanism",
    )
    assert "none" in prompt
    assert "Attention" in prompt
