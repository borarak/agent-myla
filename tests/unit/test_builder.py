"""Harness-stage tests for the graph builder and individual nodes."""

from __future__ import annotations

import pytest
from langgraph.graph.state import CompiledStateGraph

from agent_myla.domain.state import TutorState
from agent_myla.graph.builder import build_graph
from agent_myla.graph.nodes import (
    kb_writer_node,
    librarian_node,
    planner_node,
    research_node,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")  # type: ignore[misc]
def graph() -> CompiledStateGraph:
    return build_graph()


@pytest.fixture  # type: ignore[misc]
def base_state() -> TutorState:
    return TutorState(user_topic="qlora")


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_build_graph_compiles(graph: CompiledStateGraph) -> None:
    assert isinstance(graph, CompiledStateGraph)


def test_graph_has_expected_nodes(graph: CompiledStateGraph) -> None:
    node_ids = set(graph.get_graph().nodes.keys())
    assert {"planner", "researcher", "librarian", "kb_writer"}.issubset(node_ids)


def test_graph_edges_form_linear_chain(graph: CompiledStateGraph) -> None:
    edge_pairs = {(e.source, e.target) for e in graph.get_graph().edges}
    expected = {
        ("__start__", "planner"),
        ("planner", "researcher"),
        ("researcher", "librarian"),
        ("librarian", "kb_writer"),
        ("kb_writer", "__end__"),
    }
    assert expected.issubset(edge_pairs)


# ---------------------------------------------------------------------------
# Node harnesses — planner
# ---------------------------------------------------------------------------


async def test_planner_node_returns_expected_keys(base_state: TutorState) -> None:
    result = await planner_node(base_state)
    assert "assistant_message" in result
    assert "concepts_dag" in result
    assert "node_notes" in result


async def test_planner_node_message_contains_topic(base_state: TutorState) -> None:
    result = await planner_node(base_state)
    msg = result["assistant_message"]
    assert isinstance(msg, str)
    assert "qlora" in msg


async def test_planner_node_concepts_dag_is_list(base_state: TutorState) -> None:
    result = await planner_node(base_state)
    assert isinstance(result["concepts_dag"], list)


async def test_planner_node_notes_is_list(base_state: TutorState) -> None:
    result = await planner_node(base_state)
    assert isinstance(result["node_notes"], list)
    assert len(result["node_notes"]) >= 1


# ---------------------------------------------------------------------------
# Node harnesses — researcher / librarian / kb_writer
# ---------------------------------------------------------------------------


async def test_research_node_returns_node_notes(base_state: TutorState) -> None:
    result = await research_node(base_state)  # type: ignore[arg-type]
    assert "node_notes" in result
    assert isinstance(result["node_notes"], list)


async def test_librarian_node_returns_node_notes(base_state: TutorState) -> None:
    result = await librarian_node(base_state)
    assert "node_notes" in result
    assert isinstance(result["node_notes"], list)


async def test_kb_writer_node_returns_node_notes(base_state: TutorState) -> None:
    result = await kb_writer_node(base_state)
    assert "node_notes" in result
    assert isinstance(result["node_notes"], list)


# ---------------------------------------------------------------------------
# End-to-end graph run
# ---------------------------------------------------------------------------


async def test_graph_run_sets_assistant_message(graph: CompiledStateGraph) -> None:
    result = await graph.ainvoke(TutorState(user_topic="qlora"))
    assert "assistant_message" in result
    assert result["assistant_message"]


async def test_graph_run_accumulates_all_node_notes(graph: CompiledStateGraph) -> None:
    result = await graph.ainvoke(TutorState(user_topic="qlora"))
    assert "node_notes" in result
    # one note emitted per node (planner + researcher + librarian + kb_writer)
    assert len(result["node_notes"]) == 4


async def test_graph_run_stream_no_none_updates(graph: CompiledStateGraph) -> None:
    async for event in graph.astream(TutorState(user_topic="qlora"), stream_mode="updates"):
        for node_name, node_result in event.items():
            assert node_result is not None, f"node '{node_name}' emitted a None update"
