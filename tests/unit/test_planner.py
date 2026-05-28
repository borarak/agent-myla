from __future__ import annotations

from agent_myla.domain.models import KBContent
from agent_myla.graph.nodes.planner import (
    ConceptProposal,
    create_concept_proposals,
    create_plan_from_proposals,
)
from tests.unit import conftest


def make_proposal(
    concept_id: str,
    depth: int = 0,
    prereqs: list[str] | None = None,
) -> ConceptProposal:
    return ConceptProposal(
        concept_id=concept_id,
        concept_name=concept_id.replace("-", " ").title(),
        reason="test reason",
        depth=depth,
        prereqs=prereqs or [],
    )


# ── create_concept_proposals ──────────────────────────────────────────────────


async def test_create_concept_proposals_returns_at_least_one() -> None:
    proposals = await create_concept_proposals("qlora", KBContent())
    assert len(proposals) >= 1


async def test_create_concept_proposals_has_root_at_depth_zero() -> None:
    proposals = await create_concept_proposals("qlora", KBContent())
    assert any(p.depth == 0 for p in proposals)


async def test_create_concept_proposals_root_matches_topic() -> None:
    proposals = await create_concept_proposals("qlora", KBContent())
    roots = [p for p in proposals if p.depth == 0]
    assert roots[0].concept_id == "qlora"


# ── create_plan_from_proposals ────────────────────────────────────────────────


async def test_create_plan_returns_planner_output() -> None:
    result = await create_plan_from_proposals([make_proposal("qlora")], KBContent())
    assert len(result.concept_dag) == 1
    assert isinstance(result.reasoning, str)


async def test_create_plan_empty_proposals() -> None:
    result = await create_plan_from_proposals([], KBContent())
    assert result.concept_dag == []


async def test_create_plan_deduplicates_concept_ids() -> None:
    proposals = [make_proposal("qlora", depth=0), make_proposal("qlora", depth=0)]
    result = await create_plan_from_proposals(proposals, KBContent())
    ids = [c.concept_id for c in result.concept_dag]
    assert ids.count("qlora") == 1


async def test_create_plan_skips_depth_greater_than_4() -> None:
    proposals = [make_proposal("shallow", depth=0), make_proposal("too-deep", depth=5)]
    result = await create_plan_from_proposals(proposals, KBContent())
    ids = [c.concept_id for c in result.concept_dag]
    assert "too-deep" not in ids
    assert "shallow" in ids


async def test_create_plan_depth_4_is_included() -> None:
    # depth==4 is the boundary — must not be skipped
    proposals = [make_proposal("boundary", depth=4)]
    result = await create_plan_from_proposals(proposals, KBContent())
    assert len(result.concept_dag) == 1


async def test_create_plan_caps_at_10_concepts() -> None:
    proposals = [make_proposal(f"concept-{i}", depth=0) for i in range(15)]
    result = await create_plan_from_proposals(proposals, KBContent())
    assert len(result.concept_dag) == 10


async def test_create_plan_output_sorted_by_depth() -> None:
    proposals = [
        make_proposal("c3", depth=3),
        make_proposal("c1", depth=1),
        make_proposal("c0", depth=0),
        make_proposal("c2", depth=2),
    ]
    result = await create_plan_from_proposals(proposals, KBContent())
    depths = [c.depth for c in result.concept_dag]
    assert depths == sorted(depths)


async def test_create_plan_missing_from_kb_gets_missing_status() -> None:
    proposals = [make_proposal("brand-new", depth=0)]
    result = await create_plan_from_proposals(proposals, KBContent())
    assert result.concept_dag[0].existing_status == "missing"


async def test_create_plan_deep_kb_concept_gets_deep_status() -> None:
    # quantization is "deep" in mock_kb — should not need research
    kb = conftest.mock_kb()
    proposals = [make_proposal("quantization", depth=0)]
    result = await create_plan_from_proposals(proposals, kb)
    assert result.concept_dag[0].existing_status == "deep"


async def test_create_plan_stub_kb_concept_gets_stub_status() -> None:
    kb = conftest.mock_kb()
    proposals = [make_proposal("qlora", depth=0)]
    result = await create_plan_from_proposals(proposals, kb)
    assert result.concept_dag[0].existing_status == "stub"


async def test_create_plan_prereqs_preserved() -> None:
    proposals = [make_proposal("qlora", depth=0, prereqs=["quantization", "lora"])]
    result = await create_plan_from_proposals(proposals, KBContent())
    assert result.concept_dag[0].prereqs == ["quantization", "lora"]
