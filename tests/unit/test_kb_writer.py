from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr

from agent_myla.config.settings import Settings
from agent_myla.domain.models import LibrarianOutput
from agent_myla.graph.nodes.kb_writer import kb_writer_node
from agent_myla.kb.store import KBStore


def _make_note(concept_id: str = "attention", status: str = "working") -> LibrarianOutput:
    return LibrarianOutput(
        concept_id=concept_id,
        frontmatter={"title": concept_id.title(), "concept_id": concept_id, "status": status},
        markdown_body="## Intuition\nSome content.",
        new_status=status,  # type: ignore[arg-type]
        prereq_edges=[],
        reconciliation_flags=[],
    )


# --------------------------------------------------------------------------- #
# KBStore unit tests                                                           #
# --------------------------------------------------------------------------- #


def test_store_writes_file(tmp_path: Path) -> None:
    store = KBStore(tmp_path)
    filename = store.write_note(_make_note("attention"))

    assert filename == "attention.md"
    content = (tmp_path / "attention.md").read_text()
    assert "---" in content
    assert "concept_id: attention" in content
    assert "## Intuition" in content


def test_store_flushes_index(tmp_path: Path) -> None:
    store = KBStore(tmp_path)
    store.flush_index([_make_note("attention"), _make_note("qlora", "fuzzy")])

    index = json.loads((tmp_path / "index.json").read_text())
    assert index["attention"]["status"] == "working"
    assert index["qlora"]["path"] == "qlora.md"


def test_store_index_merge_preserves_existing(tmp_path: Path) -> None:
    store = KBStore(tmp_path)
    (tmp_path / "index.json").write_text(
        json.dumps({"old-concept": {"status": "deep", "path": "old-concept.md"}}),
        encoding="utf-8",
    )
    store.flush_index([_make_note("attention")])
    index = json.loads((tmp_path / "index.json").read_text())
    assert "old-concept" in index
    assert "attention" in index


def test_store_write_is_idempotent(tmp_path: Path) -> None:
    store = KBStore(tmp_path)
    note = _make_note("attention")
    store.write_note(note)
    store.write_note(note)
    assert (tmp_path / "attention.md").exists()


# --------------------------------------------------------------------------- #
# Node tests                                                                   #
# --------------------------------------------------------------------------- #


async def test_kb_writer_node_happy_path(tmp_path: Path) -> None:
    fake_settings = Settings.model_construct(kb_path=tmp_path, openai_api_key=SecretStr("sk-fake"))
    with patch("agent_myla.graph.nodes.kb_writer.get_settings", return_value=fake_settings):
        result = await kb_writer_node({"assembled_output": [_make_note("attention")]})  # type: ignore[arg-type]

    assert result["kb_writes"] == ["attention.md"]
    assert (tmp_path / "attention.md").exists()
    assert (tmp_path / "index.json").exists()


async def test_kb_writer_node_writes_multiple(tmp_path: Path) -> None:
    fake_settings = Settings.model_construct(kb_path=tmp_path, openai_api_key=SecretStr("sk-fake"))
    notes = [_make_note("attention"), _make_note("qlora", "fuzzy")]
    with patch("agent_myla.graph.nodes.kb_writer.get_settings", return_value=fake_settings):
        result = await kb_writer_node({"assembled_output": notes})  # type: ignore[arg-type]

    assert set(result["kb_writes"]) == {"attention.md", "qlora.md"}  # type: ignore[arg-type, call-overload]
    assert len(result["node_notes"]) == 2  # type: ignore[arg-type]


async def test_kb_writer_node_empty() -> None:
    result = await kb_writer_node({"assembled_output": []})  # type: ignore[arg-type]
    assert result["kb_writes"] == []
    assert result["node_notes"] == []
