from __future__ import annotations

import json
from pathlib import Path

from agent_myla.domain.models import LibrarianOutput


def _render_frontmatter(data: dict[str, str]) -> str:
    lines = ["---"]
    for key, value in data.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


class KBStore:
    """Markdown-on-disk KB store. Swap-point: replace with pgvector in v2."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def write_note(self, note: LibrarianOutput) -> str:
        """Write one concept note; return its filename relative to kb root."""
        filename = f"{note.concept_id}.md"
        path = self._root / filename
        content = _render_frontmatter(note.frontmatter) + "\n\n" + note.markdown_body
        path.write_text(content, encoding="utf-8")
        return filename

    def flush_index(self, notes: list[LibrarianOutput]) -> None:
        """Merge-update index.json with the given notes (preserves unrelated entries)."""
        index_path = self._root / "index.json"
        index: dict[str, object] = {}
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        for note in notes:
            index[note.concept_id] = {
                "status": note.new_status,
                "path": f"{note.concept_id}.md",
            }
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
