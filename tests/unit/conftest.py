"""Shared text fixture"""

from __future__ import annotations

from agent_myla.domain.models import KBContent


def mock_kb() -> KBContent:
    """A mock KB which stores a store a snapshot of the actual KB
    Mock KB has two concepts
    -> quantization (level is mastered / deep) -> and should only be linked and not-re-researched
    -> qlora (stub) -> which will need research
    """
    return KBContent(
        existing_concepts=dict(
            quantization=dict(
                status="deep",
            ),
            qlora=dict(
                status="stub",
            ),
        )
    )
