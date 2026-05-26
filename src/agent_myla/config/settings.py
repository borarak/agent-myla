from __future__ import annotations

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings for the agent to propose, plan and research"""

    max_concept_depth: int = Field(
        ge=0, le=4, description="Maximum recursion depth of a root topic into sub-topics"
    )
