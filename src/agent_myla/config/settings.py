from __future__ import annotations

import functools

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[misc]
    """Application settings for the agent to propose, plan and research"""

    model_config = SettingsConfigDict(env_prefix="MYLA_", env_file=".env", extra="ignore")

    # LLM credentials
    openai_api_key: SecretStr = Field(description="OpenAI API key")

    # Per-role model selection
    planner_model: str = Field(
        default="gpt-4o-mini", description="OpenAI model used by the planner role"
    )

    # Per-role model selection
    researcher_model: str = Field(
        default="gpt-4o-mini", description="OpenAI model used by the planner role"
    )

    # Per-role model selection
    librarian_model: str = Field(
        default="gpt-4o-mini", description="OpenAI model used by the planner role"
    )

    # Graph caps
    max_concept_depth: int = Field(
        default=2, ge=0, le=4, description="Maximum recursion depth of a root topic into sub-topics"
    )


@functools.cache
def get_settings() -> Settings:
    return Settings()
