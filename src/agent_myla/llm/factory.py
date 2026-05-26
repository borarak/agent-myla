from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agent_myla.config import get_settings


def get_planner_llm() -> BaseChatModel:
    """Return the chat model configured for the planner role."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.planner_model,
        api_key=settings.openai_api_key,
    )
