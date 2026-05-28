"""Chainlit entry point to chat with the researcher"""

from __future__ import annotations

import logging
from typing import Any

import chainlit as cl

from agent_myla.graph.builder import build_graph

log = logging.getLogger(__name__)

graph = build_graph()


@cl.on_message  # type: ignore[misc, untyped-decorator]
async def on_message(message: cl.Message) -> None:
    start_state: dict[str, str] = {"user_topic": message.content}

    final_state: dict[str, Any] = {}

    async for event in graph.astream(start_state, stream_mode="updates"):  # type: ignore[call-overload]
        for node_name, node_result in event.items():
            if node_result is None:
                log.error("node_result is None for node %s", node_name)
                continue
            final_state.update(node_result)
            notes = node_result.get("node_notes", [])
            async with cl.Step(name=node_name) as step:
                step.output = "\n ".join(notes) if notes else "(step completed)"

    reply = final_state.get("assistant_message", "(no reply produced / stub)")
    await cl.Message(content=reply).send()  # type: ignore[no-untyped-call]
