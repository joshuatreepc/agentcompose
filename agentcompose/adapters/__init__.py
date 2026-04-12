"""
agentcompose.adapters
=====================

Adapter dispatch layer. Each adapter bridges agentcompose workflows
into a specific agent framework.
"""

import os

from .base import BaseAdapter


def get_adapter(name: str | None = None) -> BaseAdapter:
    """Load an adapter by name.

    Reads from the ``AGENTCOMPOSE_ADAPTER`` env var if no name is given.
    Defaults to ``pydantic_ai``.
    """
    name = name or os.getenv("AGENTCOMPOSE_ADAPTER", "pydantic_ai")

    if name == "pydantic_ai":
        from .pydantic_ai.adapter import PydanticAIAdapter
        return PydanticAIAdapter()

    raise ValueError(
        f"Unknown adapter: {name!r}. "
        f"Available adapters: pydantic_ai"
    )
