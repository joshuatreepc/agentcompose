"""Pydantic AI runtime adapter for agentcompose primitives.

Takes an agentcompose ``SmartPrimitive`` and returns something that can be
passed to ``pydantic_ai.Agent(tools=[...])``.
"""

from .adapter import to_pydantic_tool

__all__ = ["to_pydantic_tool"]
