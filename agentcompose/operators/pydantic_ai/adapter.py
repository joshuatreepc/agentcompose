"""Runtime adapter: SmartPrimitive -> Pydantic AI tool callable.

Pydantic AI accepts plain Python functions with type hints and docstrings as
tools. It introspects the signature to build the JSON schema the LLM sees, and
uses the docstring as the tool description. Our ``SmartPrimitive`` wraps the
underlying function plus some metadata (``name``, ``kind``, docstring), so the
adapter's job is mostly to unwrap the primitive and validate that the caller is
asking for the right ``kind`` — the actual signature/docstring introspection is
Pydantic AI's problem.
"""

from typing import Callable

from agentcompose.operators.primitives import SmartPrimitive


def to_pydantic_tool(primitive: SmartPrimitive) -> Callable:
    """Convert an agentcompose primitive into a Pydantic AI tool callable.

    Args:
        primitive: A ``SmartPrimitive`` with ``kind="tool"`` (or unclassified).

    Returns:
        The underlying Python function, ready to be passed to
        ``pydantic_ai.Agent(tools=[...])``.

    Raises:
        ValueError: If the primitive's ``kind`` is not ``"tool"`` or ``None``.
    """
    if primitive.kind not in (None, "tool"):
        raise ValueError(
            f"to_pydantic_tool only accepts primitives with kind='tool' "
            f"(or unclassified); got kind={primitive.kind!r} for "
            f"'{primitive.name}'"
        )
    return primitive.func
