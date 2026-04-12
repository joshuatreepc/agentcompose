"""Runtime adapter: agentcompose → Pydantic AI.

Pydantic AI expects plain Python functions with type hints and docstrings
as tools. This adapter resolves a Workflow's requires list into those
raw callables.
"""

from typing import Any, Callable

from agentcompose.adapters.base import BaseAdapter
from agentcompose.core.primitives import PrimitiveRegistry, SmartPrimitive


def to_pydantic_tool(primitive: SmartPrimitive) -> Callable:
    """Unwrap a SmartPrimitive into a plain callable for pydantic_ai.

    Validates that the primitive is a tool (not a resource or prompt),
    then returns the underlying function.

    Raises:
        ValueError: If the primitive's kind is not ``"tool"`` or ``None``.
    """
    if primitive.kind not in (None, "tool"):
        raise ValueError(
            f"to_pydantic_tool only accepts kind='tool' "
            f"(or unclassified); got kind={primitive.kind!r} for "
            f"'{primitive.name}'"
        )
    return primitive.func


def to_callable(obj: Any) -> Callable:
    """Unwrap any agentcompose object into a plain callable for pydantic_ai.

    Handles:
    - SmartPrimitive → unwrap via to_pydantic_tool
    - DecoratedComponent → extract .func
    - Plain callable → pass through
    """
    from agentcompose.core.decorators import DecoratedComponent

    if isinstance(obj, SmartPrimitive):
        return to_pydantic_tool(obj)
    if isinstance(obj, DecoratedComponent):
        return obj.func
    if callable(obj):
        return obj

    raise TypeError(f"Cannot convert {type(obj).__name__} to a pydantic_ai tool")


class PydanticAIAdapter(BaseAdapter):

    def resolve_tools(self, workflow: Any) -> list[Callable]:
        """Resolve a Workflow's requires list to plain callables for pydantic_ai.

        Walks the requires list, looks up each dependency in the workflow's
        globals, and converts it to a raw callable via ``to_callable``.
        """
        tools: list[Callable] = []
        func_globals = workflow.func.__globals__

        for dep_name in workflow._agentcompose["requires"]:
            if "." in dep_name:
                # Registry call: e.g. "text.word_count"
                ns_name, method_name = dep_name.split(".", 1)
                ns = func_globals.get(ns_name)
                if isinstance(ns, PrimitiveRegistry):
                    primitive = getattr(ns, method_name, None)
                    if primitive is not None:
                        tools.append(to_callable(primitive))
            else:
                # Bare function call: e.g. "word_count"
                obj = func_globals.get(dep_name)
                if obj is not None:
                    tools.append(to_callable(obj))

        return tools
