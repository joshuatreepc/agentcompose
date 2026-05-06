"""Runtime adapter: agentcompose → Pydantic AI.

Pydantic AI expects plain Python functions with type hints and docstrings
as tools. This adapter resolves a Workflow's requires list into those
raw callables, with full enforcement that every declared dependency
actually resolves to a valid tool.
"""

from typing import Any, Callable

from agentcompose.adapters.base import BaseAdapter, ToolKindError, ToolResolutionError
from agentcompose.core.primitives import PrimitiveRegistry, SmartPrimitive


def to_pydantic_tool(
    primitive: SmartPrimitive,
    *,
    workflow_name: str,
    dep_name: str,
) -> Callable:
    """Unwrap a SmartPrimitive into a plain callable for pydantic_ai.

    Validates that the primitive is a tool (not a resource or prompt),
    then returns the underlying function.

    Raises:
        ToolKindError: If the primitive's kind is not ``"tool"`` or ``None``.
    """
    if primitive.kind not in (None, "tool"):
        raise ToolKindError(
            workflow_name=workflow_name,
            dependency=dep_name,
            expected="tool",
            actual=primitive.kind or "unclassified",
        )
    return primitive.func


def to_callable(
    obj: Any,
    *,
    workflow_name: str,
    dep_name: str,
) -> Callable:
    """Unwrap any agentcompose object into a plain callable for pydantic_ai.

    Handles:
    - SmartPrimitive → unwrap via to_pydantic_tool
    - DecoratedComponent → extract .func
    - Plain callable → pass through

    Raises:
        ToolResolutionError: If the object is not a callable or recognized type.
    """
    from agentcompose.core.decorators import DecoratedComponent

    if isinstance(obj, SmartPrimitive):
        return to_pydantic_tool(
            obj, workflow_name=workflow_name, dep_name=dep_name,
        )
    if isinstance(obj, DecoratedComponent):
        return obj.func
    if callable(obj):
        return obj

    raise ToolResolutionError(
        workflow_name=workflow_name,
        dependency=dep_name,
        reason=f"resolved to {type(obj).__name__}, which is not callable",
    )


class PydanticAIAdapter(BaseAdapter):

    def resolve_tools(self, workflow: Any) -> list[Callable]:
        """Resolve a Workflow's requires list to plain callables for pydantic_ai.

        Walks the requires list, looks up each dependency in the workflow's
        globals, and converts it to a raw callable via ``to_callable``.

        Raises:
            ToolResolutionError: If any declared dependency cannot be found
                or is not a valid tool.
        """
        tools: list[Callable] = []
        func_globals = workflow.func.__globals__
        workflow_name = workflow._agentcompose["name"]

        for dep_name in workflow._agentcompose["requires"]:
            if "." in dep_name:
                # Registry call: e.g. "text.word_count"
                ns_name, method_name = dep_name.split(".", 1)
                ns = func_globals.get(ns_name)

                if ns is None:
                    raise ToolResolutionError(
                        workflow_name=workflow_name,
                        dependency=dep_name,
                        reason=f"namespace '{ns_name}' not found in scope",
                    )

                if not isinstance(ns, PrimitiveRegistry):
                    raise ToolResolutionError(
                        workflow_name=workflow_name,
                        dependency=dep_name,
                        reason=(
                            f"'{ns_name}' is a {type(ns).__name__}, "
                            f"not a PrimitiveRegistry"
                        ),
                    )

                primitive = ns.get(method_name)
                if primitive is None:
                    available = [p.name for p in ns]
                    raise ToolResolutionError(
                        workflow_name=workflow_name,
                        dependency=dep_name,
                        reason=(
                            f"'{method_name}' not registered in "
                            f"'{ns_name}'. Available: {available}"
                        ),
                    )

                tools.append(to_callable(
                    primitive,
                    workflow_name=workflow_name,
                    dep_name=dep_name,
                ))
            else:
                # Bare function call: e.g. "word_count"
                obj = func_globals.get(dep_name)

                if obj is None:
                    raise ToolResolutionError(
                        workflow_name=workflow_name,
                        dependency=dep_name,
                        reason="not found in scope",
                    )

                tools.append(to_callable(
                    obj,
                    workflow_name=workflow_name,
                    dep_name=dep_name,
                ))

        return tools
