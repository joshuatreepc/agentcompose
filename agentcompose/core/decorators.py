"""
agentcompose/core/decorators.py
===============================

Public authoring decorators: `@compose` and `@component`.

This is the stable surface users import against.
"""

import ast
import inspect
import textwrap
from typing import Callable, Optional

from .primitives import PrimitiveRegistry


def _extract_dependencies(func: Callable, registry_names: set[str]) -> list[str]:
    """Scan a function body and extract the names of all functions it calls.

    Walks the AST in source order, collecting:
    - Qualified calls on known registries: ``text.word_count()`` → ``"text.word_count"``
    - Bare function calls: ``redact()`` → ``"redact"``

    Returns a list preserving call order and duplicates.
    """
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except (OSError, TypeError):
        return []

    deps: list[str] = []

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                prefix = node.func.value.id
                if prefix in registry_names:
                    deps.append(f"{prefix}.{node.func.attr}")
            elif isinstance(node.func, ast.Name):
                deps.append(node.func.id)

        for child in ast.iter_child_nodes(node):
            _visit(child)

    func_def = tree.body[0]
    if isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for node in func_def.body:
            _visit(node)

    return deps


def compose(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Declare a function as a composed workflow.

    Scans the function body to extract a dependency list (the contract),
    stamps it as metadata, and returns the function unchanged. The function
    runs as normal Python — no AST compilation or custom execution engine.

    Args:
        func: The workflow function (when used as bare ``@compose``).
        name: Override the registered name; defaults to ``fn.__name__``.

    Example:
        >>> @compose
        ... def my_workflow(query: str):
        ...     results = search_docs(query)
        ...     return summarize(results)
        >>>
        >>> my_workflow._agentcompose["requires"]
        ['search_docs', 'summarize']
    """

    def decorator(fn: Callable) -> Callable:
        registry_names: set[str] = set()
        for var_name, var_value in fn.__globals__.items():
            if isinstance(var_value, PrimitiveRegistry):
                registry_names.add(var_name)

        fn._agentcompose = {  # type: ignore[attr-defined]
            "kind": "compose",
            "name": name or fn.__name__,
            "module": fn.__module__,
            "requires": _extract_dependencies(fn, registry_names),
        }
        return fn

    return decorator(func) if func is not None else decorator



def component(
    func: Optional[Callable] = None,
    *,
    kind: str = "tool",
    name: Optional[str] = None,
    tags: tuple[str, ...] = (),
):
    """Tag a function as an agentcompose component.

    Attaches a small metadata dict at ``fn._agentcompose``. The function itself
    is returned unchanged, so type hints and docstrings remain introspectable
    by pydantic_ai and other tools.

    Args:
        kind: ``"tool"`` (model calls it) or ``"resource"`` (host injects it).
        name: Override the registered name; defaults to ``fn.__name__``.
        tags: Optional string tags for discovery/filtering.
    """

    def decorator(fn: Callable) -> Callable:
        fn._agentcompose = {  # type: ignore[attr-defined]
            "kind": kind,
            "name": name or fn.__name__,
            "tags": tags,
            "module": fn.__module__,
        }
        return fn

    return decorator(func) if func is not None else decorator
