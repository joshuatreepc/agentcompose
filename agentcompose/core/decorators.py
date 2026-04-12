"""
agentcompose/core/decorators.py
===============================

Public authoring decorators: `@compose` and `@component`.

This is the stable surface users import against.
"""

import ast
import functools
import inspect
import textwrap
from typing import Any, Callable, Optional, TypedDict, overload


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


def _extract_inputs(func: Callable) -> dict[str, str]:
    """Extract input parameter names and type annotations."""
    hints = func.__annotations__.copy()
    hints.pop("return", None)
    return {k: v.__name__ if hasattr(v, "__name__") else str(v) for k, v in hints.items()}


def _extract_output(func: Callable) -> Optional[str]:
    """Extract the return type annotation."""
    ret = func.__annotations__.get("return")
    if ret is None:
        return None
    return ret.__name__ if hasattr(ret, "__name__") else str(ret)


# ---------------------------------------------------------------------------
# Metadata shapes
# ---------------------------------------------------------------------------

class ComposeMeta(TypedDict):
    kind: str
    name: str
    module: str
    description: str
    inputs: dict[str, str]
    output: Optional[str]
    requires: list[str]


class ComponentMeta(TypedDict):
    kind: str
    name: str
    tags: tuple[str, ...]
    module: str


# ---------------------------------------------------------------------------
# Wrapper classes returned by the decorators
# ---------------------------------------------------------------------------

class Workflow:
    """Wrapper returned by ``@compose``.

    Callable (delegates to the wrapped function), printable (shows the plan),
    and carries typed metadata.
    """

    _agentcompose: ComposeMeta
    __name__: str
    __doc__: Optional[str]
    __annotations__: dict[str, Any]
    __module__: str

    def __init__(self, func: Callable, meta: ComposeMeta):
        functools.update_wrapper(self, func)
        self.func = func
        self._agentcompose = meta
        self._resolved_tools: list[Callable] = self._resolve_tools()

    def _resolve_tools(self) -> list[Any]:
        """Resolve the requires list via the configured adapter."""
        from agentcompose.adapters import get_adapter
        adapter = get_adapter()
        return adapter.resolve_tools(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __iter__(self):
        return iter(self._resolved_tools)

    def show(self) -> None:
        meta = self._agentcompose
        lines = [f"Workflow: {meta['name']}"]
        if meta["description"]:
            lines.append(f"  Description: {meta['description']}")
        if meta["inputs"]:
            inputs_str = ", ".join(f"{k}: {v}" for k, v in meta["inputs"].items())
            lines.append(f"  Inputs: {inputs_str}")
        if meta["output"]:
            lines.append(f"  Output: {meta['output']}")
        if meta["requires"]:
            lines.append("  Requires:")
            for dep in meta["requires"]:
                lines.append(f"    - {dep}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        return f"Workflow({self._agentcompose['name']!r})"


class DecoratedComponent:
    """Wrapper returned by ``@component``.

    Callable (delegates to the wrapped function), printable (shows metadata),
    and carries typed metadata.
    """

    _agentcompose: ComponentMeta
    __name__: str
    __doc__: Optional[str]
    __annotations__: dict[str, Any]
    __module__: str

    def __init__(self, func: Callable, meta: ComponentMeta):
        functools.update_wrapper(self, func)
        self.func = func
        self._agentcompose = meta

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def show(self) -> None:
        meta = self._agentcompose
        lines = [f"Component: {meta['name']}"]
        lines.append(f"  Kind: {meta['kind']}")
        if meta.get("tags"):
            lines.append(f"  Tags: {', '.join(meta['tags'])}")
        lines.append(f"  Module: {meta['module']}")
        if self.func.__doc__:
            lines.append(f"  Description: {self.func.__doc__.strip().splitlines()[0]}")
        inputs = _extract_inputs(self.func)
        if inputs:
            inputs_str = ", ".join(f"{k}: {v}" for k, v in inputs.items())
            lines.append(f"  Inputs: {inputs_str}")
        output = _extract_output(self.func)
        if output:
            lines.append(f"  Output: {output}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        return f"DecoratedComponent({self._agentcompose['name']!r})"


# ---------------------------------------------------------------------------
# Decorator singletons
# ---------------------------------------------------------------------------

class Compose:
    """Decorator class for declaring composed workflows.

    Usage::

        @compose
        def my_workflow(query: str) -> str:
            results = search_docs(query)
            return summarize(results)

        my_workflow.show()   # prints the plan
    """

    def __init__(self):
        self._registered: list[Workflow] = []

    @overload
    def __call__(self, func: Callable, *, name: Optional[str] = None) -> Workflow: ...
    @overload
    def __call__(self, func: None = None, *, name: Optional[str] = None) -> Callable[[Callable], Workflow]: ...

    def __call__(self, func: Optional[Callable] = None, *, name: Optional[str] = None) -> Workflow | Callable[[Callable], Workflow]:
        def decorator(fn: Callable) -> Workflow:
            registry_names: set[str] = set()
            for var_name, var_value in fn.__globals__.items():
                if isinstance(var_value, PrimitiveRegistry):
                    registry_names.add(var_name)

            meta = ComposeMeta(
                kind="compose",
                name=name or fn.__name__,
                module=fn.__module__,
                description=fn.__doc__ or "",
                inputs=_extract_inputs(fn),
                output=_extract_output(fn),
                requires=_extract_dependencies(fn, registry_names),
            )
            workflow = Workflow(fn, meta)
            self._registered.append(workflow)
            return workflow

        return decorator(func) if func is not None else decorator


    def __repr__(self) -> str:
        return f"Compose({len(self._registered)} workflows)"


class Component:
    """Decorator class for tagging functions as agentcompose components.

    Usage::

        @component
        def word_count(content: str) -> int:
            return len(content.split())

        word_count.show()    # prints component metadata
    """

    def __init__(self):
        self._registered: list[DecoratedComponent] = []

    @overload
    def __call__(self, func: Callable, *, kind: str = "tool", name: Optional[str] = None, tags: tuple[str, ...] = ()) -> DecoratedComponent: ...
    @overload
    def __call__(self, func: None = None, *, kind: str = "tool", name: Optional[str] = None, tags: tuple[str, ...] = ()) -> Callable[[Callable], DecoratedComponent]: ...

    def __call__(
        self,
        func: Optional[Callable] = None,
        *,
        kind: str = "tool",
        name: Optional[str] = None,
        tags: tuple[str, ...] = (),
    ) -> DecoratedComponent | Callable[[Callable], DecoratedComponent]:
        def decorator(fn: Callable) -> DecoratedComponent:
            meta = ComponentMeta(
                kind=kind,
                name=name or fn.__name__,
                tags=tags,
                module=fn.__module__,
            )
            dc = DecoratedComponent(fn, meta)
            self._registered.append(dc)
            return dc

        return decorator(func) if func is not None else decorator


    def __repr__(self) -> str:
        return f"Component({len(self._registered)} components)"


compose = Compose()
component = Component()
