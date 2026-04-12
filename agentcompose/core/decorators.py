"""
agentcompose/core/decorators.py
===============================

Public authoring decorators: `@compose` and `@component`.

This is the stable surface users import against. The compilation machinery
lives in `compiler.py` and is an implementation detail.
"""

import ast
import inspect
import logging
import textwrap
from typing import Any, Callable, Dict, Optional

from .compiler import PipelineCompiler
from .primitives import PrimitiveRegistry

logger = logging.getLogger(__name__)


def compose(
    func: Optional[Callable] = None,
    *,
    debug: bool = False,
    steps: Optional[list] = None,
    **namespaces,
):
    """Decorator that converts a function body into a composed pipeline.

    Analyzes the AST of the decorated function and extracts a sequence of
    primitive calls, creating a pipeline that applies them in order.

    Args:
        func: Function to convert into a pipeline (when used as `@compose`).
        debug: If True, logs each step as it is applied.
        steps: Optional list of pre-configured steps (bypasses AST parsing).
        **namespaces: Explicit namespace overrides; any PrimitiveRegistry
            instances in the function's globals are auto-detected.

    Returns:
        A compiled pipeline callable.

    Example:
        >>> from agentcompose.core import PrimitiveRegistry, compose
        >>>
        >>> text = PrimitiveRegistry("text")
        >>>
        >>> @text.register()
        ... def lower(s): return s.lower()
        >>>
        >>> @text.register()
        ... def strip_ws(s): return s.strip()
        >>>
        >>> @compose
        ... def clean():
        ...     text.lower()
        ...     text.strip_ws()
    """

    def decorator(fn: Callable):
        if steps is not None:
            def pipeline(col: Any) -> Any:
                result = col
                for step in steps:  # type: ignore[union-attr]
                    result = step(result)
                return result

            pipeline.__name__ = fn.__name__
            pipeline.__doc__ = fn.__doc__
            return pipeline

        for var_name, var_value in fn.__globals__.items():
            if isinstance(var_value, PrimitiveRegistry):
                if var_name not in namespaces:
                    namespaces[var_name] = var_value

        try:
            compiler = PipelineCompiler(namespaces, debug, fn.__globals__)
            pipeline = compiler.compile(fn)

            if debug and pipeline.steps:
                logger.info(
                    f"Successfully compiled '{fn.__name__}' with {len(pipeline.steps)} steps"
                )

            return pipeline
        except Exception as e:
            logger.warning(
                f"Advanced compilation failed for '{fn.__name__}': {e}. "
                f"Falling back to sequential extraction."
            )
            if debug:
                logger.debug("Compilation error details:", exc_info=True)

            return _fallback_compose(fn, namespaces, debug)

    if func is None:
        return decorator
    return decorator(func)


def component(
    func: Optional[Callable] = None,
    *,
    kind: str = "tool",
    name: Optional[str] = None,
    tags: tuple[str, ...] = (),
):
    """Tag a function as an agentcompose component.

    Attaches a small metadata dict at `fn._agentcompose`. The function itself
    is returned unchanged, so type hints and docstrings remain introspectable
    by pydantic_ai and other tools.

    Args:
        kind: "tool" (model calls it) or "resource" (host injects it).
        name: Override the registered name; defaults to `fn.__name__`.
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


def _fallback_compose(func: Callable, namespaces: Dict, debug: bool) -> Callable:
    """Fallback for when compilation fails - extracts sequential calls only."""
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        func_def = tree.body[0]

        steps = []
        if isinstance(func_def, ast.FunctionDef):
            for node in func_def.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        namespace_name = (
                            node.value.func.value.id
                            if isinstance(node.value.func.value, ast.Name)
                            else None
                        )
                        method_name = node.value.func.attr
                        namespace = (
                            namespaces.get(namespace_name) if namespace_name else None
                        ) or (
                            func.__globals__.get(namespace_name)
                            if namespace_name
                            else None
                        )
                        if namespace and hasattr(namespace, method_name):
                            method = getattr(namespace, method_name)

                            kwargs = {}
                            for keyword in node.value.keywords:
                                try:
                                    kwargs[keyword.arg] = ast.literal_eval(
                                        keyword.value
                                    )
                                except Exception:
                                    pass

                            steps.append(method(**kwargs) if kwargs else method)

        def pipeline(col: Any) -> Any:
            result = col
            for step in steps:
                if debug:
                    logger.debug(f"Executing step: {getattr(step, '__name__', step)}")
                result = step(result)
            return result

        pipeline.__name__ = func.__name__
        pipeline.__doc__ = func.__doc__
        return pipeline

    except Exception as e:
        logger.error(
            f"Failed to create pipeline for '{func.__name__}': {e}. "
            f"Returning identity function."
        )

        def pipeline(col: Any) -> Any:
            return col

        pipeline.__name__ = func.__name__
        pipeline.__doc__ = f"Failed to compile {func.__name__}"
        return pipeline
