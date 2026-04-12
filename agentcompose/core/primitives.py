"""
agentcompose/core/primitives.py
===============================

Core primitive types: `SmartPrimitive` and `PrimitiveRegistry`.

The `compose` and `component` decorators live in `decorators.py` because
composition is a cross-cutting assembly operation, not a property of any
single registry.
"""

import logging
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SmartPrimitive:
    """Wraps a primitive function to enable partial application.

    SmartPrimitive handles both single-argument and multi-argument primitives:
    1. Single: `primitive(x)` or `primitive(param=value)(x)`
    2. Multi: `primitive(x, y)` or `primitive(param=value)(x, y)`

    The behavior is auto-detected based on the number of arguments passed.

    You usually won't use this directly — it's constructed by
    `PrimitiveRegistry.register`.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        kind: Optional[str] = None,
    ):
        """Initialize a SmartPrimitive.

        Args:
            func: The function to wrap.
            name: Optional name for the primitive (defaults to ``func.__name__``).
            kind: MCP-style classification (``'tool'``, ``'resource'``, ``'prompt'``).
                None means unclassified; generators can ignore or default.
        """
        self.func = func
        self.name = name or func.__name__
        self.kind = kind
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        """Apply the primitive or return a configured (partially-applied) version.

        - 0 args: returns a configured function that can be called later.
        - 1+ args: invokes the primitive directly.
        """
        if args:
            return self.func(*args, **kwargs)

        @wraps(self.func)
        def configured(*a):
            return self.func(*a, **kwargs)

        configured.__name__ = (
            f"{self.name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        )
        return configured


class PrimitiveRegistry:
    """Container for organizing related primitives under a namespace.

    ``PrimitiveRegistry`` groups ``SmartPrimitive`` instances under a common
    namespace, making them accessible as attributes.

    Example:
        >>> text = PrimitiveRegistry("text")
        >>>
        >>> @text.register(kind="tool")
        ... def word_count(content: str) -> int:
        ...     return len(content.split())
        >>>
        >>> text.word_count("hello world")
        2
    """

    def __init__(self, namespace_name: str):
        """Initialize a PrimitiveRegistry.

        Args:
            namespace_name: Name for this namespace (used in error messages).
        """
        self.namespace_name = namespace_name
        self._primitives = {}
        self._conditionals = {}

    def register(
        self,
        name: Optional[str] = None,
        is_conditional: Optional[bool] = None,
        kind: Optional[str] = None,
    ):
        """Decorator to register a function as a SmartPrimitive in this namespace.

        Args:
            name: Optional name for the primitive (defaults to function name).
            is_conditional: Optional flag to mark as conditional. If ``None``,
                auto-detects based on function name patterns.
            kind: MCP-style classification (``'tool'``, ``'resource'``, ``'prompt'``).

        Returns:
            Decorator function that wraps the target function as a SmartPrimitive.
        """

        if kind is not None and kind.lower() not in ("tool", "resource", "prompt"):
            raise ValueError(
                f"kind must be one of 'tool', 'resource', or 'prompt'; got {kind!r}"
            )

        def decorator(func: Callable):
            primitive_name = name or func.__name__

            if is_conditional is None:
                conditional_patterns = [
                    "is_",
                    "has_",
                    "needs_",
                    "should_",
                    "can_",
                    "contains_",
                    "matches_",
                    "equals_",
                    "starts_with_",
                    "ends_with_",
                ]
                is_conditional_auto = any(
                    primitive_name.startswith(pattern)
                    for pattern in conditional_patterns
                )
            else:
                is_conditional_auto = is_conditional

            normalized_kind = kind.lower() if kind else None

            if is_conditional_auto:
                self._conditionals[primitive_name] = SmartPrimitive(
                    func, primitive_name, kind=normalized_kind
                )
                setattr(self, primitive_name, self._conditionals[primitive_name])
            else:
                self._primitives[primitive_name] = SmartPrimitive(
                    func, primitive_name, kind=normalized_kind
                )
                setattr(self, primitive_name, self._primitives[primitive_name])

            return func

        return decorator

    def __getattr__(self, name):
        if name in self._primitives:
            return self._primitives[name]
        elif name in self._conditionals:
            return self._conditionals[name]
        else:
            raise AttributeError(f"No primitive '{name}' in {self.namespace_name}")


__all__ = ["SmartPrimitive", "PrimitiveRegistry"]
