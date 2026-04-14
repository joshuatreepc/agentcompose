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
from typing import Any, Callable, Optional

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
        validate: Optional[Callable[..., Optional[str]]] = None,
        readonly: Optional[bool] = None,
    ):
        """Initialize a SmartPrimitive.

        Args:
            func: The function to wrap.
            name: Optional name for the primitive (defaults to ``func.__name__``).
            kind: MCP-style classification (``'tool'``, ``'resource'``, ``'prompt'``).
                None means unclassified; generators can ignore or default.
            validate: Optional validation function. Called with the same args/kwargs
                before execution. Should return an error message string if invalid,
                or ``None`` if valid.
            readonly: Whether this primitive is free of side effects. ``True`` means
                safe to run in parallel, ``False`` means mutator. ``None`` (default)
                means unknown — the side-effect profile depends on how it's used.
        """
        self.func = func
        self.name = name or func.__name__
        self.kind = kind
        self.validate = validate
        self.readonly = readonly
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        """Apply the primitive or return a configured (partially-applied) version.

        - 0 args: returns a configured function that can be called later.
        - 1+ args: validates (if a validator is set), then invokes the primitive.

        Raises:
            ValueError: If the validation function returns an error message.
        """
        if args:
            if self.validate is not None:
                error = self.validate(*args, **kwargs)
                if error is not None:
                    raise ValueError(
                        f"Validation failed for '{self.name}': {error}"
                    )
            return self.func(*args, **kwargs)

        @wraps(self.func)
        def configured(*a):
            if self.validate is not None:
                error = self.validate(*a, **kwargs)
                if error is not None:
                    raise ValueError(
                        f"Validation failed for '{self.name}': {error}"
                    )
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
        self._context: dict[str, Any] = {}

    # -- Context management ------------------------------------------------

    def configure(self, **ctx: Any) -> "PrimitiveRegistry":
        """Pre-bind infrastructure context for registered functions.

        Registered functions can access these values via the registry's
        ``context`` property. This lets the host inject sandbox managers,
        config objects, callbacks, etc. without polluting the model-facing
        function signature.

        Returns the registry for chaining.

        Example:
            >>> shell.configure(
            ...     sandbox_manager=my_sandbox,
            ...     config=my_config,
            ...     on_output=my_callback,
            ... )
            >>> # Later, inside a registered function:
            >>> # sandbox = shell.context.get("sandbox_manager")
        """
        self._context.update(ctx)
        return self

    @property
    def context(self) -> dict[str, Any]:
        """Read-only access to the pre-bound infrastructure context."""
        return self._context

    def register(
        self,
        name: Optional[str] = None,
        is_conditional: Optional[bool] = None,
        kind: Optional[str] = None,
        validate: Optional[Callable[..., Optional[str]]] = None,
        readonly: Optional[bool] = None,
    ):
        """Decorator to register a function as a SmartPrimitive in this namespace.

        Args:
            name: Optional name for the primitive (defaults to function name).
            is_conditional: Optional flag to mark as conditional. If ``None``,
                auto-detects based on function name patterns.
            kind: MCP-style classification (``'tool'``, ``'resource'``, ``'prompt'``).
            validate: Optional validation function. Called with the same arguments
                as the primitive before execution. Should return an error message
                string if invalid, or ``None`` if valid.
            readonly: Whether this primitive is free of side effects. ``True`` means
                safe to run in parallel, ``False`` means mutator. ``None`` (default)
                means unknown — depends on how the primitive is used.

        Returns:
            Decorator function that wraps the target function as a SmartPrimitive.

        Example:
            >>> files = PrimitiveRegistry("files")
            >>>
            >>> @files.register(kind="tool", readonly=True)
            ... def read_file(path: str) -> str:
            ...     return open(path).read()
            >>>
            >>> @files.register(kind="tool", readonly=False)
            ... def write_file(path: str, content: str) -> None:
            ...     open(path, "w").write(content)
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

            primitive = SmartPrimitive(
                func, primitive_name, kind=normalized_kind,
                validate=validate, readonly=readonly,
            )

            if is_conditional_auto:
                self._conditionals[primitive_name] = primitive
            else:
                self._primitives[primitive_name] = primitive

            setattr(self, primitive_name, primitive)
            return func

        return decorator

    # -- Introspection -----------------------------------------------------

    def list_primitives(
        self,
        kind: Optional[str] = None,
        readonly: Optional[bool] = None,
    ) -> list["SmartPrimitive"]:
        """Return all registered primitives, optionally filtered.

        Args:
            kind: If provided, only return primitives matching this kind
                (``'tool'``, ``'resource'``, ``'prompt'``).
            readonly: If provided, filter by readonly status. ``True`` returns
                only side-effect-free primitives (safe for parallel execution),
                ``False`` returns only mutators.

        Returns:
            List of SmartPrimitive instances.

        Example:
            >>> text.list_primitives()
            [SmartPrimitive('to_lower'), SmartPrimitive('word_count'), ...]
            >>> text.list_primitives(readonly=False)
            []
        """
        all_prims = list(self._primitives.values()) + list(self._conditionals.values())
        if kind is not None:
            all_prims = [p for p in all_prims if p.kind == kind.lower()]
        if readonly is not None:
            all_prims = [p for p in all_prims if p.readonly is readonly]
        return all_prims

    def get(self, name: str) -> Optional["SmartPrimitive"]:
        """Look up a primitive by name, returning None if not found.

        Args:
            name: The registered name of the primitive.

        Returns:
            The SmartPrimitive, or None.
        """
        return self._primitives.get(name) or self._conditionals.get(name)

    def has(self, name: str) -> bool:
        """Check if a primitive with the given name is registered."""
        return name in self._primitives or name in self._conditionals

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self):
        """Iterate over all registered SmartPrimitives."""
        yield from self._primitives.values()
        yield from self._conditionals.values()

    def __len__(self) -> int:
        """Return the total number of registered primitives."""
        return len(self._primitives) + len(self._conditionals)

    def __getattr__(self, name):
        if name in self._primitives:
            return self._primitives[name]
        elif name in self._conditionals:
            return self._conditionals[name]
        else:
            raise AttributeError(f"No primitive '{name}' in {self.namespace_name}")

    def __repr__(self) -> str:
        return f"PrimitiveRegistry({self.namespace_name!r}, {len(self)} primitives)"


__all__ = ["SmartPrimitive", "PrimitiveRegistry"]
