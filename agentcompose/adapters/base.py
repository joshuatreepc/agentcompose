"""Base adapter interface that all adapters must implement."""

from abc import ABC, abstractmethod
from typing import Any


class ToolResolutionError(Exception):
    """Raised when a required tool cannot be resolved.

    Attributes:
        workflow_name: The workflow that declared the dependency.
        dependency: The dependency string that failed to resolve.
        reason: Human-readable explanation of what went wrong.
    """

    def __init__(self, workflow_name: str, dependency: str, reason: str):
        self.workflow_name = workflow_name
        self.dependency = dependency
        self.reason = reason
        super().__init__(
            f"[{workflow_name}] Failed to resolve '{dependency}': {reason}"
        )


class ToolKindError(ToolResolutionError):
    """Raised when a resolved tool has an incompatible kind."""

    def __init__(
        self, workflow_name: str, dependency: str, expected: str, actual: str,
    ):
        self.expected = expected
        self.actual = actual
        super().__init__(
            workflow_name,
            dependency,
            f"expected kind={expected!r}, got kind={actual!r}",
        )


class BaseAdapter(ABC):

    @abstractmethod
    def resolve_tools(self, workflow: Any) -> list[Any]:
        """Resolve a Workflow's requires list into the format the target framework expects.

        Implementations MUST raise ``ToolResolutionError`` if a declared
        dependency cannot be found, rather than silently skipping it.

        Args:
            workflow: A ``Workflow`` instance from ``@compose``.

        Returns:
            A list of tools in whatever format the target framework accepts.

        Raises:
            ToolResolutionError: If a required dependency is missing or invalid.
            ToolKindError: If a dependency exists but has an incompatible kind.
        """
        ...
