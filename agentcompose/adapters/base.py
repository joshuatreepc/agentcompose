"""Base adapter interface that all adapters must implement."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):

    @abstractmethod
    def resolve_tools(self, workflow: Any) -> list[Any]:
        """Resolve a Workflow's requires list into the format the target framework expects.

        Args:
            workflow: A ``Workflow`` instance from ``@compose``.

        Returns:
            A list of tools in whatever format the target framework accepts.
        """
        ...
