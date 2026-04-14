"""
Todo list management primitive.

Ported from the Google ADK TypeScript WriteTodosTool.
Provides a simple todo list that the model can manage to track
multi-step task progress. The model writes the full list each time,
replacing any previous state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agentcompose.core import PrimitiveRegistry

write_todos = PrimitiveRegistry("write_todos")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled", "blocked"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class Todo:
    """A single todo item."""

    description: str
    status: str = "pending"


@dataclass
class ToolResult:
    """Result formatted for LLM and human display."""

    llm_content: str
    return_display: str = ""
    todos: list[Todo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_todos(todos: list[dict]) -> Optional[str]:
    """Validate a list of todo dicts.

    Returns an error message string, or None if valid.
    """
    if not isinstance(todos, list):
        return "`todos` parameter must be a list"

    for todo in todos:
        if not isinstance(todo, dict):
            return "Each todo item must be a dict"
        desc = todo.get("description", "")
        if not isinstance(desc, str) or not desc.strip():
            return "Each todo must have a non-empty description string"
        status = todo.get("status", "pending")
        if status not in VALID_STATUSES:
            return (
                f"Each todo must have a valid status "
                f"({', '.join(sorted(VALID_STATUSES))})"
            )

    in_progress_count = sum(
        1 for t in todos if t.get("status") == "in_progress"
    )
    if in_progress_count > 1:
        return 'Only one task can be "in_progress" at a time.'

    return None


# ---------------------------------------------------------------------------
# WriteTodosInvocation
# ---------------------------------------------------------------------------


class WriteTodosInvocation:
    """Manages a single write_todos operation.

    Validates the todo list, formats it for the LLM, and stores the
    current state. The full list is overwritten each time.
    """

    def __init__(self, todos: list[dict]):
        self.raw_todos = todos

    def get_description(self) -> str:
        count = len(self.raw_todos)
        if count == 0:
            return "Cleared todo list"
        return f"Set {count} todo(s)"

    def validate(self) -> Optional[str]:
        return validate_todos(self.raw_todos)

    def execute(self) -> ToolResult:
        error = self.validate()
        if error:
            return ToolResult(
                llm_content=f"Error: {error}",
                return_display=error,
            )

        todos = [
            Todo(
                description=t["description"],
                status=t.get("status", "pending"),
            )
            for t in self.raw_todos
        ]

        if not todos:
            return ToolResult(
                llm_content="Successfully cleared the todo list.",
                return_display="Todo list cleared.",
                todos=[],
            )

        todo_str = "\n".join(
            f"{i}. [{t.status}] {t.description}"
            for i, t in enumerate(todos, 1)
        )

        return ToolResult(
            llm_content=(
                f"Successfully updated the todo list. "
                f"The current list is now:\n{todo_str}"
            ),
            return_display=f"{len(todos)} todo(s) set.",
            todos=todos,
        )


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@write_todos.register(kind="tool", readonly=False)
def execute(todos: list[dict]) -> ToolResult:
    """Set the full todo list, replacing any previous state.

    Each item must have a 'description' (string) and optionally a 'status'
    (one of: pending, in_progress, completed, cancelled, blocked).
    Only one item may be 'in_progress' at a time.

    Args:
        todos: List of todo items, each a dict with 'description' and 'status'.

    Returns:
        The formatted todo list, or an error if validation fails.
    """
    invocation = WriteTodosInvocation(todos)
    return invocation.execute()
