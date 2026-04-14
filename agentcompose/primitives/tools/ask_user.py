"""
Ask-user primitive for human-in-the-loop agent interactions.

Ported from the Google ADK TypeScript AskUserTool/AskUserInvocation.
Provides a structured way for the model to ask the user questions:
- Free-text questions
- Multiple-choice questions with 2-4 options
- Validation of question structure and option counts
- Pluggable input handler via registry context
- Formatted results for both LLM and human display
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

ask_user = PrimitiveRegistry("ask_user")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    """Types of questions the model can ask."""

    FREE_TEXT = "free_text"
    CHOICE = "choice"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Option:
    """A selectable option for a choice question."""

    label: str
    description: str = ""


@dataclass
class Question:
    """A single question to present to the user."""

    question: str
    type: str = "free_text"
    header: Optional[str] = None
    options: Optional[list[Option]] = None


@dataclass
class ToolResult:
    """Result formatted for LLM and human display."""

    llm_content: str
    return_display: str = ""
    dismissed: bool = False
    answers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


InputHandler = Callable[[list[Question]], Optional[dict[str, str]]]
"""Takes a list of Questions, returns {question_index: answer} or None if dismissed."""


@runtime_checkable
class AsyncInputHandler(Protocol):
    """Async protocol for input handlers (CLI prompts, web dialogs, etc.)."""

    async def ask(self, questions: list[Question]) -> Optional[dict[str, str]]:
        """Present questions and return answers, or None if dismissed."""
        ...


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_questions(questions: list[dict]) -> Optional[str]:
    """Validate a list of question dicts.

    Returns an error message string, or None if valid.
    """
    if not isinstance(questions, list) or len(questions) == 0:
        return "At least one question is required."

    for i, q in enumerate(questions, 1):
        if not isinstance(q, dict):
            return f"Question {i}: must be a dict."

        question_type = q.get("type", "free_text")

        # Choice questions need options
        if question_type == QuestionType.CHOICE:
            options = q.get("options")
            if not options or len(options) < 2:
                return (
                    f"Question {i}: type='choice' requires 'options' "
                    f"array with 2-4 items."
                )
            if len(options) > 4:
                return (
                    f"Question {i}: 'options' array must have at most 4 items."
                )

        # Validate option structure if present
        options = q.get("options")
        if options:
            for j, opt in enumerate(options, 1):
                if not isinstance(opt, dict):
                    return f"Question {i}, option {j}: must be a dict."
                label = opt.get("label", "")
                if not isinstance(label, str) or not label.strip():
                    return (
                        f"Question {i}, option {j}: 'label' is required "
                        f"and must be a non-empty string."
                    )
                desc = opt.get("description")
                if desc is not None and not isinstance(desc, str):
                    return (
                        f"Question {i}, option {j}: 'description' must be a string."
                    )

    return None


def _parse_questions(raw: list[dict]) -> list[Question]:
    """Convert raw question dicts to Question objects."""
    questions: list[Question] = []
    for q in raw:
        options = None
        if q.get("options"):
            options = [
                Option(
                    label=o.get("label", ""),
                    description=o.get("description", ""),
                )
                for o in q["options"]
            ]
        questions.append(Question(
            question=q.get("question", ""),
            type=q.get("type", "free_text"),
            header=q.get("header"),
            options=options,
        ))
    return questions


# ---------------------------------------------------------------------------
# Default input handler (stdin)
# ---------------------------------------------------------------------------


def default_input_handler(questions: list[Question]) -> Optional[dict[str, str]]:
    """Simple stdin-based input handler for CLI usage."""
    answers: dict[str, str] = {}
    for i, q in enumerate(questions):
        print()
        header = q.header or f"Question {i + 1}"
        print(f"  {header}: {q.question}")

        if q.type == QuestionType.CHOICE and q.options:
            for j, opt in enumerate(q.options, 1):
                desc = f" - {opt.description}" if opt.description else ""
                print(f"    {j}. {opt.label}{desc}")
            try:
                choice = input("  Your choice (number): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(q.options):
                    answers[str(i)] = q.options[idx].label
                else:
                    answers[str(i)] = choice
            except (ValueError, EOFError, KeyboardInterrupt):
                return None
        else:
            try:
                answer = input("  Your answer: ").strip()
                answers[str(i)] = answer
            except (EOFError, KeyboardInterrupt):
                return None

    return answers


# ---------------------------------------------------------------------------
# AskUserInvocation
# ---------------------------------------------------------------------------


class AskUserInvocation:
    """Manages a single ask_user interaction.

    Mirrors the TypeScript AskUserInvocation. Handles:
    - Question validation
    - Presenting questions via a pluggable input handler
    - Formatting answers for LLM consumption
    - Handling dismissal/cancellation
    """

    def __init__(
        self,
        questions: list[dict],
        input_handler: Optional[InputHandler] = None,
    ):
        self.raw_questions = questions
        self.input_handler = input_handler or default_input_handler

    def get_description(self) -> str:
        parsed = _parse_questions(self.raw_questions)
        return f"Asking user: {', '.join(q.question for q in parsed)}"

    def validate(self) -> Optional[str]:
        return validate_questions(self.raw_questions)

    def execute(self) -> ToolResult:
        # Validate
        error = self.validate()
        if error:
            return ToolResult(
                llm_content=f"Error: {error}",
                return_display=error,
            )

        questions = _parse_questions(self.raw_questions)

        # Get answers from the input handler
        answers = self.input_handler(questions)

        # User dismissed
        if answers is None:
            return ToolResult(
                llm_content="User dismissed ask_user dialog without answering.",
                return_display="User dismissed dialog.",
                dismissed=True,
            )

        has_answers = len(answers) > 0

        # Format display
        if has_answers:
            lines: list[str] = ["**User answered:**"]
            for idx_str, answer in answers.items():
                idx = int(idx_str)
                q = questions[idx] if idx < len(questions) else None
                category = q.header if q and q.header else f"Q{idx_str}"
                prefix = f"  {category} → "
                indent = " " * len(prefix)
                answer_lines = answer.split("\n")
                lines.append(prefix + ("\n" + indent).join(answer_lines))
            return_display = "\n".join(lines)
        else:
            return_display = "User submitted without answering questions."

        return ToolResult(
            llm_content=json.dumps({"answers": answers}),
            return_display=return_display,
            answers=answers,
        )


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@ask_user.register(kind="tool", readonly=True)
def execute(questions: list[dict]) -> ToolResult:
    """Ask the user one or more questions and return their answers.

    Each question is a dict with:
    - 'question' (str): The question text.
    - 'type' (str): 'free_text' or 'choice'. Defaults to 'free_text'.
    - 'header' (str, optional): Category label for display.
    - 'options' (list, optional): For 'choice' type, 2-4 items each with 'label' and 'description'.

    Args:
        questions: List of question dicts to present to the user.

    Returns:
        The user's answers as JSON, or a dismissal message.
    """
    ctx = ask_user.context
    invocation = AskUserInvocation(
        questions=questions,
        input_handler=ctx.get("input_handler"),
    )
    return invocation.execute()
