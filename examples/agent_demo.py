"""Demo: @compose + @component + pydantic_ai agent."""

import os
import sys

from pydantic_ai import Agent

from agentcompose.core import component, compose
from dotenv import load_dotenv
MODEL = "anthropic:claude-haiku-4-5-20251001"

from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")


# --- Components (tools the agent can call) ---

@component(kind="tool")
def word_count(content: str) -> int:
    """Count the number of words in the given text."""
    return len(content.split())


@component(kind="tool")
def to_lower(content: str) -> str:
    """Convert text to lowercase."""
    return content.lower()


@component(kind="tool")
def strip_whitespace(content: str) -> str:
    """Remove leading and trailing whitespace from text."""
    return content.strip()


# --- Compose a workflow ---

@compose
def text_processing_agent(content: str) -> int:
    """Clean up text and count words.

    Lowercase the input, strip whitespace, then count words.
    """
    content = to_lower(content)
    content = strip_whitespace(content)
    return word_count(content)


# --- Show the plan ---

print("=== Workflow Plan ===")
text_processing_agent.show()
print()

print("=== Components ===")
to_lower.show()
print()
word_count.show()
print()
strip_whitespace.show()
print()

# --- Run with pydantic_ai ---

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not set, skipping live agent run.")
    sys.exit(0)

print("=== Running Agent ===")

agent = Agent(
    MODEL,
    tools=text_processing_agent,
    system_prompt=(
        "You are a text-processing assistant. Use the available "
        "tools to complete the user's request step by step."
    ),
)

result = agent.run_sync(
    "Take the text '  HELLO WORLD  ', lowercase it, strip whitespace, "
    "then tell me the word count."
)

print(f"Agent response: {result.output}")
