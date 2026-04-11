"""Live integration tests for the Pydantic AI runtime adapter.

These tests hit the real Anthropic API and cost real money. They're skipped
automatically when ``ANTHROPIC_API_KEY`` is not set, so CI and offline runs
stay green.

What these tests validate that unit tests cannot:

1. The tool docstring is clear enough that a real model *chooses* to call it
   for the right user intent.
2. The model correctly infers arguments from a natural-language prompt.
3. The tool's return value is in a shape the model can use in its final answer.

If any of these fail, it usually means the primitive's docstring or type
hints need to be improved — not that the adapter is broken.
"""

import os

import pytest

from agentcompose.primitives.text import text

from .conftest import build_agent, run_and_collect

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live Anthropic integration tests",
)


class TestLiveWordCount:
    def test_model_picks_word_count_for_counting_intent(self):
        """A real model given only the word_count tool should use it."""
        agent = build_agent(
            [text._primitives["word_count"]],
            "You are a helpful assistant. Use the tools available to "
            "answer the user's question precisely.",
        )

        _, tool_call_names = run_and_collect(
            agent,
            "How many words are in this sentence: 'the quick brown fox jumps'?",
        )

        assert "word_count" in tool_call_names, (
            f"Model did not call word_count. All tool calls: {tool_call_names}"
        )

    def test_model_final_answer_contains_correct_count(self):
        """The tool result should flow back into a plausible final answer."""
        agent = build_agent(
            [text._primitives["word_count"]],
            "You are a helpful assistant. When asked to count words, use "
            "the word_count tool and return just the number.",
        )

        result, _ = run_and_collect(
            agent, "Count the words: 'one two three four five'"
        )

        assert "5" in result.output


class TestLiveChainOfTools:
    def test_model_chains_multiple_tools_when_needed(self):
        """With several text tools, the model should orchestrate them."""
        agent = build_agent(
            [
                text._primitives["to_lower"],
                text._primitives["strip_whitespace"],
                text._primitives["word_count"],
            ],
            "You are a text-processing assistant. Use the available "
            "tools to complete the user's request step by step.",
        )

        result, tool_call_names = run_and_collect(
            agent,
            "Take the text '  HELLO WORLD  ', lowercase it, strip any "
            "surrounding whitespace, then tell me the word count.",
        )

        assert "word_count" in set(tool_call_names)
        assert "2" in result.output
