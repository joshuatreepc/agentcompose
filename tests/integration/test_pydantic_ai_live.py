"""Live integration tests for the Pydantic AI runtime adapter.

These tests hit the real OpenAI API and cost real money (fractions of a cent
per run on ``gpt-4o-mini``). They're skipped automatically when
``OPENAI_API_KEY`` is not set, so CI and offline runs stay green.

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
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from agentcompose.operators.pydantic_ai import to_pydantic_tool
from agentcompose.primitives.text import text

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live OpenAI integration tests",
)

MODEL = "openai:gpt-4o-mini"


class TestLiveWordCount:
    def test_model_picks_word_count_for_counting_intent(self):
        """A real model given only the word_count tool should use it."""
        tool = to_pydantic_tool(text._primitives["word_count"])
        agent = Agent(
            MODEL,
            tools=[tool],
            system_prompt=(
                "You are a helpful assistant. Use the tools available to "
                "answer the user's question precisely."
            ),
        )

        result = agent.run_sync(
            "How many words are in this sentence: 'the quick brown fox jumps'?"
        )

        tool_calls = [
            part
            for msg in result.all_messages()
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert any(tc.tool_name == "word_count" for tc in tool_calls), (
            "Model did not call word_count. All tool calls: "
            f"{[tc.tool_name for tc in tool_calls]}"
        )

    def test_model_final_answer_contains_correct_count(self):
        """The tool result should flow back into a plausible final answer."""
        tool = to_pydantic_tool(text._primitives["word_count"])
        agent = Agent(
            MODEL,
            tools=[tool],
            system_prompt=(
                "You are a helpful assistant. When asked to count words, use "
                "the word_count tool and return just the number."
            ),
        )

        result = agent.run_sync(
            "Count the words: 'one two three four five'"
        )

        # gpt-4o-mini is reliable enough that "5" will appear in the output
        # for this prompt, but we keep the assertion loose in case it pads.
        assert "5" in result.output


class TestLiveChainOfTools:
    def test_model_chains_multiple_tools_when_needed(self):
        """With several text tools, the model should orchestrate them."""
        tools = [
            to_pydantic_tool(text._primitives["to_lower"]),
            to_pydantic_tool(text._primitives["strip_whitespace"]),
            to_pydantic_tool(text._primitives["word_count"]),
        ]
        agent = Agent(
            MODEL,
            tools=tools,
            system_prompt=(
                "You are a text-processing assistant. Use the available "
                "tools to complete the user's request step by step."
            ),
        )

        result = agent.run_sync(
            "Take the text '  HELLO WORLD  ', lowercase it, strip any "
            "surrounding whitespace, then tell me the word count."
        )

        tool_call_names = {
            part.tool_name
            for msg in result.all_messages()
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        }
        # Don't require strict ordering — just that word_count was reached.
        assert "word_count" in tool_call_names
        assert "2" in result.output
