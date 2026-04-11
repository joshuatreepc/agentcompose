"""Tests for the Pydantic AI runtime adapter.

Two layers of testing:

1. **Shape tests** — verify ``to_pydantic_tool`` unwraps a ``SmartPrimitive``
   into a plain Python callable with the correct name, docstring, and kind
   validation.

2. **Integration tests** — actually instantiate a ``pydantic_ai.Agent`` with
   the adapted tool and run it against a ``TestModel``. TestModel is Pydantic
   AI's built-in fake model that introspects the agent's tools and calls each
   one with generated arguments, so if the adapter produces something unusable
   these tests will fail at agent construction or run time.
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from agentcompose.operators import PrimitiveRegistry
from agentcompose.operators.pydantic_ai import to_pydantic_tool
from agentcompose.primitives.text import text


class TestAdapterShape:
    def test_returns_a_callable(self):
        tool = to_pydantic_tool(text._primitives["word_count"])
        assert callable(tool)

    def test_preserves_function_name(self):
        tool = to_pydantic_tool(text._primitives["word_count"])
        assert tool.__name__ == "word_count"

    def test_preserves_docstring(self):
        tool = to_pydantic_tool(text._primitives["word_count"])
        assert tool.__doc__ is not None
        assert "word" in tool.__doc__.lower()

    def test_preserves_signature(self):
        import inspect

        tool = to_pydantic_tool(text._primitives["word_count"])
        sig = inspect.signature(tool)
        assert list(sig.parameters.keys()) == ["content"]
        assert sig.parameters["content"].annotation is str
        assert sig.return_annotation is int

    def test_unclassified_primitive_is_accepted(self):
        reg = PrimitiveRegistry("tmp_unclassified")

        @reg.register()
        def no_kind(content: str) -> int:
            """Count characters."""
            return len(content)

        tool = to_pydantic_tool(reg._primitives["no_kind"])
        assert tool("hi") == 2

    def test_resource_primitive_is_rejected(self):
        reg = PrimitiveRegistry("tmp_reject")

        @reg.register(kind="resource")
        def fake_resource(uri: str) -> str:
            """Return something from a URI."""
            return "data"

        with pytest.raises(ValueError, match="kind='resource'"):
            to_pydantic_tool(reg._primitives["fake_resource"])

    def test_prompt_primitive_is_rejected(self):
        reg = PrimitiveRegistry("tmp_reject_prompt")

        @reg.register(kind="prompt")
        def fake_prompt(topic: str) -> str:
            """Render a prompt template."""
            return f"Tell me about {topic}"

        with pytest.raises(ValueError, match="kind='prompt'"):
            to_pydantic_tool(reg._primitives["fake_prompt"])


class TestAdapterIntegrationWithTestModel:
    """End-to-end: run a real pydantic_ai.Agent with our adapted tool."""

    def test_agent_constructs_without_error(self):
        tool = to_pydantic_tool(text._primitives["word_count"])
        agent = Agent(TestModel(), tools=[tool])
        # If the adapter produced something Pydantic AI can't introspect,
        # this would raise during construction.
        assert agent is not None

    def test_agent_run_invokes_the_adapted_tool(self):
        tool = to_pydantic_tool(text._primitives["word_count"])
        agent = Agent(TestModel(), tools=[tool])

        result = agent.run_sync("Please count the words.")

        tool_call_names = [
            part.tool_name
            for msg in result.all_messages()
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert "word_count" in tool_call_names

    def test_tool_return_value_flows_back_to_model(self):
        tool = to_pydantic_tool(text._primitives["word_count"])
        agent = Agent(TestModel(), tools=[tool])

        result = agent.run_sync("Count the words.")

        tool_returns = [
            part
            for msg in result.all_messages()
            for part in msg.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert any(tr.tool_name == "word_count" for tr in tool_returns)
        # The actual return value should be an int from word_count(str).
        word_count_return = next(
            tr for tr in tool_returns if tr.tool_name == "word_count"
        )
        assert isinstance(word_count_return.content, int)

    def test_multiple_adapted_tools_work_together(self):
        tools = [
            to_pydantic_tool(text._primitives["to_lower"]),
            to_pydantic_tool(text._primitives["strip_whitespace"]),
            to_pydantic_tool(text._primitives["word_count"]),
        ]
        agent = Agent(TestModel(), tools=tools)

        result = agent.run_sync("Do some text processing.")

        tool_call_names = {
            part.tool_name
            for msg in result.all_messages()
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        }
        # TestModel calls every registered tool at least once.
        assert {"to_lower", "strip_whitespace", "word_count"} <= tool_call_names
