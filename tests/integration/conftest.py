"""Shared helpers for live Pydantic AI integration tests."""

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from agentcompose.operators.pydantic_ai import to_pydantic_tool

MODEL = "anthropic:claude-haiku-4-5-20251001"


def build_agent(primitives, system_prompt: str) -> Agent:
    tools = [to_pydantic_tool(p) for p in primitives]
    return Agent(MODEL, tools=tools, system_prompt=system_prompt)


def run_and_collect(agent: Agent, prompt: str):
    result = agent.run_sync(prompt)
    tool_call_names = [
        part.tool_name
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    return result, tool_call_names
