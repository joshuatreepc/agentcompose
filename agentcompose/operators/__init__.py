"""
agentcompose.operators
======================

Core framework for building composable agent pipelines.

Main components:
- SmartPrimitive: enables partial application of primitives
- PrimitiveRegistry: container for organizing related primitives
- AgentRegistry: registry for agent-facing resources
- PipelineCompiler: compiles declarative syntax into executable pipelines
- StablePipeline: runtime executor for compiled pipelines
"""

from .primitives import AgentRegistry, PrimitiveRegistry, SmartPrimitive

__all__ = [
    "SmartPrimitive",
    "PrimitiveRegistry",
    "AgentRegistry",
]

__version__ = "0.5.0"
