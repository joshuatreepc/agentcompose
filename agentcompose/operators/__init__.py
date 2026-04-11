"""
agentcompose.operators
======================

Core framework for building composable agent pipelines.

Main components:
- SmartPrimitive: enables partial application of primitives
- PrimitiveRegistry: container for organizing related primitives
- PipelineCompiler: compiles declarative syntax into executable pipelines
- StablePipeline: runtime executor for compiled pipelines
"""

from .compose import CompiledStep, PipelineCompiler, StablePipeline, compose
from .primitives import PrimitiveRegistry, SmartPrimitive

__all__ = [
    "SmartPrimitive",
    "PrimitiveRegistry",
    "compose",
    "CompiledStep",
    "PipelineCompiler",
    "StablePipeline",
]

__version__ = "0.5.0"
