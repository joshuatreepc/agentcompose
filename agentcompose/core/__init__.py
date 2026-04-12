"""
agentcompose.core
==================

Core framework for building composable agent pipelines.

Main components:
- SmartPrimitive: enables partial application of primitives
- PrimitiveRegistry: container for organizing related primitives
- compose: decorator that turns a function body into an executable pipeline
- component: decorator that tags a function as a tool or resource
"""

from .compiler import CompiledStep, PipelineCompiler, StablePipeline
from .decorators import component, compose
from .primitives import PrimitiveRegistry, SmartPrimitive

__all__ = [
    "SmartPrimitive",
    "PrimitiveRegistry",
    "compose",
    "component",
    "CompiledStep",
    "PipelineCompiler",
    "StablePipeline",
]

__version__ = "0.5.0"
