"""
agentcompose.core
==================

Core framework for building composable agent workflows.

- SmartPrimitive: enables partial application of primitives
- PrimitiveRegistry: container for organizing related primitives
- compose: decorator that declares a workflow and extracts its dependency contract
- component: decorator that tags a function as a tool or resource
"""

from .decorators import component, compose
from .primitives import PrimitiveRegistry, SmartPrimitive

__all__ = [
    "SmartPrimitive",
    "PrimitiveRegistry",
    "compose",
    "component",
]

__version__ = "0.1.0"
