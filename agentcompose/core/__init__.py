"""
agentcompose.core
==================

Core framework for building composable agent workflows.

- SmartPrimitive: enables partial application of primitives
- PrimitiveRegistry: container for organizing related primitives
- compose: decorator that declares a workflow and extracts its dependency contract
- component: decorator that tags a function as a tool or resource
"""

from .decorators import (
    Component,
    ComponentMeta,
    Compose,
    ComposeMeta,
    DecoratedComponent,
    Workflow,
    component,
    compose,
)
from .primitives import PrimitiveRegistry, SmartPrimitive

__all__ = [
    "SmartPrimitive",
    "PrimitiveRegistry",
    "compose",
    "component",
    "Compose",
    "Component",
    "Workflow",
    "DecoratedComponent",
    "ComposeMeta",
    "ComponentMeta",
]

__version__ = "0.1.0"
