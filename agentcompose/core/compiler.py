"""
agentcompose/core/compiler.py
=============================

Internal AST-to-pipeline compilation machinery.

This module is an implementation detail of the `@compose` decorator. Users
should not import from here directly — the public surface lives in
`decorators.py` and is re-exported from `agentcompose.core`.
"""

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class CompiledStep:
    """A compiled pipeline step."""

    step_type: str
    action: Optional[Callable] = None
    condition: Optional[Callable] = None
    then_branch: Optional[List["CompiledStep"]] = None
    else_branch: Optional[List["CompiledStep"]] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        valid_types = {"transform", "conditional"}

        if self.step_type not in valid_types:
            raise ValueError(
                f"Invalid step_type '{self.step_type}'. Must be one of {valid_types}"
            )

        if self.step_type == "transform":
            if not callable(self.action):
                raise ValueError(
                    f"Transform step requires a callable action, "
                    f"got {type(self.action).__name__}"
                )
            if self.condition is not None:
                logger.warning("Transform step has condition which will be ignored")
            if self.then_branch is not None or self.else_branch is not None:
                logger.warning("Transform step has branches which will be ignored")

        elif self.step_type == "conditional":
            if not callable(self.condition):
                raise ValueError(
                    f"Conditional step requires a callable condition, "
                    f"got {type(self.condition).__name__ if self.condition else 'None'}"
                )
            if not self.then_branch:
                raise ValueError("Conditional step requires at least a then_branch")
            if self.action is not None:
                logger.warning("Conditional step has action which will be ignored")

            for step in self.then_branch:
                if not isinstance(step, CompiledStep):
                    raise TypeError(
                        f"then_branch must contain CompiledStep instances, "
                        f"got {type(step).__name__}"
                    )

            if self.else_branch:
                for step in self.else_branch:
                    if not isinstance(step, CompiledStep):
                        raise TypeError(
                            f"else_branch must contain CompiledStep instances, "
                            f"got {type(step).__name__}"
                        )


class StablePipeline:
    """Runtime pipeline executor."""

    def __init__(self, steps: Optional[List[CompiledStep]] = None, debug: bool = False):
        self.steps = steps or []
        self.debug = debug
        self.__name__ = "pipeline"
        self.__doc__ = "Compiled pipeline"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._validate_pipeline()

    def __call__(self, col: Any) -> Any:
        return self._execute_steps(self.steps, col)

    def _execute_steps(self, steps: List[CompiledStep], col: Any) -> Any:
        result = col

        for step in steps:
            if self.debug:
                step_name = (
                    getattr(step.action, "__name__", step.step_type)
                    if step.action
                    else step.step_type
                )
                self.logger.debug(f"Executing step: {step_name}")

            if step.step_type == "transform":
                if callable(step.action):
                    result = step.action(result)

            elif step.step_type == "conditional":
                if step.then_branch and step.condition is not None:
                    if step.condition(result):
                        result = self._execute_steps(step.then_branch, result)
                    elif step.else_branch:
                        result = self._execute_steps(step.else_branch, result)

        return result

    def _validate_pipeline(self):
        if not self.steps:
            self.logger.debug("Empty pipeline - no steps to validate")
            return

        for i, step in enumerate(self.steps):
            if not isinstance(step, CompiledStep):
                raise TypeError(
                    f"Pipeline step {i} must be a CompiledStep instance, "
                    f"got {type(step).__name__}"
                )

        self.logger.debug(f"Pipeline validated with {len(self.steps)} steps")


class PipelineCompiler:
    def __init__(
        self,
        namespaces: Dict[str, Any],
        debug: bool = False,
        func_globals: Optional[Dict] = None,
    ):
        self.namespaces = namespaces
        self.debug = debug
        self.func_globals = func_globals or {}

    def compile(self, func: Callable) -> StablePipeline:
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
            func_def = tree.body[0]

            if not isinstance(func_def, ast.FunctionDef):
                raise ValueError(f"Expected FunctionDef, got {type(func_def).__name__}")

            steps = self._compile_body(func_def.body)
            pipeline = StablePipeline(steps, self.debug)
            pipeline.__name__ = func.__name__
            pipeline.__doc__ = func.__doc__

            return pipeline

        except Exception as e:
            logger.warning(
                f"Failed to compile '{func.__name__}': {e}. "
                f"Creating empty pipeline as fallback."
            )
            if self.debug:
                logger.debug(f"Compilation error details: {e}", exc_info=True)
            return StablePipeline([], self.debug)

    def _compile_body(self, nodes: Sequence[ast.AST]) -> List[CompiledStep]:
        steps = []

        for node in nodes:
            if isinstance(node, ast.If):
                step = self._compile_if(node)
                if step:
                    steps.append(step)

            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                step = self._compile_call(node.value)
                if step:
                    steps.append(step)

        return steps

    def _compile_if(self, node: ast.If) -> Optional[CompiledStep]:
        condition = self._compile_condition(node.test)
        then_branch = self._compile_body(node.body)
        else_branch = self._compile_body(node.orelse) if node.orelse else None

        try:
            return CompiledStep(
                step_type="conditional",
                condition=condition,
                then_branch=then_branch,
                else_branch=else_branch,
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to compile conditional: {e}")
            if self.debug:
                logger.debug("Conditional compilation error details:", exc_info=True)
            return None

    def _compile_condition(self, node: ast.AST) -> Callable:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            namespace_name = (
                node.func.value.id if isinstance(node.func.value, ast.Name) else None
            )
            method_name = node.func.attr

            namespace = (
                self.namespaces.get(namespace_name) if namespace_name else None
            ) or (self.func_globals.get(namespace_name) if namespace_name else None)
            if namespace and hasattr(namespace, method_name):
                method = getattr(namespace, method_name)

                kwargs = {}
                for keyword in node.keywords:
                    kwargs[keyword.arg] = self._get_value(keyword.value)

                return method(**kwargs) if kwargs else method

        return lambda *_: True

    def _compile_call(self, node: ast.Call) -> Optional[CompiledStep]:
        if isinstance(node.func, ast.Attribute):
            namespace_name = (
                node.func.value.id if isinstance(node.func.value, ast.Name) else None
            )
            method_name = node.func.attr

            namespace = (
                self.namespaces.get(namespace_name) if namespace_name else None
            ) or (self.func_globals.get(namespace_name) if namespace_name else None)
            if namespace and hasattr(namespace, method_name):
                method = getattr(namespace, method_name)

                kwargs = {}
                for keyword in node.keywords:
                    kwargs[keyword.arg] = self._get_value(keyword.value)

                action = method(**kwargs) if kwargs else method

                try:
                    return CompiledStep(step_type="transform", action=action)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to compile transform: {e}")
                    if self.debug:
                        logger.debug(
                            "Transform compilation error details:", exc_info=True
                        )
                    return None

        return None

    def _get_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        try:
            return ast.literal_eval(node)
        except Exception as e:
            logger.debug(f"Failed to extract value from AST node: {e}")
            return None
