"""Smoke tests for the @compose decorator chaining primitives end-to-end.

These tests exercise the current chain-based semantics of @compose: each step
receives the previous step's output as its single argument. They do NOT involve
an LLM or any agent framework — they verify that the AST compiler and
StablePipeline correctly thread a value through a sequence of primitives.

When @compose moves to collector semantics (producing an agent spec rather
than a runnable chain), these tests will need to be rewritten or replaced.
"""

from agentcompose.core import StablePipeline, compose
from agentcompose.primitives.text import text


@compose
def clean_and_count():
    """Lowercase, strip whitespace, then count words."""
    text.to_lower()
    text.strip_whitespace()
    text.word_count()


@compose
def safe_word_count():
    """Return 0 for empty inputs, otherwise lowercase/strip/word-count."""
    if text.is_empty():
        text.constant_zero()
    else:
        text.to_lower()
        text.strip_whitespace()
        text.word_count()


@compose
def count_if_non_empty():
    """Conditional with no else branch — counts words only if non-empty."""
    if text.is_empty():
        text.constant_zero()


@compose
def count_if_long_enough():
    """Condition with an explicit kwarg: only count if longer than 3 chars."""
    if text.is_longer_than(limit=3):
        text.word_count()
    else:
        text.constant_zero()


@compose
def nested_conditional_count():
    """Nested conditional inside an else branch.

    - Empty input: return 0.
    - Short input (<= 3 chars): return 0.
    - Long input: lowercase/strip/word-count.
    """
    if text.is_empty():
        text.constant_zero()
    else:
        if text.is_longer_than(limit=3):
            text.to_lower()
            text.strip_whitespace()
            text.word_count()
        else:
            text.constant_zero()


class TestComposePipelineShape:
    def test_returns_a_stable_pipeline(self):
        assert isinstance(clean_and_count, StablePipeline)

    def test_pipeline_has_three_steps(self):
        assert len(clean_and_count.steps) == 3

    def test_pipeline_preserves_source_function_name(self):
        assert clean_and_count.__name__ == "clean_and_count"

    def test_pipeline_preserves_source_docstring(self):
        assert clean_and_count.__doc__ is not None
        assert "count words" in clean_and_count.__doc__

    def test_every_step_is_a_transform(self):
        for step in clean_and_count.steps:
            assert step.step_type == "transform"


class TestComposePipelineExecution:
    def test_threads_value_through_all_steps(self):
        assert clean_and_count("  HELLO WORLD  ") == 2

    def test_handles_empty_string(self):
        assert clean_and_count("") == 0

    def test_handles_already_clean_input(self):
        assert clean_and_count("hello world foo") == 3

    def test_handles_mixed_case_and_padding(self):
        assert clean_and_count("   One Two THREE four   ") == 4

    def test_collapses_internal_whitespace_via_split(self):
        # strip_whitespace only trims edges; word_count's split() handles the rest
        assert clean_and_count("  hello   world  ") == 2


class TestComposeConditionalShape:
    def test_pipeline_has_single_conditional_step(self):
        assert len(safe_word_count.steps) == 1
        assert safe_word_count.steps[0].step_type == "conditional"

    def test_conditional_has_then_branch_with_one_step(self):
        conditional = safe_word_count.steps[0]
        assert conditional.then_branch is not None
        assert len(conditional.then_branch) == 1
        assert conditional.then_branch[0].step_type == "transform"

    def test_conditional_has_else_branch_with_three_steps(self):
        conditional = safe_word_count.steps[0]
        assert conditional.else_branch is not None
        assert len(conditional.else_branch) == 3
        for step in conditional.else_branch:
            assert step.step_type == "transform"

    def test_conditional_has_callable_condition(self):
        conditional = safe_word_count.steps[0]
        assert callable(conditional.condition)


class TestComposeConditionalExecution:
    def test_then_branch_runs_for_empty_string(self):
        assert safe_word_count("") == 0

    def test_then_branch_runs_for_whitespace_only(self):
        assert safe_word_count("   \t\n  ") == 0

    def test_else_branch_runs_for_non_empty_string(self):
        assert safe_word_count("  HELLO WORLD  ") == 2

    def test_else_branch_runs_for_single_word(self):
        assert safe_word_count("lonely") == 1

    def test_else_branch_handles_clean_input(self):
        assert safe_word_count("one two three four") == 4


class TestComposeConditionalNoElse:
    def test_pipeline_has_single_conditional_step(self):
        assert len(count_if_non_empty.steps) == 1
        assert count_if_non_empty.steps[0].step_type == "conditional"

    def test_then_branch_has_one_step(self):
        conditional = count_if_non_empty.steps[0]
        assert conditional.then_branch is not None
        assert len(conditional.then_branch) == 1

    def test_else_branch_is_absent(self):
        conditional = count_if_non_empty.steps[0]
        assert not conditional.else_branch

    def test_then_branch_runs_on_empty_input(self):
        assert count_if_non_empty("") == 0

    def test_missing_else_falls_through_unchanged(self):
        # With no else branch, the original input is returned unchanged.
        assert count_if_non_empty("hello world") == "hello world"


class TestComposeConditionalWithKwarg:
    def test_pipeline_compiles(self):
        assert len(count_if_long_enough.steps) == 1
        assert count_if_long_enough.steps[0].step_type == "conditional"

    def test_condition_callable_honors_kwarg(self):
        conditional = count_if_long_enough.steps[0]
        # is_longer_than(limit=3): "hi" is length 2, 2 > 3 is False
        assert conditional.condition("hi") is False
        # "hello" is length 5, 5 > 3 is True
        assert conditional.condition("hello") is True

    def test_short_input_hits_else_branch(self):
        assert count_if_long_enough("hi") == 0

    def test_long_input_hits_then_branch(self):
        assert count_if_long_enough("hello world") == 2


class TestComposeNestedConditional:
    def test_pipeline_compiles_with_nested_structure(self):
        outer = nested_conditional_count.steps[0]
        assert outer.step_type == "conditional"
        assert outer.else_branch is not None
        # The else branch should contain a nested conditional step.
        assert len(outer.else_branch) == 1
        inner = outer.else_branch[0]
        assert inner.step_type == "conditional"
        assert inner.then_branch is not None
        assert inner.else_branch is not None
        assert len(inner.then_branch) == 3
        assert len(inner.else_branch) == 1

    def test_empty_hits_outer_then_branch(self):
        assert nested_conditional_count("") == 0

    def test_short_hits_inner_else_branch(self):
        assert nested_conditional_count("hi") == 0

    def test_long_hits_inner_then_branch(self):
        assert nested_conditional_count("  HELLO WORLD  ") == 2
