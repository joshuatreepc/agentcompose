"""Tests for the @compose decorator.

Verifies that @compose:
1. Stamps correct metadata (kind, name, module, requires)
2. Returns the original function unchanged
3. Extracts the dependency contract from the function body
"""

from agentcompose.core import compose
from agentcompose.primitives.text import text


@compose
def clean_and_count(content: str) -> int:
    """Lowercase, strip whitespace, then count words."""
    content = text.to_lower(content)
    content = text.strip_whitespace(content)
    return text.word_count(content)


@compose
def safe_word_count(content: str) -> int:
    """Return 0 for empty inputs, otherwise lowercase/strip/word-count."""
    if text.is_empty(content):
        return text.constant_zero(content)
    else:
        content = text.to_lower(content)
        content = text.strip_whitespace(content)
        return text.word_count(content)


@compose
def nested_conditional_count(content: str):
    """Nested conditional inside an else branch."""
    if text.is_empty(content):
        return text.constant_zero(content)
    else:
        if text.is_longer_than(content, limit=3):
            content = text.to_lower(content)
            content = text.strip_whitespace(content)
            return text.word_count(content)
        else:
            return text.constant_zero(content)


@compose
def empty_workflow():
    """A compose with no calls."""
    pass


@compose
def calls_bare_function(content: str) -> str:
    """A compose that calls a non-registry function."""
    return content.upper()


class TestComposeMetadata:
    def test_kind_is_compose(self):
        assert clean_and_count._agentcompose["kind"] == "compose"

    def test_name_matches_function_name(self):
        assert clean_and_count._agentcompose["name"] == "clean_and_count"

    def test_module_is_captured(self):
        assert clean_and_count._agentcompose["module"] == __name__

    def test_requires_key_exists(self):
        assert "requires" in clean_and_count._agentcompose

    def test_custom_name(self):
        @compose(name="custom")
        def my_workflow():
            pass

        assert my_workflow._agentcompose["name"] == "custom"


class TestComposePreservesFunction:
    def test_returns_original_function(self):
        def original(x: str) -> str:
            return x

        decorated = compose(original)
        assert decorated is original

    def test_function_remains_callable(self):
        assert clean_and_count("  HELLO WORLD  ") == 2

    def test_docstring_preserved(self):
        assert clean_and_count.__doc__ == "Lowercase, strip whitespace, then count words."

    def test_name_preserved(self):
        assert clean_and_count.__name__ == "clean_and_count"

    def test_type_hints_preserved(self):
        hints = clean_and_count.__annotations__
        assert hints["content"] is str
        assert hints["return"] is int


class TestComposeContractExtraction:
    def test_extracts_sequential_calls(self):
        requires = clean_and_count._agentcompose["requires"]
        assert requires == [
            "text.to_lower",
            "text.strip_whitespace",
            "text.word_count",
        ]

    def test_extracts_calls_from_both_branches(self):
        requires = safe_word_count._agentcompose["requires"]
        assert "text.is_empty" in requires
        assert "text.constant_zero" in requires
        assert "text.to_lower" in requires
        assert "text.word_count" in requires

    def test_extracts_calls_from_nested_conditionals(self):
        requires = nested_conditional_count._agentcompose["requires"]
        assert "text.is_empty" in requires
        assert "text.is_longer_than" in requires
        assert "text.to_lower" in requires
        assert "text.strip_whitespace" in requires
        assert "text.word_count" in requires
        assert requires.count("text.constant_zero") == 2

    def test_empty_body_produces_empty_requires(self):
        assert empty_workflow._agentcompose["requires"] == []

    def test_preserves_call_order(self):
        requires = clean_and_count._agentcompose["requires"]
        assert requires.index("text.to_lower") < requires.index("text.strip_whitespace")
        assert requires.index("text.strip_whitespace") < requires.index("text.word_count")

    def test_condition_functions_included(self):
        requires = safe_word_count._agentcompose["requires"]
        assert requires[0] == "text.is_empty"


class TestComposeExecution:
    def test_sequential_pipeline(self):
        assert clean_and_count("  HELLO WORLD  ") == 2

    def test_conditional_empty_input(self):
        assert safe_word_count("") == 0
        assert safe_word_count("   ") == 0

    def test_conditional_non_empty_input(self):
        assert safe_word_count("  HELLO WORLD  ") == 2

    def test_nested_conditional_empty(self):
        assert nested_conditional_count("") == 0

    def test_nested_conditional_short(self):
        assert nested_conditional_count("hi") == 0

    def test_nested_conditional_long(self):
        assert nested_conditional_count("  HELLO WORLD  ") == 2
