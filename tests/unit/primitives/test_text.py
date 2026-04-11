"""Unit tests for agentcompose.primitives.text."""

from agentcompose.primitives.text import text, word_count


class TestWordCount:
    def test_returns_zero_for_empty_string(self):
        assert word_count("") == 0

    def test_counts_single_word(self):
        assert word_count("hello") == 1

    def test_counts_multiple_words(self):
        assert word_count("hello world") == 2

    def test_collapses_repeated_whitespace(self):
        assert word_count("hello   world") == 2

    def test_counts_across_mixed_whitespace(self):
        assert word_count("one\ttwo\nthree four") == 4


class TestTextRegistry:
    def test_word_count_is_accessible_via_registry(self):
        assert text.word_count("hello world") == 2

    def test_word_count_is_registered_as_tool_kind(self):
        primitive = text._primitives["word_count"]
        assert primitive.kind == "tool"

    def test_word_count_preserves_docstring(self):
        primitive = text._primitives["word_count"]
        assert primitive.__doc__ is not None
        assert "word" in primitive.__doc__.lower()

    def test_word_count_name_matches(self):
        primitive = text._primitives["word_count"]
        assert primitive.name == "word_count"
