"""Tests for the @component decorator."""

from agentcompose.core import component


class TestComponentMetadata:
    def test_bare_decorator_sets_defaults(self):
        @component
        def my_func(x: str) -> str:
            return x

        assert my_func._agentcompose == {
            "kind": "tool",
            "name": "my_func",
            "tags": (),
            "module": __name__,
        }

    def test_explicit_kind(self):
        @component(kind="resource")
        def my_resource() -> str:
            return "data"

        assert my_resource._agentcompose["kind"] == "resource"

    def test_custom_name(self):
        @component(name="custom_name")
        def my_func(x: str) -> str:
            return x

        assert my_func._agentcompose["name"] == "custom_name"

    def test_tags(self):
        @component(tags=("aws", "s3"))
        def get_object(bucket: str, key: str) -> bytes:
            return b""

        assert get_object._agentcompose["tags"] == ("aws", "s3")

    def test_module_is_captured(self):
        @component
        def my_func():
            pass

        assert my_func._agentcompose["module"] == __name__


class TestComponentPreservesFunction:
    def test_function_remains_callable(self):
        @component
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_docstring_preserved(self):
        @component
        def documented(x: str) -> str:
            """This is the docstring."""
            return x

        assert documented.__doc__ == "This is the docstring."

    def test_name_preserved(self):
        @component
        def original_name(x: str) -> str:
            return x

        assert original_name.__name__ == "original_name"

    def test_type_hints_preserved(self):
        @component
        def typed(content: str, limit: int = 5) -> bool:
            return len(content) > limit

        hints = typed.__annotations__
        assert hints["content"] is str
        assert hints["limit"] is int
        assert hints["return"] is bool


class TestComponentWithKwargs:
    def test_all_kwargs_together(self):
        @component(kind="resource", name="s3_bucket", tags=("aws", "storage"))
        def get_bucket() -> dict:
            return {}

        meta = get_bucket._agentcompose
        assert meta["kind"] == "resource"
        assert meta["name"] == "s3_bucket"
        assert meta["tags"] == ("aws", "storage")


class TestComponentWithBranching:
    def test_if_else_in_component(self):
        @component(kind="tool")
        def process_text(content: str) -> str:
            if len(content) > 10:
                return content[:10] + "..."
            else:
                return content

        assert process_text("short") == "short"
        assert process_text("this is a very long string") == "this is a ..."

    def test_branching_with_guard(self):
        @component(kind="tool")
        def safe_divide(a: float, b: float) -> float:
            if b == 0:
                return 0.0
            else:
                return a / b

        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0

    def test_nested_branching(self):
        @component(kind="tool")
        def classify(score: int) -> str:
            if score >= 90:
                return "excellent"
            elif score >= 70:
                return "good"
            elif score >= 50:
                return "fair"
            else:
                return "poor"

        assert classify(95) == "excellent"
        assert classify(75) == "good"
        assert classify(55) == "fair"
        assert classify(30) == "poor"

    def test_branching_calling_other_functions(self):
        def redact(text: str) -> str:
            return "***"

        def passthrough(text: str) -> str:
            return text

        @component(kind="tool")
        def maybe_redact(content: str, sensitive: bool = False) -> str:
            if sensitive:
                return redact(content)
            else:
                return passthrough(content)

        assert maybe_redact("hello", sensitive=True) == "***"
        assert maybe_redact("hello", sensitive=False) == "hello"

    def test_metadata_still_set_with_branching(self):
        @component(kind="tool", name="branchy", tags=("guard",))
        def branchy(x: int) -> str:
            if x > 0:
                return "positive"
            else:
                return "non-positive"

        assert branchy._agentcompose["kind"] == "tool"
        assert branchy._agentcompose["name"] == "branchy"
        assert branchy._agentcompose["tags"] == ("guard",)
