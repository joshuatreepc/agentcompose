"""Unit tests for the write_todos primitive."""

from agentcompose.primitives.tools.write_todos import (
    Todo,
    TodoStatus,
    VALID_STATUSES,
    WriteTodosInvocation,
    validate_todos,
    write_todos,
)


class TestValidateTodos:
    def test_returns_none_for_valid_list(self):
        assert validate_todos([{"description": "a", "status": "pending"}]) is None

    def test_rejects_non_list(self):
        assert validate_todos("not a list") == "`todos` parameter must be a list"

    def test_rejects_non_dict_item(self):
        assert validate_todos(["raw string"]) == "Each todo item must be a dict"

    def test_rejects_empty_description(self):
        err = validate_todos([{"description": "", "status": "pending"}])
        assert "non-empty description" in err

    def test_rejects_whitespace_only_description(self):
        err = validate_todos([{"description": "   ", "status": "pending"}])
        assert "non-empty description" in err

    def test_rejects_non_string_description(self):
        err = validate_todos([{"description": 42, "status": "pending"}])
        assert "non-empty description" in err

    def test_rejects_invalid_status(self):
        err = validate_todos([{"description": "a", "status": "bogus"}])
        assert "valid status" in err

    def test_accepts_default_status(self):
        assert validate_todos([{"description": "a"}]) is None

    def test_rejects_multiple_in_progress(self):
        todos = [
            {"description": "a", "status": "in_progress"},
            {"description": "b", "status": "in_progress"},
        ]
        err = validate_todos(todos)
        assert "Only one task" in err

    def test_accepts_single_in_progress(self):
        todos = [
            {"description": "a", "status": "in_progress"},
            {"description": "b", "status": "pending"},
        ]
        assert validate_todos(todos) is None

    def test_accepts_empty_list(self):
        assert validate_todos([]) is None

    def test_all_valid_statuses_accepted(self):
        for status in VALID_STATUSES:
            assert validate_todos([{"description": "x", "status": status}]) is None


class TestTodoStatusEnum:
    def test_pending_value(self):
        assert TodoStatus.PENDING == "pending"

    def test_all_enum_values_in_valid_set(self):
        assert {s.value for s in TodoStatus} == VALID_STATUSES


class TestWriteTodosInvocation:
    def test_get_description_for_empty(self):
        inv = WriteTodosInvocation([])
        assert inv.get_description() == "Cleared todo list"

    def test_get_description_for_some(self):
        inv = WriteTodosInvocation([{"description": "a"}, {"description": "b"}])
        assert inv.get_description() == "Set 2 todo(s)"

    def test_validate_passes_through(self):
        assert WriteTodosInvocation([{"description": "a"}]).validate() is None

    def test_execute_returns_error_on_invalid(self):
        result = WriteTodosInvocation([{"description": ""}]).execute()
        assert result.llm_content.startswith("Error:")
        assert result.todos == []

    def test_execute_clears_when_empty(self):
        result = WriteTodosInvocation([]).execute()
        assert "cleared" in result.llm_content.lower()
        assert result.todos == []

    def test_execute_formats_numbered_list(self):
        result = WriteTodosInvocation([
            {"description": "first", "status": "pending"},
            {"description": "second", "status": "in_progress"},
        ]).execute()
        assert "1. [pending] first" in result.llm_content
        assert "2. [in_progress] second" in result.llm_content
        assert len(result.todos) == 2
        assert result.todos[0] == Todo(description="first", status="pending")

    def test_execute_uses_default_status(self):
        result = WriteTodosInvocation([{"description": "a"}]).execute()
        assert result.todos[0].status == "pending"


class TestWriteTodosRegistry:
    def test_execute_via_registry(self):
        result = write_todos.execute([{"description": "task one"}])
        assert "task one" in result.llm_content

    def test_registered_as_tool(self):
        primitive = write_todos.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is False
