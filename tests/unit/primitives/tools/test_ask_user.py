"""Unit tests for the ask_user primitive."""

import io
import json

import pytest

from agentcompose.primitives.tools.ask_user import (
    AskUserInvocation,
    Option,
    Question,
    QuestionType,
    _parse_questions,
    ask_user,
    default_input_handler,
    validate_questions,
)


class TestValidateQuestions:
    def test_empty_list_rejected(self):
        assert validate_questions([]) == "At least one question is required."

    def test_non_dict_question_rejected(self):
        err = validate_questions(["string"])
        assert "must be a dict" in err

    def test_choice_requires_options(self):
        err = validate_questions([{"question": "pick", "type": "choice"}])
        assert "options" in err

    def test_choice_requires_at_least_two_options(self):
        err = validate_questions([{
            "question": "pick",
            "type": "choice",
            "options": [{"label": "only"}],
        }])
        assert "2-4" in err

    def test_choice_rejects_more_than_four_options(self):
        err = validate_questions([{
            "question": "pick",
            "type": "choice",
            "options": [{"label": str(i)} for i in range(5)],
        }])
        assert "at most 4" in err

    def test_option_must_have_label(self):
        err = validate_questions([{
            "question": "pick",
            "type": "choice",
            "options": [{"label": "ok"}, {"description": "missing label"}],
        }])
        assert "label" in err

    def test_option_label_must_be_non_empty(self):
        err = validate_questions([{
            "question": "pick",
            "type": "choice",
            "options": [{"label": "ok"}, {"label": "  "}],
        }])
        assert "label" in err

    def test_option_description_must_be_string(self):
        err = validate_questions([{
            "question": "pick",
            "type": "choice",
            "options": [
                {"label": "a"},
                {"label": "b", "description": 42},
            ],
        }])
        assert "description" in err

    def test_valid_free_text(self):
        assert validate_questions([{"question": "Q", "type": "free_text"}]) is None

    def test_valid_choice(self):
        assert validate_questions([{
            "question": "Pick",
            "type": "choice",
            "options": [{"label": "a"}, {"label": "b"}],
        }]) is None


class TestParseQuestions:
    def test_parses_free_text(self):
        questions = _parse_questions([{"question": "Q1", "type": "free_text"}])
        assert len(questions) == 1
        assert questions[0].question == "Q1"
        assert questions[0].type == "free_text"
        assert questions[0].options is None

    def test_parses_choice_with_options(self):
        questions = _parse_questions([{
            "question": "Pick",
            "type": "choice",
            "options": [{"label": "a", "description": "alpha"}],
        }])
        assert questions[0].options == [Option(label="a", description="alpha")]


class TestAskUserInvocation:
    def test_dismiss_returns_dismissed_flag(self):
        inv = AskUserInvocation(
            questions=[{"question": "Q"}],
            input_handler=lambda qs: None,
        )
        result = inv.execute()
        assert result.dismissed is True

    def test_returns_answers_as_json(self):
        inv = AskUserInvocation(
            questions=[{"question": "Name?"}],
            input_handler=lambda qs: {"0": "Bob"},
        )
        result = inv.execute()
        payload = json.loads(result.llm_content)
        assert payload == {"answers": {"0": "Bob"}}
        assert result.answers == {"0": "Bob"}
        assert result.dismissed is False

    def test_validation_error_short_circuits(self):
        inv = AskUserInvocation(
            questions=[],
            input_handler=lambda qs: {"0": "X"},
        )
        result = inv.execute()
        assert result.llm_content.startswith("Error:")

    def test_handler_not_called_on_validation_error(self):
        called = []
        inv = AskUserInvocation(
            questions=[],
            input_handler=lambda qs: called.append(qs) or {"0": "X"},
        )
        inv.execute()
        assert called == []

    def test_get_description_lists_questions(self):
        inv = AskUserInvocation(
            questions=[{"question": "First?"}, {"question": "Second?"}],
        )
        d = inv.get_description()
        assert "First?" in d
        assert "Second?" in d


class TestAskUserRegistry:
    def test_registered_as_tool(self):
        primitive = ask_user.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"

    def test_registry_uses_configured_input_handler(self):
        ask_user.configure(input_handler=lambda qs: {"0": "stub"})
        try:
            result = ask_user.execute([{"question": "Q?"}])
            assert result.answers == {"0": "stub"}
        finally:
            ask_user.configure(input_handler=None)


class TestValidationOptionShape:
    def test_option_must_be_dict(self):
        err = validate_questions([{
            "question": "pick",
            "type": "choice",
            "options": [{"label": "ok"}, "not-a-dict"],
        }])
        assert "must be a dict" in err


class TestDefaultInputHandler:
    def test_free_text_uses_input(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda prompt="": "Alice")
        questions = _parse_questions([{"question": "Name?", "type": "free_text"}])
        answers = default_input_handler(questions)
        assert answers == {"0": "Alice"}

    def test_choice_valid_index(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        questions = _parse_questions([{
            "question": "Pick",
            "type": "choice",
            "options": [{"label": "a"}, {"label": "b"}],
        }])
        answers = default_input_handler(questions)
        assert answers == {"0": "b"}

    def test_choice_out_of_range_returns_raw(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "99")
        questions = _parse_questions([{
            "question": "Pick",
            "type": "choice",
            "options": [{"label": "a"}, {"label": "b"}],
        }])
        answers = default_input_handler(questions)
        assert answers == {"0": "99"}

    def test_choice_value_error_returns_none(self, monkeypatch):
        def bad_input(prompt=""):
            raise ValueError("not a number")

        monkeypatch.setattr("builtins.input", bad_input)
        questions = _parse_questions([{
            "question": "Pick",
            "type": "choice",
            "options": [{"label": "a"}, {"label": "b"}],
        }])
        assert default_input_handler(questions) is None

    def test_free_text_eof_returns_none(self, monkeypatch):
        def eof(prompt=""):
            raise EOFError()

        monkeypatch.setattr("builtins.input", eof)
        questions = _parse_questions([{"question": "Q?"}])
        assert default_input_handler(questions) is None

    def test_header_label_displayed(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda prompt="": "x")
        questions = _parse_questions([{
            "question": "Q?",
            "header": "Custom Header",
        }])
        default_input_handler(questions)
        captured = capsys.readouterr()
        assert "Custom Header" in captured.out


class TestAskUserNoAnswers:
    def test_empty_answers_dict_returns_no_answer_display(self):
        inv = AskUserInvocation(
            questions=[{"question": "Q?"}],
            input_handler=lambda qs: {},
        )
        result = inv.execute()
        assert result.dismissed is False
        # Hits the `else` branch at line 266: "User submitted without answering"
        assert "without answering" in result.return_display


class TestQuestionTypeEnum:
    def test_values(self):
        assert QuestionType.FREE_TEXT == "free_text"
        assert QuestionType.CHOICE == "choice"
