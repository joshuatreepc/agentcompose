"""Unit tests for the grep primitive.

These tests focus on the GrepToolInvocation and pure-Python fallback so they
behave the same regardless of whether ``git grep`` or system ``grep`` are
available. Tests of the parsing helpers run against representative inputs.
"""

import os

import sys

import agentcompose.primitives.tools.grep  # noqa: F401  (ensure module loaded)

# The package's `grep` attribute is the registry, which shadows the submodule;
# fetch the actual module from sys.modules to monkeypatch its private helpers.
grep_module = sys.modules["agentcompose.primitives.tools.grep"]

from agentcompose.primitives.tools.grep import (
    GrepMatch,
    GrepToolInvocation,
    GrepToolParams,
    ToolErrorType,
    _python_grep,
    format_grep_results,
    grep,
    parse_grep_line,
)


# ---------------------------------------------------------------------------
# parse_grep_line
# ---------------------------------------------------------------------------


class TestParseGrepLine:
    def test_parses_simple_line(self, tmp_path):
        line = "src/file.py:42:    def foo():"
        result = parse_grep_line(line, str(tmp_path))
        assert result is not None
        assert result.line_number == 42
        assert "def foo()" in result.line
        # file_path is relative to base_path
        assert result.file_path == os.path.join("src", "file.py")

    def test_returns_none_for_blank(self, tmp_path):
        assert parse_grep_line("", str(tmp_path)) is None

    def test_returns_none_for_malformed(self, tmp_path):
        assert parse_grep_line("not a grep line", str(tmp_path)) is None

    def test_security_rejects_outside_base(self, tmp_path):
        # Path containing ".." that escapes base_path should be rejected
        line = "../etc/passwd:1:root:x:0:0"
        result = parse_grep_line(line, str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# format_grep_results
# ---------------------------------------------------------------------------


class TestFormatGrepResults:
    def _params(self, **kw):
        return GrepToolParams(pattern="x", **kw)

    def test_empty_returns_no_match_message(self):
        result = format_grep_results([], self._params(), 'in path "."', 100)
        assert result.matches == []
        assert "No matches" in result.llm_content

    def test_groups_matches_by_file(self):
        m1 = GrepMatch("a.py", "/p/a.py", 1, "foo")
        m2 = GrepMatch("a.py", "/p/a.py", 5, "bar")
        m3 = GrepMatch("b.py", "/p/b.py", 2, "baz")
        result = format_grep_results([m1, m2, m3], self._params(), 'in "."', 100)
        assert "a.py:" in result.llm_content
        assert "b.py:" in result.llm_content
        assert "1: foo" in result.llm_content
        assert "5: bar" in result.llm_content

    def test_names_only_lists_unique_files(self):
        m1 = GrepMatch("a.py", "/p/a.py", 1, "x")
        m2 = GrepMatch("a.py", "/p/a.py", 2, "x")
        m3 = GrepMatch("b.py", "/p/b.py", 1, "x")
        params = self._params(names_only=True)
        result = format_grep_results([m1, m2, m3], params, 'in "."', 100)
        assert "a.py" in result.llm_content
        assert "b.py" in result.llm_content
        # Only listed once each (names_only collapses)
        assert result.llm_content.count("a.py") == 1

    def test_cap_warning_when_at_max(self):
        matches = [GrepMatch("a.py", "/p/a.py", i, "x") for i in range(1, 11)]
        result = format_grep_results(matches, self._params(), 'in "."', 10)
        assert "capped" in result.llm_content.lower()


# ---------------------------------------------------------------------------
# Python fallback
# ---------------------------------------------------------------------------


class TestPythonGrepFallback:
    def test_finds_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world\nfoo bar\n")
        (tmp_path / "b.py").write_text("nothing here\n")
        matches = _python_grep("hello", str(tmp_path), include_pattern=None)
        assert len(matches) == 1
        assert matches[0].file_path == "a.py"
        assert matches[0].line_number == 1

    def test_case_insensitive_default(self, tmp_path):
        (tmp_path / "a.py").write_text("HELLO\n")
        matches = _python_grep("hello", str(tmp_path), include_pattern=None)
        assert len(matches) == 1

    def test_include_pattern_filters_by_filename(self, tmp_path):
        (tmp_path / "a.py").write_text("foo\n")
        (tmp_path / "a.txt").write_text("foo\n")
        matches = _python_grep("foo", str(tmp_path), include_pattern="*.py")
        assert len(matches) == 1
        assert matches[0].file_path == "a.py"

    def test_skips_dot_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("foo\n")
        (tmp_path / "src.py").write_text("foo\n")
        matches = _python_grep("foo", str(tmp_path), include_pattern=None)
        assert len(matches) == 1
        assert "src.py" in matches[0].file_path

    def test_max_matches_cap(self, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("hit\n")
        matches = _python_grep(
            "hit", str(tmp_path), include_pattern=None, max_matches=3,
        )
        assert len(matches) == 3

    def test_max_matches_per_file_cap(self, tmp_path):
        (tmp_path / "a.py").write_text("hit\n" * 10)
        matches = _python_grep(
            "hit", str(tmp_path), include_pattern=None,
            max_matches_per_file=2,
        )
        assert len(matches) == 2


# ---------------------------------------------------------------------------
# GrepToolInvocation
# ---------------------------------------------------------------------------


class TestGrepToolInvocation:
    def test_validation_invalid_regex(self, tmp_path):
        inv = GrepToolInvocation(
            GrepToolParams(pattern="[unclosed"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.INVALID_PATTERN

    def test_validation_invalid_exclude_regex(self, tmp_path):
        inv = GrepToolInvocation(
            GrepToolParams(pattern="x", exclude_pattern="[bad"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None

    def test_validation_max_matches_per_file_must_be_at_least_one(self, tmp_path):
        inv = GrepToolInvocation(
            GrepToolParams(pattern="x", max_matches_per_file=0),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None

    def test_validation_total_max_must_be_at_least_one(self, tmp_path):
        inv = GrepToolInvocation(
            GrepToolParams(pattern="x", total_max_matches=0),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None

    def test_missing_dir_path(self, tmp_path):
        inv = GrepToolInvocation(
            GrepToolParams(pattern="x", dir_path="missing"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_FOUND

    def test_dir_path_pointing_to_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("")
        inv = GrepToolInvocation(
            GrepToolParams(pattern="x", dir_path="x.txt"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_A_DIRECTORY

    def test_finds_matches_via_python_fallback(self, tmp_path, monkeypatch):
        # Force the Python fallback by stubbing both git and system grep
        monkeypatch.setattr(grep_module, "_git_grep", lambda *a, **kw: None)
        monkeypatch.setattr(grep_module, "_system_grep", lambda *a, **kw: None)

        (tmp_path / "a.py").write_text("alpha\nbeta\n")
        (tmp_path / "b.py").write_text("gamma\n")
        inv = GrepToolInvocation(
            GrepToolParams(pattern="beta"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.matches
        assert any("beta" in m.line for m in result.matches)

    def test_exclude_pattern_filters(self, tmp_path, monkeypatch):
        monkeypatch.setattr(grep_module, "_git_grep", lambda *a, **kw: None)
        monkeypatch.setattr(grep_module, "_system_grep", lambda *a, **kw: None)

        (tmp_path / "a.py").write_text("alpha keep\nalpha drop\n")
        inv = GrepToolInvocation(
            GrepToolParams(pattern="alpha", exclude_pattern="drop"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.matches
        assert all("drop" not in m.line for m in result.matches)

    def test_get_description(self, tmp_path):
        inv = GrepToolInvocation(
            GrepToolParams(
                pattern="foo",
                include_pattern="*.py",
                dir_path="src",
            ),
            target_dir=str(tmp_path),
        )
        d = inv.get_description()
        assert "foo" in d
        assert "*.py" in d
        assert "src" in d


class TestGrepRegistry:
    def test_registered_as_tool_readonly(self):
        primitive = grep.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is True
