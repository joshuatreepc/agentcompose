"""Unit tests for the rip_grep primitive.

Tests that don't require an actual ``rg`` binary (parsing, validation,
formatting) always run. Live execution tests are skipped when ripgrep is
not installed.
"""

import asyncio
import json

import pytest

from agentcompose.primitives.tools.rip_grep import (
    GrepMatch,
    RipGrepToolInvocation,
    RipGrepToolParams,
    ToolErrorType,
    can_use_ripgrep,
    format_grep_results,
    parse_ripgrep_json_line,
    rip_grep,
)


needs_rg = pytest.mark.skipif(
    not can_use_ripgrep(),
    reason="ripgrep binary not available",
)


# ---------------------------------------------------------------------------
# parse_ripgrep_json_line
# ---------------------------------------------------------------------------


class TestParseRipgrepJsonLine:
    def test_match_line(self, tmp_path):
        payload = {
            "type": "match",
            "data": {
                "path": {"text": "src/file.py"},
                "lines": {"text": "def foo():\n"},
                "line_number": 7,
            },
        }
        result = parse_ripgrep_json_line(json.dumps(payload), str(tmp_path))
        assert result is not None
        assert result.line_number == 7
        assert result.line == "def foo():"
        assert result.is_context is False
        assert result.file_path == "src/file.py" or result.file_path.endswith("file.py")

    def test_context_line_marked(self, tmp_path):
        payload = {
            "type": "context",
            "data": {
                "path": {"text": "x.py"},
                "lines": {"text": "around\n"},
                "line_number": 1,
            },
        }
        result = parse_ripgrep_json_line(json.dumps(payload), str(tmp_path))
        assert result is not None
        assert result.is_context is True

    def test_other_types_ignored(self, tmp_path):
        payload = {"type": "begin", "data": {}}
        assert parse_ripgrep_json_line(json.dumps(payload), str(tmp_path)) is None

    def test_blank_line(self, tmp_path):
        assert parse_ripgrep_json_line("", str(tmp_path)) is None

    def test_invalid_json(self, tmp_path):
        assert parse_ripgrep_json_line("not json", str(tmp_path)) is None

    def test_path_escape_rejected(self, tmp_path):
        payload = {
            "type": "match",
            "data": {
                "path": {"text": "../../etc/passwd"},
                "lines": {"text": "x\n"},
                "line_number": 1,
            },
        }
        assert parse_ripgrep_json_line(json.dumps(payload), str(tmp_path)) is None


# ---------------------------------------------------------------------------
# format_grep_results
# ---------------------------------------------------------------------------


class TestFormatRipGrepResults:
    def _params(self, **kw):
        return RipGrepToolParams(pattern="x", **kw)

    def test_no_matches(self):
        result = format_grep_results([], self._params(), 'in path "."', 100)
        assert "No matches" in result.llm_content

    def test_separates_context_from_real_matches(self):
        m_real = GrepMatch("a.py", "/p/a.py", 5, "real", is_context=False)
        m_ctx = GrepMatch("a.py", "/p/a.py", 4, "around", is_context=True)
        result = format_grep_results([m_ctx, m_real], self._params(), 'in "."', 100)
        assert "1 match" in result.llm_content  # only one real match
        # both lines are still rendered with different prefixes
        assert "real" in result.llm_content
        assert "around" in result.llm_content

    def test_names_only(self):
        m1 = GrepMatch("a.py", "/p/a.py", 1, "x")
        m2 = GrepMatch("b.py", "/p/b.py", 1, "x")
        result = format_grep_results(
            [m1, m2], self._params(names_only=True), 'in "."', 100,
        )
        assert "a.py" in result.llm_content
        assert "b.py" in result.llm_content


# ---------------------------------------------------------------------------
# RipGrepToolInvocation validation
# ---------------------------------------------------------------------------


class TestRipGrepValidation:
    def test_invalid_regex_rejected(self, tmp_path):
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="[bad"),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        assert result.error is not None
        assert result.error.type == ToolErrorType.INVALID_PATTERN

    def test_fixed_strings_skips_regex_compile(self, tmp_path):
        # `[bad` is bad regex but valid as a literal — fixed_strings should skip the check.
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="[bad", fixed_strings=True),
            target_dir=str(tmp_path),
        )
        # Validate directly (not run rg)
        assert inv.validate() is None

    def test_max_per_file_zero_rejected(self, tmp_path):
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="x", max_matches_per_file=0),
            target_dir=str(tmp_path),
        )
        assert inv.validate() is not None

    def test_total_max_zero_rejected(self, tmp_path):
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="x", total_max_matches=0),
            target_dir=str(tmp_path),
        )
        assert inv.validate() is not None

    def test_missing_dir_path_rejected(self, tmp_path):
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="x", dir_path="missing_dir"),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        assert result.error is not None

    def test_get_description(self, tmp_path):
        inv = RipGrepToolInvocation(
            RipGrepToolParams(
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


# ---------------------------------------------------------------------------
# Live ripgrep execution (skipped if rg unavailable)
# ---------------------------------------------------------------------------


@needs_rg
class TestRipGrepLive:
    def test_finds_match_in_tmp(self, tmp_path):
        (tmp_path / "a.py").write_text("alpha\nbeta\n")
        (tmp_path / "b.py").write_text("gamma\n")
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="beta"),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        assert result.error is None
        assert any("beta" in m.line for m in result.matches if not m.is_context)

    def test_include_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("hit\n")
        (tmp_path / "a.txt").write_text("hit\n")
        inv = RipGrepToolInvocation(
            RipGrepToolParams(pattern="hit", include_pattern="*.py"),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        # only a.py file should be in the results
        files = {m.file_path for m in result.matches}
        assert any(f.endswith("a.py") for f in files)
        assert not any(f.endswith("a.txt") for f in files)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRipGrepRegistry:
    def test_registered_as_tool_readonly(self):
        primitive = rip_grep.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is True
