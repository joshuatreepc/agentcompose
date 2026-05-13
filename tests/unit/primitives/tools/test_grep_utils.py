"""Unit tests for the grep_utils module."""

from agentcompose.primitives.tools.grep_utils import (
    FormattedGrepResult,
    GrepMatch,
    GrepResultSummary,
    MAX_LINE_LENGTH,
    _truncate_line,
    enrich_with_auto_context,
    format_grep_results,
    group_matches_by_file,
    read_file_lines,
)


def _match(file_path, line_number, line, *, absolute_path=None, is_context=False):
    return GrepMatch(
        file_path=file_path,
        absolute_path=absolute_path or f"/abs/{file_path}",
        line_number=line_number,
        line=line,
        is_context=is_context,
    )


# ---------------------------------------------------------------------------
# group_matches_by_file
# ---------------------------------------------------------------------------


class TestGroupMatchesByFile:
    def test_empty(self):
        assert group_matches_by_file([]) == {}

    def test_groups_and_sorts_by_line(self):
        matches = [
            _match("a.py", 30, "x"),
            _match("a.py", 10, "y"),
            _match("b.py", 5, "z"),
        ]
        grouped = group_matches_by_file(matches)
        assert list(grouped.keys()) == ["a.py", "b.py"]
        assert [m.line_number for m in grouped["a.py"]] == [10, 30]
        assert [m.line_number for m in grouped["b.py"]] == [5]


# ---------------------------------------------------------------------------
# read_file_lines
# ---------------------------------------------------------------------------


class TestReadFileLines:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("alpha\nbeta\ngamma")
        lines = read_file_lines(str(f))
        assert lines == ["alpha", "beta", "gamma"]

    def test_missing_returns_none(self, tmp_path):
        assert read_file_lines(str(tmp_path / "no.txt")) is None

    def test_directory_returns_none(self, tmp_path):
        # Opening a directory raises OSError
        assert read_file_lines(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _truncate_line
# ---------------------------------------------------------------------------


class TestTruncateLine:
    def test_short_unchanged(self):
        assert _truncate_line("hello", 100) == "hello"

    def test_long_truncated_with_notice(self):
        out = _truncate_line("a" * 600, 500)
        assert out.startswith("a" * 500)
        assert "[truncated]" in out

    def test_default_max_length(self):
        out = _truncate_line("x" * (MAX_LINE_LENGTH + 10))
        assert "[truncated]" in out


# ---------------------------------------------------------------------------
# enrich_with_auto_context
# ---------------------------------------------------------------------------


class TestEnrichWithAutoContext:
    def test_no_enrichment_when_too_many_matches(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("L1\nL2\nL3\nL4\n")
        groups = {"x.py": [_match("x.py", 2, "L2", absolute_path=str(f))]}
        enrich_with_auto_context(groups, match_count=4)
        # match_count > 3 → no enrichment
        assert len(groups["x.py"]) == 1

    def test_no_enrichment_when_zero_matches(self, tmp_path):
        groups = {}
        enrich_with_auto_context(groups, match_count=0)
        assert groups == {}

    def test_no_enrichment_in_names_only_mode(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("L1\nL2\nL3\n")
        groups = {"x.py": [_match("x.py", 2, "L2", absolute_path=str(f))]}
        enrich_with_auto_context(groups, match_count=1, names_only=True)
        assert len(groups["x.py"]) == 1

    def test_no_enrichment_when_context_already_specified(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("L1\nL2\nL3\n")
        groups = {"x.py": [_match("x.py", 2, "L2", absolute_path=str(f))]}
        enrich_with_auto_context(groups, match_count=1, context=2)
        assert len(groups["x.py"]) == 1

    def test_single_match_adds_50_lines_context(self, tmp_path):
        f = tmp_path / "x.py"
        lines = "\n".join(f"line{i}" for i in range(1, 101))
        f.write_text(lines)
        groups = {"x.py": [_match("x.py", 50, "line50", absolute_path=str(f))]}
        enrich_with_auto_context(groups, match_count=1)
        # Should include surrounding 50 lines on each side
        result = groups["x.py"]
        line_numbers = [m.line_number for m in result]
        assert 50 in line_numbers
        assert min(line_numbers) <= 1 or 50 - min(line_numbers) <= 50
        # The actual match should not be marked as context
        match_line = next(m for m in result if m.line_number == 50)
        assert match_line.is_context is False

    def test_three_matches_use_15_lines_context(self, tmp_path):
        f = tmp_path / "x.py"
        lines = "\n".join(f"line{i}" for i in range(1, 101))
        f.write_text(lines)
        groups = {
            "x.py": [
                _match("x.py", 50, "line50", absolute_path=str(f)),
                _match("x.py", 60, "line60", absolute_path=str(f)),
                _match("x.py", 70, "line70", absolute_path=str(f)),
            ]
        }
        enrich_with_auto_context(groups, match_count=3)
        result = groups["x.py"]
        line_numbers = {m.line_number for m in result}
        # Real matches should still appear
        assert {50, 60, 70}.issubset(line_numbers)
        # Real matches should not be flagged as context
        for m in result:
            if m.line_number in (50, 60, 70):
                assert m.is_context is False

    def test_unreadable_file_skipped(self, tmp_path):
        groups = {
            "missing.py": [
                _match(
                    "missing.py",
                    1,
                    "x",
                    absolute_path=str(tmp_path / "does_not_exist.py"),
                )
            ]
        }
        # Should not raise; matches left unchanged
        enrich_with_auto_context(groups, match_count=1)
        assert len(groups["missing.py"]) == 1


# ---------------------------------------------------------------------------
# format_grep_results
# ---------------------------------------------------------------------------


class TestFormatGrepResults:
    def test_no_matches(self):
        result = format_grep_results(
            [],
            pattern="foo",
            search_location="in /tmp",
            total_max_matches=100,
        )
        assert isinstance(result, FormattedGrepResult)
        assert "No matches found" in result.llm_content
        assert "foo" in result.llm_content
        assert result.return_display.matches == []

    def test_no_matches_with_filter_note(self):
        result = format_grep_results(
            [],
            pattern="foo",
            search_location="in /tmp",
            total_max_matches=100,
            include_pattern="*.py",
        )
        assert "*.py" in result.llm_content

    def test_names_only_output(self, tmp_path):
        matches = [
            _match("a.py", 1, "match"),
            _match("a.py", 2, "match"),
            _match("b.py", 1, "match"),
        ]
        result = format_grep_results(
            matches,
            pattern="match",
            search_location="in /tmp",
            total_max_matches=100,
            names_only=True,
        )
        assert "a.py" in result.llm_content
        assert "b.py" in result.llm_content
        assert "Found 2 files" in result.llm_content

    def test_full_output_with_context_separator(self):
        matches = [
            _match("a.py", 10, "the match", is_context=False),
            _match("a.py", 11, "context line", is_context=True),
        ]
        result = format_grep_results(
            matches,
            pattern="match",
            search_location="in /tmp",
            total_max_matches=100,
            context=2,  # prevents auto-enrichment
        )
        # Real match uses ':' separator, context uses '-'
        assert "L10: the match" in result.llm_content
        assert "L11- context line" in result.llm_content

    def test_truncation_note_when_capped(self):
        matches = [_match("a.py", i, "x") for i in range(1, 6)]
        result = format_grep_results(
            matches,
            pattern="x",
            search_location="in /tmp",
            total_max_matches=5,  # match_count >= total_max_matches
            context=1,
        )
        assert "results limited" in result.llm_content
        assert "(limited)" in result.return_display.summary

    def test_long_line_truncated_in_output(self):
        long = "a" * 800
        result = format_grep_results(
            [_match("a.py", 1, long)],
            pattern="a",
            search_location="in /tmp",
            total_max_matches=100,
            context=1,
        )
        assert "[truncated]" in result.llm_content

    def test_single_match_term(self):
        result = format_grep_results(
            [_match("a.py", 1, "x")],
            pattern="x",
            search_location="in /tmp",
            total_max_matches=100,
            context=1,
        )
        assert "Found 1 match " in result.llm_content

    def test_plural_match_term(self):
        matches = [_match("a.py", i, "x") for i in (1, 2)]
        result = format_grep_results(
            matches,
            pattern="x",
            search_location="in /tmp",
            total_max_matches=100,
            context=1,
        )
        assert "Found 2 matches " in result.llm_content

    def test_return_display_excludes_context_lines(self):
        matches = [
            _match("a.py", 10, "real", is_context=False),
            _match("a.py", 11, "ctx", is_context=True),
        ]
        result = format_grep_results(
            matches,
            pattern="x",
            search_location="in /tmp",
            total_max_matches=100,
            context=1,
        )
        # Only the real match is in return_display.matches
        assert len(result.return_display.matches) == 1
        assert result.return_display.matches[0].line == "real"
