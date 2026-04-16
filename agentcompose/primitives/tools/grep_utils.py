"""
Grep result formatting and auto-context enrichment utilities.

Ported from the Google ADK TypeScript grep-utils module.
Provides shared helpers used by both the standard grep and ripgrep primitives:
- Match grouping by file with line-number sorting
- File reading for context enrichment
- Auto-context injection for low-match-count queries (1-3 matches)
- LLM-formatted result output with inline context lines
- Long line truncation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LINE_LENGTH = 500


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GrepMatch:
    """A single grep match."""

    file_path: str       # relative to search root
    absolute_path: str
    line_number: int
    line: str
    is_context: bool = False


@dataclass
class GrepResultSummary:
    """Human-readable summary returned alongside LLM content."""

    summary: str
    matches: list[GrepMatch] = field(default_factory=list)


@dataclass
class FormattedGrepResult:
    """Formatted grep output for both LLM and human consumption."""

    llm_content: str
    return_display: GrepResultSummary


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_matches_by_file(
    matches: list[GrepMatch],
) -> dict[str, list[GrepMatch]]:
    """Group matches by file path and sort each group by line number."""
    groups: dict[str, list[GrepMatch]] = {}
    for match in matches:
        groups.setdefault(match.file_path, []).append(match)
    for file_matches in groups.values():
        file_matches.sort(key=lambda m: m.line_number)
    return groups


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def read_file_lines(absolute_path: str) -> Optional[list[str]]:
    """Read a file and split it into lines.

    Returns None if the file cannot be read.
    """
    try:
        with open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().split("\n")
    except OSError as e:
        logger.warning("Failed to read file for context: %s (%s)", absolute_path, e)
        return None


# ---------------------------------------------------------------------------
# Auto-context enrichment
# ---------------------------------------------------------------------------


def enrich_with_auto_context(
    matches_by_file: dict[str, list[GrepMatch]],
    match_count: int,
    *,
    names_only: bool = False,
    context: Optional[int] = None,
    before: Optional[int] = None,
    after: Optional[int] = None,
) -> None:
    """Enrich grep results with surrounding context when the match count is low.

    When there are 1-3 actual matches and the caller didn't request explicit
    context lines, this reads the matched files and injects surrounding lines
    so the LLM can reason about the code without a follow-up read_file call.

    This optimisation reduces agent turn count by ~10% in benchmarks.

    Mutates ``matches_by_file`` in place.
    """
    if not (1 <= match_count <= 3):
        return
    if names_only:
        return
    if any(x is not None for x in (context, before, after)):
        return

    context_lines = 50 if match_count == 1 else 15

    for file_path, file_matches in matches_by_file.items():
        if not file_matches:
            continue

        file_lines = read_file_lines(file_matches[0].absolute_path)
        if file_lines is None:
            continue

        # Sort matches to process in order
        file_matches.sort(key=lambda m: m.line_number)

        new_matches: list[GrepMatch] = []
        seen_lines: set[int] = set()

        for match in file_matches:
            start = max(0, match.line_number - 1 - context_lines)
            end = min(len(file_lines), match.line_number - 1 + context_lines + 1)

            for i in range(start, end):
                line_num = i + 1
                if line_num not in seen_lines:
                    new_matches.append(GrepMatch(
                        absolute_path=match.absolute_path,
                        file_path=match.file_path,
                        line_number=line_num,
                        line=file_lines[i],
                        is_context=(line_num != match.line_number),
                    ))
                    seen_lines.add(line_num)
                elif line_num == match.line_number:
                    # Promote a previously-added context line to a real match
                    for existing in new_matches:
                        if existing.line_number == line_num:
                            existing.is_context = False
                            break

        matches_by_file[file_path] = sorted(
            new_matches, key=lambda m: m.line_number,
        )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _truncate_line(line: str, max_length: int = MAX_LINE_LENGTH) -> str:
    """Truncate a line if it exceeds max_length graphemes."""
    graphemes = list(line)
    if len(graphemes) > max_length:
        return "".join(graphemes[:max_length]) + "... [truncated]"
    return line


def format_grep_results(
    all_matches: list[GrepMatch],
    *,
    pattern: str,
    search_location: str,
    total_max_matches: int,
    names_only: bool = False,
    include_pattern: Optional[str] = None,
    context: Optional[int] = None,
    before: Optional[int] = None,
    after: Optional[int] = None,
) -> FormattedGrepResult:
    """Format grep matches into structured output for LLM consumption.

    Handles:
    - Empty results with helpful messaging
    - names_only mode (file paths only)
    - Auto-context enrichment for low match counts
    - Per-line context/match markers (``-`` vs ``:``)
    - Long line truncation
    - Truncation notice when results are capped
    """
    filter_note = f' (filter: "{include_pattern}")' if include_pattern else ""

    if not all_matches:
        msg = f'No matches found for pattern "{pattern}" {search_location}{filter_note}.'
        return FormattedGrepResult(
            llm_content=msg,
            return_display=GrepResultSummary(
                summary="No matches found",
                matches=[],
            ),
        )

    matches_by_file = group_matches_by_file(all_matches)
    real_matches = [m for m in all_matches if not m.is_context]
    match_count = len(real_matches)
    match_term = "match" if match_count == 1 else "matches"

    # Auto-context enrichment
    enrich_with_auto_context(
        matches_by_file,
        match_count,
        names_only=names_only,
        context=context,
        before=before,
        after=after,
    )

    was_truncated = match_count >= total_max_matches
    truncated_note = (
        f" (results limited to {total_max_matches} matches for performance)"
        if was_truncated
        else ""
    )

    # Names-only mode
    if names_only:
        file_paths = sorted(matches_by_file.keys())
        llm_content = (
            f'Found {len(file_paths)} files with matches for pattern '
            f'"{pattern}" {search_location}{filter_note}{truncated_note}:\n'
        )
        llm_content += "\n".join(file_paths)
        return FormattedGrepResult(
            llm_content=llm_content.strip(),
            return_display=GrepResultSummary(
                summary=f"Found {len(file_paths)} files"
                + (" (limited)" if was_truncated else ""),
                matches=[],
            ),
        )

    # Full output with context
    llm_content = (
        f'Found {match_count} {match_term} for pattern '
        f'"{pattern}" {search_location}{filter_note}{truncated_note}:\n---\n'
    )

    for file_path, file_matches in matches_by_file.items():
        llm_content += f"File: {file_path}\n"
        for match in file_matches:
            separator = "-" if match.is_context else ":"
            line_content = _truncate_line(match.line.rstrip())
            llm_content += f"L{match.line_number}{separator} {line_content}\n"
        llm_content += "---\n"

    return FormattedGrepResult(
        llm_content=llm_content.strip(),
        return_display=GrepResultSummary(
            summary=f"Found {match_count} {match_term}"
            + (" (limited)" if was_truncated else ""),
            matches=[m for m in all_matches if not m.is_context],
        ),
    )
