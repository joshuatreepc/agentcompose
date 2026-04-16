"""
Glob file search primitive.

Ported from the Google ADK TypeScript GlobTool/GlobToolInvocation.
Provides fast file pattern matching with:
- Standard glob patterns (``**/*.py``, ``src/**/*.tsx``)
- Case-sensitive and case-insensitive matching
- Recency-aware sorting (recent files first, then alphabetical)
- File ignore pattern support (.gitignore, custom ignore files)
- Directory scoping with path validation
- Workspace access control
- Formatted results for LLM consumption
"""

from __future__ import annotations

import fnmatch
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

glob_tool = PrimitiveRegistry("glob")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONE_DAY_S = 24 * 60 * 60

# Common directories to exclude from glob searches
DEFAULT_EXCLUDES: list[str] = [
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".eggs", ".cache",
    ".DS_Store",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of glob errors."""

    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    PATH_NOT_A_DIRECTORY = "path_not_a_directory"
    PATH_NOT_FOUND = "path_not_found"
    INVALID_PATTERN = "invalid_pattern"
    GLOB_EXECUTION_ERROR = "glob_execution_error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GlobToolParams:
    """Parameters for a glob invocation."""

    pattern: str
    dir_path: Optional[str] = None
    case_sensitive: bool = False


@dataclass
class GlobEntry:
    """A single matched file with metadata."""

    absolute_path: str
    relative_path: str
    mtime: float = 0.0


@dataclass
class ToolError:
    """Structured error from a tool execution."""

    message: str
    type: ToolErrorType


@dataclass
class ToolResult:
    """Result formatted for consumption by an LLM agent and human display."""

    llm_content: str
    return_display: str = ""
    error: Optional[ToolError] = None
    entries: Optional[list[GlobEntry]] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class GlobConfig(Protocol):
    """Protocol for glob configuration providers."""

    def get_target_dir(self) -> str: ...

    def validate_path_access(self, path: str, mode: str = "read") -> Optional[str]: ...

    def get_ignore_patterns(self) -> list[str]:
        """Return glob patterns for files/dirs to ignore."""
        ...


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def make_relative(file_path: str, base_dir: str) -> str:
    """Make a path relative to the base directory."""
    try:
        return os.path.relpath(file_path, base_dir)
    except ValueError:
        return file_path


def shorten_path(file_path: str, max_length: int = 60) -> str:
    """Shorten a path for display, keeping the filename and trimming the middle."""
    if len(file_path) <= max_length:
        return file_path
    parts = Path(file_path).parts
    if len(parts) <= 2:
        return file_path
    filename = parts[-1]
    remaining = max_length - len(filename) - 4
    if remaining <= 0:
        return f".../{filename}"
    prefix = str(Path(*parts[:2]))
    if len(prefix) > remaining:
        prefix = prefix[:remaining]
    return f"{prefix}/.../{filename}"


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def sort_file_entries(
    entries: list[GlobEntry],
    now_timestamp: float,
    recency_threshold_s: float = ONE_DAY_S,
) -> list[GlobEntry]:
    """Sort file entries: recent files first (newest to oldest), then alphabetical.

    Files modified within ``recency_threshold_s`` of ``now_timestamp`` are
    considered recent and sorted by mtime descending. Older files are
    sorted alphabetically by path.
    """
    def sort_key(entry: GlobEntry) -> tuple[int, float, str]:
        is_recent = (now_timestamp - entry.mtime) < recency_threshold_s
        if is_recent:
            # Group 0 (first), sort by -mtime (newest first)
            return (0, -entry.mtime, "")
        else:
            # Group 1 (second), sort alphabetically
            return (1, 0.0, entry.relative_path)

    return sorted(entries, key=sort_key)


# ---------------------------------------------------------------------------
# Glob matching engine
# ---------------------------------------------------------------------------


def _should_exclude(path_str: str, exclude_patterns: list[str]) -> bool:
    """Check if a path matches any exclusion pattern."""
    name = os.path.basename(path_str)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False


def _fnmatch_case(name: str, pattern: str, case_sensitive: bool) -> bool:
    """Match a name against a glob pattern, with optional case folding."""
    if case_sensitive:
        return fnmatch.fnmatchcase(name, pattern)
    return fnmatch.fnmatch(name.lower(), pattern.lower())


def _glob_search(
    pattern: str,
    search_dir: str,
    case_sensitive: bool = False,
    exclude_patterns: Optional[list[str]] = None,
) -> tuple[list[GlobEntry], int]:
    """Walk the directory tree and match files against a glob pattern.

    For case-sensitive mode, uses ``pathlib.Path.glob`` directly.
    For case-insensitive mode (the default), walks the tree manually with
    ``fnmatch`` so that ``**/*.PY`` matches ``.py`` files.

    Args:
        pattern: The glob pattern (e.g. ``**/*.py``, ``src/*.ts``).
        search_dir: Absolute path to the root search directory.
        case_sensitive: Whether matching is case-sensitive.
        exclude_patterns: File/directory patterns to skip.

    Returns:
        Tuple of (matched entries, ignored count).
    """
    excludes = exclude_patterns or DEFAULT_EXCLUDES
    search_path = Path(search_dir)
    entries: list[GlobEntry] = []
    ignored_count = 0

    # If the pattern resolves to an exact existing file, return it directly
    exact = search_path / pattern
    if exact.is_file():
        try:
            st = exact.stat()
        except OSError:
            st = None
        rel = os.path.relpath(str(exact), search_dir)
        entries.append(GlobEntry(
            absolute_path=str(exact),
            relative_path=rel,
            mtime=st.st_mtime if st else 0.0,
        ))
        return entries, 0

    # Case-sensitive: use pathlib.glob directly (faster)
    # Case-insensitive: walk the tree and fnmatch manually
    if case_sensitive:
        candidates = _pathlib_glob(search_path, pattern)
    else:
        candidates = _walk_glob(search_path, pattern)

    for match in candidates:
        abs_path = str(match)
        rel_path = os.path.relpath(abs_path, search_dir)

        # Check if any path component matches an exclude
        parts = Path(rel_path).parts
        if any(_should_exclude(part, excludes) for part in parts[:-1]):
            ignored_count += 1
            continue
        if _should_exclude(parts[-1], excludes):
            ignored_count += 1
            continue

        try:
            st = match.stat()
            mtime = st.st_mtime
        except OSError:
            mtime = 0.0

        entries.append(GlobEntry(
            absolute_path=abs_path,
            relative_path=rel_path,
            mtime=mtime,
        ))

    return entries, ignored_count


def _pathlib_glob(search_path: Path, pattern: str) -> list[Path]:
    """Case-sensitive glob via pathlib."""
    try:
        return [m for m in search_path.glob(pattern) if m.is_file()]
    except OSError as e:
        logger.debug("Glob search error: %s", e)
        return []


def _walk_glob(search_path: Path, pattern: str) -> list[Path]:
    """Case-insensitive glob via os.walk + fnmatch.

    Splits the pattern into directory components and a file pattern,
    then walks the tree matching each level case-insensitively.
    """
    results: list[Path] = []

    # Normalise: split pattern into parts for component-level matching
    # e.g. "src/**/*.py" → ["src", "**", "*.py"]
    pat_parts = Path(pattern).parts
    file_pattern = pat_parts[-1] if pat_parts else "*"
    is_recursive = "**" in pat_parts

    try:
        for root, _dirs, files in os.walk(str(search_path)):
            rel_root = os.path.relpath(root, str(search_path))
            if rel_root == ".":
                rel_root = ""

            # For non-recursive patterns with directory components,
            # check that we're in the right subdirectory
            if not is_recursive and len(pat_parts) > 1:
                dir_pattern = str(Path(*pat_parts[:-1]))
                if not _fnmatch_case(rel_root, dir_pattern, case_sensitive=False):
                    continue

            for filename in files:
                if is_recursive:
                    # Match full relative path against the pattern
                    rel_file = os.path.join(rel_root, filename) if rel_root else filename
                    if _fnmatch_case(rel_file, pattern, case_sensitive=False):
                        results.append(Path(os.path.join(root, filename)))
                else:
                    # Match just the filename against the file part
                    if _fnmatch_case(filename, file_pattern, case_sensitive=False):
                        results.append(Path(os.path.join(root, filename)))
    except OSError as e:
        logger.debug("Glob walk error: %s", e)

    return results


# ---------------------------------------------------------------------------
# GlobToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class GlobToolInvocation:
    """Manages a single glob search with full lifecycle support.

    Mirrors the TypeScript GlobToolInvocation class. Handles:
    - Parameter validation (pattern, path access)
    - Glob pattern matching via pathlib
    - File exclusion via ignore patterns
    - Recency-aware sorting
    - Path validation and workspace access control
    - Formatted results for LLM consumption
    """

    def __init__(
        self,
        params: GlobToolParams,
        *,
        target_dir: Optional[str] = None,
        config: Optional[GlobConfig] = None,
    ):
        self.params = params
        self.target_dir = target_dir or os.getcwd()
        self.config = config

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution.

        Returns an error message string, or None if valid.
        """
        if not self.params.pattern or not self.params.pattern.strip():
            return "The 'pattern' parameter cannot be empty."

        search_dir = os.path.abspath(
            os.path.join(self.target_dir, self.params.dir_path or ".")
        )

        if self.config:
            error = self.config.validate_path_access(search_dir, "read")
            if error:
                return error

        if not os.path.exists(search_dir):
            return f"Search path does not exist: {search_dir}"
        if not os.path.isdir(search_dir):
            return f"Search path is not a directory: {search_dir}"

        return None

    # -- Description -------------------------------------------------------

    def get_description(self) -> str:
        """Human-readable description of this glob search."""
        desc = f"'{self.params.pattern}'"
        if self.params.dir_path:
            search_dir = os.path.abspath(
                os.path.join(self.target_dir, self.params.dir_path)
            )
            rel = make_relative(search_dir, self.target_dir)
            desc += f" within {shorten_path(rel)}"
        return desc

    # -- Main execution ----------------------------------------------------

    def execute(self) -> ToolResult:
        """Run the glob search and return formatted results."""
        validation_error = self.validate()
        if validation_error:
            error_type = ToolErrorType.INVALID_PATTERN
            if "directory" in validation_error.lower():
                error_type = ToolErrorType.PATH_NOT_A_DIRECTORY
            elif "not exist" in validation_error.lower():
                error_type = ToolErrorType.PATH_NOT_FOUND
            elif "workspace" in validation_error.lower():
                error_type = ToolErrorType.PATH_NOT_IN_WORKSPACE
            return ToolResult(
                llm_content=validation_error,
                return_display="Validation failed.",
                error=ToolError(message=validation_error, type=error_type),
            )

        # Resolve search directory
        search_dir = os.path.abspath(
            os.path.join(self.target_dir, self.params.dir_path or ".")
        )
        search_display = self.params.dir_path or "."

        # Get exclude patterns
        exclude_patterns = (
            self.config.get_ignore_patterns() if self.config else DEFAULT_EXCLUDES
        )

        try:
            entries, ignored_count = _glob_search(
                pattern=self.params.pattern,
                search_dir=search_dir,
                case_sensitive=self.params.case_sensitive,
                exclude_patterns=exclude_patterns,
            )
        except Exception as e:
            msg = f"Error during glob search operation: {e}"
            logger.warning(msg)
            return ToolResult(
                llm_content=msg,
                return_display="Error: An unexpected error occurred.",
                error=ToolError(
                    message=msg, type=ToolErrorType.GLOB_EXECUTION_ERROR,
                ),
            )

        if not entries:
            msg = (
                f'No files found matching pattern "{self.params.pattern}" '
                f'within {search_display}'
            )
            if ignored_count > 0:
                msg += f" ({ignored_count} files were ignored)"
            return ToolResult(
                llm_content=msg,
                return_display="No files found.",
                entries=[],
            )

        # Sort: recent files first, then alphabetical
        now = time.time()
        sorted_entries = sort_file_entries(entries, now)

        # Format output
        file_list = "\n".join(e.relative_path for e in sorted_entries)
        file_count = len(sorted_entries)

        result_msg = (
            f'Found {file_count} file(s) matching "{self.params.pattern}" '
            f"within {search_display}"
        )
        if ignored_count > 0:
            result_msg += f" ({ignored_count} additional files were ignored)"
        result_msg += (
            f", sorted by modification time (newest first):\n{file_list}"
        )

        return ToolResult(
            llm_content=result_msg,
            return_display=f"Found {file_count} matching file(s).",
            entries=sorted_entries,
        )


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@glob_tool.register(kind="tool", readonly=True)
def execute(
    pattern: str,
    dir_path: Optional[str] = None,
    case_sensitive: bool = False,
) -> ToolResult:
    """Find files by glob pattern.

    Fast file pattern matching tool that works with any codebase size.
    Supports standard glob patterns like ``**/*.py`` or ``src/**/*.ts``.
    Returns matching file paths sorted by modification time (recent first).

    Args:
        pattern: The glob pattern to match files against (e.g. ``**/*.py``).
        dir_path: Directory to search in (relative to workspace root).
            Defaults to the workspace root.
        case_sensitive: Whether matching is case-sensitive. Defaults to False.

    Returns:
        Matching file paths sorted by modification time.
    """
    ctx = glob_tool.context
    invocation = GlobToolInvocation(
        params=GlobToolParams(
            pattern=pattern,
            dir_path=dir_path,
            case_sensitive=case_sensitive,
        ),
        target_dir=ctx.get("target_dir"),
        config=ctx.get("config"),
    )
    return invocation.execute()
