"""
Full-featured grep/search primitive.

Ported from the Google ADK TypeScript GrepTool/GrepToolInvocation.
Provides content searching with:
- Three-strategy search: git grep → system grep → pure Python fallback
- Regex pattern matching with case-insensitive search
- File inclusion/exclusion patterns
- Per-file and total match limits
- Directory scoping with path validation
- Configurable search timeout
- Formatted results for LLM consumption
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

grep = PrimitiveRegistry("grep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOTAL_MAX_MATCHES = 100
DEFAULT_SEARCH_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of grep errors."""

    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    PATH_NOT_A_DIRECTORY = "path_not_a_directory"
    PATH_NOT_FOUND = "path_not_found"
    INVALID_PATTERN = "invalid_pattern"
    GREP_EXECUTION_ERROR = "grep_execution_error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GrepToolParams:
    """Parameters for a grep invocation."""

    pattern: str
    dir_path: Optional[str] = None
    include_pattern: Optional[str] = None
    exclude_pattern: Optional[str] = None
    names_only: bool = False
    max_matches_per_file: Optional[int] = None
    total_max_matches: int = DEFAULT_TOTAL_MAX_MATCHES


@dataclass
class GrepMatch:
    """A single grep match."""

    file_path: str       # relative to search root
    absolute_path: str
    line_number: int
    line: str


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
    matches: list[GrepMatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class GrepConfig(Protocol):
    """Protocol for grep configuration providers."""

    def get_target_dir(self) -> str: ...

    def validate_path_access(self, path: str, mode: str = "read") -> Optional[str]: ...

    def get_ignore_patterns(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# Grep line parsing
# ---------------------------------------------------------------------------


_GREP_LINE_RE = re.compile(r"^(.+?):(\d+):(.*)$")


def parse_grep_line(line: str, base_path: str) -> Optional[GrepMatch]:
    """Parse a single grep output line (file:line_number:content).

    Args:
        line: Raw grep output line.
        base_path: Absolute directory for path resolution.

    Returns:
        A GrepMatch, or None if the line is malformed.
    """
    line = line.strip()
    if not line:
        return None

    m = _GREP_LINE_RE.match(line)
    if not m:
        return None

    file_raw, line_num_str, content = m.group(1), m.group(2), m.group(3)
    line_number = int(line_num_str)

    abs_path = os.path.abspath(os.path.join(base_path, file_raw))

    # Security: ensure resolved path is within base
    try:
        Path(abs_path).relative_to(base_path)
    except ValueError:
        return None

    rel_path = os.path.relpath(abs_path, base_path)

    return GrepMatch(
        file_path=rel_path,
        absolute_path=abs_path,
        line_number=line_number,
        line=content,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def format_grep_results(
    matches: list[GrepMatch],
    params: GrepToolParams,
    search_location: str,
    total_max: int,
) -> ToolResult:
    """Format grep matches into a ToolResult for LLM consumption."""
    if not matches:
        parts = [f'No matches found for pattern "{params.pattern}" {search_location}.']
        if params.include_pattern:
            parts.append(f"  File filter: {params.include_pattern}")
        return ToolResult(
            llm_content="\n".join(parts),
            return_display=f"No matches found.",
            matches=[],
        )

    if params.names_only:
        unique_files = list(dict.fromkeys(m.file_path for m in matches))
        content = f'Files matching "{params.pattern}" {search_location}:\n'
        content += "\n".join(f"  {f}" for f in unique_files)
        return ToolResult(
            llm_content=content,
            return_display=f"{len(unique_files)} file(s) matched.",
            matches=matches,
        )

    # Group by file
    by_file: dict[str, list[GrepMatch]] = {}
    for m in matches:
        by_file.setdefault(m.file_path, []).append(m)

    lines: list[str] = []
    lines.append(
        f'Found {len(matches)} match(es) for "{params.pattern}" {search_location}:'
    )

    if len(matches) >= total_max:
        lines.append(
            f"(Results capped at {total_max}. Use dir_path or include_pattern to narrow.)"
        )

    lines.append("")

    for file_path, file_matches in by_file.items():
        lines.append(f"{file_path}:")
        for m in file_matches:
            lines.append(f"  {m.line_number}: {m.line}")
        lines.append("")

    return ToolResult(
        llm_content="\n".join(lines),
        return_display=f"{len(matches)} match(es) across {len(by_file)} file(s).",
        matches=matches,
    )


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------


def _is_git_repository(path: str) -> bool:
    """Check if a path is inside a git repository."""
    current = os.path.abspath(path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _is_command_available(command: str) -> bool:
    """Check if a command is available on the system PATH."""
    try:
        subprocess.run(
            ["command", "-v", command],
            capture_output=True,
            shell=True,
            timeout=5,
        )
        # Use shutil.which as a more reliable cross-platform check
        import shutil
        return shutil.which(command) is not None
    except Exception:
        return False


def _run_grep_command(
    args: list[str],
    cwd: str,
    timeout_s: float,
) -> list[str]:
    """Run a grep-like command and return output lines."""
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        # grep returns 1 for no matches, which is fine
        if proc.returncode > 1:
            return []
        return proc.stdout.splitlines()
    except (subprocess.TimeoutExpired, OSError):
        return []


def _git_grep(
    pattern: str,
    search_dir: str,
    include_pattern: Optional[str],
    max_matches_per_file: Optional[int],
    timeout_s: float,
) -> Optional[list[str]]:
    """Try git grep. Returns lines or None if unavailable/failed."""
    if not _is_git_repository(search_dir):
        return None
    if not _is_command_available("git"):
        return None

    args = ["git", "grep", "--untracked", "-n", "-E", "--ignore-case", pattern]
    if max_matches_per_file:
        args.extend(["--max-count", str(max_matches_per_file)])
    if include_pattern:
        args.extend(["--", include_pattern])

    try:
        lines = _run_grep_command(args, search_dir, timeout_s)
        return lines
    except Exception:
        return None


def _system_grep(
    pattern: str,
    search_dir: str,
    include_pattern: Optional[str],
    max_matches_per_file: Optional[int],
    ignore_dirs: list[str],
    timeout_s: float,
) -> Optional[list[str]]:
    """Try system grep. Returns lines or None if unavailable/failed."""
    if not _is_command_available("grep"):
        return None

    args = ["grep", "-r", "-n", "-H", "-E", "-I", "--ignore-case"]
    for d in ignore_dirs:
        args.append(f"--exclude-dir={d}")
    if max_matches_per_file:
        args.extend(["--max-count", str(max_matches_per_file)])
    if include_pattern:
        args.append(f"--include={include_pattern}")
    args.append(pattern)
    args.append(".")

    try:
        lines = _run_grep_command(args, search_dir, timeout_s)
        return lines
    except Exception:
        return None


def _python_grep(
    pattern: str,
    search_dir: str,
    include_pattern: Optional[str],
    exclude_pattern: Optional[re.Pattern] = None,
    max_matches: int = DEFAULT_TOTAL_MAX_MATCHES,
    max_matches_per_file: Optional[int] = None,
    ignore_patterns: Optional[list[str]] = None,
) -> list[GrepMatch]:
    """Pure Python fallback grep implementation."""
    regex = re.compile(pattern, re.IGNORECASE)
    matches: list[GrepMatch] = []
    ignore = ignore_patterns or []

    for root, dirs, files in os.walk(search_dir):
        # Filter ignored directories in-place
        dirs[:] = [
            d for d in dirs
            if not any(fnmatch.fnmatch(d, p) for p in ignore)
            and not d.startswith(".")
        ]

        for filename in files:
            if len(matches) >= max_matches:
                return matches

            # Include filter
            if include_pattern and not fnmatch.fnmatch(filename, include_pattern):
                continue

            # Ignore filter
            if any(fnmatch.fnmatch(filename, p) for p in ignore):
                continue

            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, search_dir)

            try:
                with open(file_path, "r", errors="replace") as f:
                    matches_in_file = 0
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Exclude filter
                            if exclude_pattern and exclude_pattern.search(line):
                                continue
                            matches.append(GrepMatch(
                                file_path=rel_path,
                                absolute_path=os.path.abspath(file_path),
                                line_number=line_num,
                                line=line.rstrip("\n\r"),
                            ))
                            matches_in_file += 1
                            if len(matches) >= max_matches:
                                return matches
                            if max_matches_per_file and matches_in_file >= max_matches_per_file:
                                break
            except (PermissionError, OSError, UnicodeDecodeError):
                continue

    return matches


# ---------------------------------------------------------------------------
# GrepToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class GrepToolInvocation:
    """Manages a single grep search with full lifecycle support.

    Mirrors the TypeScript GrepToolInvocation class. Handles:
    - Parameter validation (regex, paths, limits)
    - Three-tier search strategy: git grep → system grep → Python fallback
    - Match parsing and filtering (include/exclude patterns)
    - Per-file and total match limits
    - Path validation and workspace access control
    - Formatted results for LLM consumption
    """

    def __init__(
        self,
        params: GrepToolParams,
        *,
        target_dir: Optional[str] = None,
        config: Optional[GrepConfig] = None,
        timeout_s: float = DEFAULT_SEARCH_TIMEOUT_S,
        ignore_dirs: Optional[list[str]] = None,
    ):
        self.params = params
        self.target_dir = target_dir or os.getcwd()
        self.config = config
        self.timeout_s = timeout_s
        self.ignore_dirs = ignore_dirs or [
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".next", ".nuxt", "coverage", ".eggs",
        ]

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution."""
        # Validate regex
        try:
            re.compile(self.params.pattern)
        except re.error as e:
            return f"Invalid regular expression pattern: {self.params.pattern}. Error: {e}"

        if self.params.exclude_pattern:
            try:
                re.compile(self.params.exclude_pattern)
            except re.error as e:
                return f"Invalid exclude pattern: {self.params.exclude_pattern}. Error: {e}"

        if self.params.max_matches_per_file is not None and self.params.max_matches_per_file < 1:
            return "max_matches_per_file must be at least 1."

        if self.params.total_max_matches < 1:
            return "total_max_matches must be at least 1."

        # Validate dir_path
        if self.params.dir_path:
            resolved = os.path.abspath(
                os.path.join(self.target_dir, self.params.dir_path)
            )

            if self.config:
                error = self.config.validate_path_access(resolved, "read")
                if error:
                    return error

            if not os.path.exists(resolved):
                return f"Path does not exist: {resolved}"
            if not os.path.isdir(resolved):
                return f"Path is not a directory: {resolved}"

        return None

    # -- Description -------------------------------------------------------

    def get_description(self) -> str:
        """Human-readable description of this search."""
        desc = f"'{self.params.pattern}'"
        if self.params.include_pattern:
            desc += f" in {self.params.include_pattern}"
        if self.params.dir_path:
            desc += f" within {self.params.dir_path}"
        return desc

    # -- Main execution ----------------------------------------------------

    def execute(self) -> ToolResult:
        """Run the grep search and return formatted results."""
        # Validate
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
        if self.params.dir_path:
            search_dir = os.path.abspath(
                os.path.join(self.target_dir, self.params.dir_path)
            )
        else:
            search_dir = self.target_dir

        search_display = self.params.dir_path or "."
        total_max = self.params.total_max_matches

        # Build exclude regex
        exclude_regex: Optional[re.Pattern] = None
        if self.params.exclude_pattern:
            exclude_regex = re.compile(self.params.exclude_pattern, re.IGNORECASE)

        # Get ignore patterns from config or use defaults
        ignore_patterns = (
            self.config.get_ignore_patterns() if self.config else self.ignore_dirs
        )

        # --- Strategy 1: git grep ---
        matches: list[GrepMatch] = []
        lines = _git_grep(
            self.params.pattern,
            search_dir,
            self.params.include_pattern,
            self.params.max_matches_per_file,
            self.timeout_s,
        )

        if lines is None:
            # --- Strategy 2: system grep ---
            lines = _system_grep(
                self.params.pattern,
                search_dir,
                self.params.include_pattern,
                self.params.max_matches_per_file,
                self.ignore_dirs,
                self.timeout_s,
            )

        if lines is not None:
            # Parse output from git grep or system grep
            for line in lines:
                if len(matches) >= total_max:
                    break
                m = parse_grep_line(line, search_dir)
                if m is None:
                    continue
                if exclude_regex and exclude_regex.search(m.line):
                    continue
                matches.append(m)
        else:
            # --- Strategy 3: Pure Python fallback ---
            logger.debug("Falling back to Python grep implementation.")
            matches = _python_grep(
                self.params.pattern,
                search_dir,
                self.params.include_pattern,
                exclude_regex,
                total_max,
                self.params.max_matches_per_file,
                ignore_patterns,
            )

        search_location = f'in path "{search_display}"'
        return format_grep_results(matches, self.params, search_location, total_max)


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@grep.register(kind="tool", readonly=True)
def execute(
    pattern: str,
    dir_path: Optional[str] = None,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    names_only: bool = False,
    max_matches_per_file: Optional[int] = None,
    total_max_matches: int = DEFAULT_TOTAL_MAX_MATCHES,
) -> ToolResult:
    """Search file contents using a regular expression pattern.

    Uses git grep, system grep, or a Python fallback depending on availability.

    Args:
        pattern: Regular expression pattern to search for.
        dir_path: Directory to search in (relative to workspace root). Defaults to workspace root.
        include_pattern: File glob to include (e.g. "*.py", "*.{ts,tsx}").
        exclude_pattern: Regex pattern to exclude from results.
        names_only: If True, only return matching file paths, not line content.
        max_matches_per_file: Cap matches per file to avoid noisy results.
        total_max_matches: Cap total matches returned. Defaults to 100.

    Returns:
        Matching lines grouped by file, or matching file paths if names_only is True.
    """
    ctx = grep.context
    invocation = GrepToolInvocation(
        params=GrepToolParams(
            pattern=pattern,
            dir_path=dir_path,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            names_only=names_only,
            max_matches_per_file=max_matches_per_file,
            total_max_matches=total_max_matches,
        ),
        target_dir=ctx.get("target_dir"),
        config=ctx.get("config"),
        timeout_s=ctx.get("timeout_s", DEFAULT_SEARCH_TIMEOUT_S),
        ignore_dirs=ctx.get("ignore_dirs"),
    )
    return invocation.execute()
