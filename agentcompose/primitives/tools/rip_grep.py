"""
Ripgrep-backed search primitive.

Ported from the Google ADK TypeScript RipGrepTool/GrepToolInvocation.
Provides fast content searching via the bundled ``rg`` binary with:
- JSON output parsing for structured match extraction
- Case-sensitive and fixed-string search modes
- Context lines (before, after, symmetric)
- File include/exclude glob and regex patterns
- Per-file and total match limits
- Auto-context enrichment for low-match-count queries
- Configurable search timeout with abort support
- File ignore pattern support (.gitignore, custom ignore files)
- Directory exclusion for common build/cache directories
- Path validation and workspace access control
- Formatted results for LLM consumption
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

rip_grep = PrimitiveRegistry("rip_grep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOTAL_MAX_MATCHES = 100
DEFAULT_SEARCH_TIMEOUT_S = 30.0

# Common directories to exclude from search when not respecting .gitignore
COMMON_DIRECTORY_EXCLUDES: list[str] = [
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".eggs", ".cache",
]

# Additional file patterns to exclude
COMMON_FILE_EXCLUDES: list[str] = [
    "*.log",
    "*.tmp",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of grep errors."""

    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    PATH_NOT_A_DIRECTORY = "path_not_a_directory"
    PATH_NOT_FOUND = "path_not_found"
    FILE_NOT_FOUND = "file_not_found"
    INVALID_PATTERN = "invalid_pattern"
    GREP_EXECUTION_ERROR = "grep_execution_error"
    RIPGREP_NOT_FOUND = "ripgrep_not_found"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RipGrepToolParams:
    """Parameters for a ripgrep invocation."""

    pattern: str
    dir_path: Optional[str] = None
    include_pattern: Optional[str] = None
    exclude_pattern: Optional[str] = None
    names_only: bool = False
    case_sensitive: bool = False
    fixed_strings: bool = False
    context: Optional[int] = None
    after: Optional[int] = None
    before: Optional[int] = None
    no_ignore: bool = False
    max_matches_per_file: Optional[int] = None
    total_max_matches: int = DEFAULT_TOTAL_MAX_MATCHES


@dataclass
class GrepMatch:
    """A single grep match."""

    file_path: str       # relative to search root
    absolute_path: str
    line_number: int
    line: str
    is_context: bool = False


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
class RipGrepConfig(Protocol):
    """Protocol for ripgrep configuration providers."""

    def get_target_dir(self) -> str: ...

    def validate_path_access(self, path: str, mode: str = "read") -> Optional[str]: ...

    def get_ignore_patterns(self) -> list[str]:
        """Return glob patterns for files/dirs to ignore."""
        ...

    def get_ignore_file_paths(self) -> list[str]:
        """Return paths to ignore files (e.g. .gitignore, .agentignore)."""
        ...

    def get_respect_gitignore(self) -> bool:
        """Whether to let ripgrep honour .gitignore files."""
        ...

    def get_debug_mode(self) -> bool: ...


# ---------------------------------------------------------------------------
# Ripgrep binary resolution
# ---------------------------------------------------------------------------


def _get_ripgrep_path() -> Optional[str]:
    """Locate the ``rg`` binary.

    Checks:
    1. A bundled vendor directory relative to this file.
    2. The system PATH via ``shutil.which``.

    Returns the absolute path to the binary, or None.
    """
    system = platform.system().lower()
    arch = platform.machine().lower()

    # Normalise arch names
    if arch in ("x86_64", "amd64"):
        arch = "x86_64"
    elif arch in ("arm64", "aarch64"):
        arch = "arm64"

    ext = ".exe" if system == "win32" or system == "windows" else ""
    bin_name = f"rg-{system}-{arch}{ext}"

    # Check bundled vendor paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "vendor", "ripgrep", bin_name),
        os.path.join(this_dir, "..", "..", "vendor", "ripgrep", bin_name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # Fall back to system-installed rg
    system_rg = shutil.which("rg")
    if system_rg:
        return system_rg

    return None


def can_use_ripgrep() -> bool:
    """Check if ripgrep is available."""
    return _get_ripgrep_path() is not None


def ensure_rg_path() -> str:
    """Return the path to ripgrep, or raise if not found."""
    rg = _get_ripgrep_path()
    if rg is not None:
        return rg
    raise FileNotFoundError(
        "Cannot find ripgrep binary. Install it with your package manager "
        "(e.g. `brew install ripgrep`, `apt install ripgrep`) or place a "
        "bundled binary in the vendor/ripgrep directory."
    )


# ---------------------------------------------------------------------------
# Ripgrep JSON line parser
# ---------------------------------------------------------------------------


def parse_ripgrep_json_line(
    line: str,
    base_path: str,
) -> Optional[GrepMatch]:
    """Parse a single JSON line from ``rg --json`` output.

    Ripgrep emits one JSON object per line. We care about ``type: "match"``
    and ``type: "context"`` messages.

    Returns a GrepMatch, or None if the line is not a match/context line.
    """
    line = line.strip()
    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    msg_type = data.get("type")
    if msg_type not in ("match", "context"):
        return None

    payload = data.get("data", {})
    path_info = payload.get("path", {})
    lines_info = payload.get("lines", {})

    path_text = path_info.get("text")
    lines_text = lines_info.get("text")

    if not path_text or lines_text is None:
        return None

    absolute_path = os.path.abspath(os.path.join(base_path, path_text))

    # Security: ensure resolved path stays within base
    try:
        Path(absolute_path).relative_to(base_path)
    except ValueError:
        return None

    relative_path = os.path.relpath(absolute_path, base_path)

    return GrepMatch(
        file_path=relative_path or os.path.basename(absolute_path),
        absolute_path=absolute_path,
        line_number=payload.get("line_number", 0),
        line=lines_text.rstrip("\n\r"),
        is_context=(msg_type == "context"),
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def format_grep_results(
    matches: list[GrepMatch],
    params: RipGrepToolParams,
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
            return_display="No matches found.",
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

    # Group by file preserving order
    by_file: dict[str, list[GrepMatch]] = {}
    for m in matches:
        by_file.setdefault(m.file_path, []).append(m)

    real_matches = [m for m in matches if not m.is_context]
    lines: list[str] = []
    lines.append(
        f'Found {len(real_matches)} match(es) for "{params.pattern}" {search_location}:'
    )

    if len(real_matches) >= total_max:
        lines.append(
            f"(Results capped at {total_max}. Use dir_path or include_pattern to narrow.)"
        )

    lines.append("")

    for file_path, file_matches in by_file.items():
        lines.append(f"{file_path}:")
        for m in file_matches:
            prefix = " " if m.is_context else ">"
            lines.append(f"  {prefix} {m.line_number}: {m.line}")
        lines.append("")

    return ToolResult(
        llm_content="\n".join(lines),
        return_display=f"{len(real_matches)} match(es) across {len(by_file)} file(s).",
        matches=matches,
    )


# ---------------------------------------------------------------------------
# RipGrepToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class RipGrepToolInvocation:
    """Manages a single ripgrep search with full lifecycle support.

    Mirrors the TypeScript GrepToolInvocation class. Handles:
    - Parameter validation (regex, paths, limits)
    - Ripgrep binary resolution and argument building
    - Streaming JSON output parsing with match limit enforcement
    - Auto-context enrichment for low-match-count queries
    - File filtering via ignore patterns and file discovery
    - Path validation and workspace access control
    - Search timeout with cancellation support
    - Formatted results for LLM consumption
    """

    def __init__(
        self,
        params: RipGrepToolParams,
        *,
        target_dir: Optional[str] = None,
        config: Optional[RipGrepConfig] = None,
        timeout_s: float = DEFAULT_SEARCH_TIMEOUT_S,
    ):
        self.params = params
        self.target_dir = target_dir or os.getcwd()
        self.config = config
        self.timeout_s = timeout_s

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution.

        Returns an error message string, or None if valid.
        """
        if not self.params.fixed_strings:
            try:
                re.compile(self.params.pattern)
            except re.error as e:
                return (
                    f"Invalid regular expression pattern: "
                    f"{self.params.pattern}. Error: {e}"
                )

        if self.params.exclude_pattern:
            try:
                re.compile(self.params.exclude_pattern)
            except re.error as e:
                return (
                    f"Invalid exclude pattern: "
                    f"{self.params.exclude_pattern}. Error: {e}"
                )

        if (
            self.params.max_matches_per_file is not None
            and self.params.max_matches_per_file < 1
        ):
            return "max_matches_per_file must be at least 1."

        if self.params.total_max_matches < 1:
            return "total_max_matches must be at least 1."

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
            if not os.path.isdir(resolved) and not os.path.isfile(resolved):
                return f"Path is not a valid directory or file: {resolved}"

        return None

    # -- Description -------------------------------------------------------

    def get_description(self) -> str:
        """Human-readable description of this search."""
        desc = f"'{self.params.pattern}'"
        if self.params.include_pattern:
            desc += f" in {self.params.include_pattern}"
        if self.params.dir_path:
            desc += f" within {self.params.dir_path}"
        else:
            desc += " within ./"
        return desc

    # -- Ripgrep argument building -----------------------------------------

    def _build_rg_args(
        self,
        search_paths: list[str],
        *,
        context_override: Optional[int] = None,
    ) -> list[str]:
        """Build the argument list for the ``rg`` subprocess."""
        args = ["--json"]

        if not self.params.case_sensitive:
            args.append("--ignore-case")

        if self.params.fixed_strings:
            args.append("--fixed-strings")

        args.extend(["--regexp", self.params.pattern])

        # Context lines
        ctx = context_override if context_override is not None else self.params.context
        if ctx is not None:
            args.extend(["--context", str(ctx)])
        if context_override is None:
            if self.params.after is not None:
                args.extend(["--after-context", str(self.params.after)])
            if self.params.before is not None:
                args.extend(["--before-context", str(self.params.before)])

        if self.params.no_ignore:
            args.append("--no-ignore")
        else:
            # Respect gitignore by default; config can override
            if self.config and not self.config.get_respect_gitignore():
                args.extend(["--no-ignore-vcs", "--no-ignore-exclude"])

            # Exclude common directories and file patterns
            for d in COMMON_DIRECTORY_EXCLUDES:
                args.extend(["--glob", f"!{d}"])
            for f in COMMON_FILE_EXCLUDES:
                args.extend(["--glob", f"!{f}"])

            # Add custom ignore files from config
            if self.config:
                for ignore_path in self.config.get_ignore_file_paths():
                    args.extend(["--ignore-file", ignore_path])

        if self.params.max_matches_per_file is not None:
            args.extend(["--max-count", str(self.params.max_matches_per_file)])

        if self.params.include_pattern:
            args.extend(["--glob", self.params.include_pattern])

        args.extend(["--threads", "4"])
        args.extend(search_paths)

        return args

    # -- Ripgrep execution -------------------------------------------------

    async def _run_ripgrep(
        self,
        search_paths: list[str],
        base_path: str,
        max_matches: int,
        *,
        context_override: Optional[int] = None,
    ) -> list[GrepMatch]:
        """Execute ripgrep and stream-parse JSON output.

        Args:
            search_paths: Absolute paths to search.
            base_path: Base directory for computing relative paths.
            max_matches: Stop after this many non-context matches.
            context_override: Override context lines (used by auto-enrichment).

        Returns:
            List of parsed GrepMatch results.
        """
        rg_path = ensure_rg_path()
        args = self._build_rg_args(
            search_paths, context_override=context_override,
        )

        proc = await asyncio.create_subprocess_exec(
            rg_path, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdout is not None

        results: list[GrepMatch] = []
        matches_found = 0

        exclude_regex: Optional[re.Pattern] = None
        if self.params.exclude_pattern:
            flags = 0 if self.params.case_sensitive else re.IGNORECASE
            exclude_regex = re.compile(self.params.exclude_pattern, flags)

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break

            line_str = line_bytes.decode(errors="replace")
            match = parse_ripgrep_json_line(line_str, base_path)
            if match is None:
                continue

            if exclude_regex and exclude_regex.search(match.line):
                continue

            results.append(match)
            if not match.is_context:
                matches_found += 1

            if matches_found >= max_matches:
                proc.kill()
                break

        # Wait for process to finish (might already be done)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        return results

    # -- Auto-context enrichment -------------------------------------------

    def _filter_ignored_files(
        self,
        matches: list[GrepMatch],
    ) -> list[GrepMatch]:
        """Remove matches in files that should be ignored per config.

        Mirrors the TypeScript ``fileDiscoveryService.filterFiles()`` call
        that runs after both the initial search and the enrichment search.
        """
        if self.params.no_ignore:
            return matches
        if not self.config:
            return matches

        ignore_patterns = self.config.get_ignore_patterns()
        if not ignore_patterns:
            return matches

        import fnmatch

        def is_ignored(abs_path: str) -> bool:
            name = os.path.basename(abs_path)
            for pattern in ignore_patterns:
                if fnmatch.fnmatch(name, pattern):
                    return True
                if fnmatch.fnmatch(abs_path, pattern):
                    return True
            return False

        return [m for m in matches if not is_ignored(m.absolute_path)]

    async def _enrich_with_auto_context(
        self,
        matches: list[GrepMatch],
        search_dir: str,
    ) -> list[GrepMatch]:
        """Re-run ripgrep with added context if the match count is low.

        When there are 1-3 matches and the user didn't request explicit
        context, we re-search with expanded context to give the LLM more
        surrounding code for reasoning.

        Returns the enriched matches, or the original matches if enrichment
        was skipped.
        """
        real_count = sum(1 for m in matches if not m.is_context)

        if not (1 <= real_count <= 3):
            return matches

        if self.params.names_only:
            return matches

        if any(x is not None for x in (
            self.params.context, self.params.before, self.params.after,
        )):
            return matches

        context_lines = 50 if real_count == 1 else 15
        unique_files = list({m.absolute_path for m in matches})

        enriched = await self._run_ripgrep(
            search_paths=unique_files,
            base_path=search_dir,
            max_matches=self.params.total_max_matches,
            context_override=context_lines,
        )

        enriched = self._filter_ignored_files(enriched)

        # Update params so downstream formatting knows context was applied
        self.params.context = context_lines

        return enriched

    # -- Main execution ----------------------------------------------------

    async def execute(self) -> ToolResult:
        """Run the ripgrep search and return formatted results."""
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
        path_param = self.params.dir_path or "."
        search_dir = os.path.abspath(
            os.path.join(self.target_dir, path_param)
        )

        # Path access validation
        if self.config:
            access_error = self.config.validate_path_access(search_dir, "read")
            if access_error:
                return ToolResult(
                    llm_content=access_error,
                    return_display="Error: Path not in workspace.",
                    error=ToolError(
                        message=access_error,
                        type=ToolErrorType.PATH_NOT_IN_WORKSPACE,
                    ),
                )

        # Existence check
        if not os.path.exists(search_dir):
            msg = f"Path does not exist: {search_dir}"
            return ToolResult(
                llm_content=msg,
                return_display="Error: Path does not exist.",
                error=ToolError(message=msg, type=ToolErrorType.FILE_NOT_FOUND),
            )

        total_max = self.params.total_max_matches

        # Run the search with a timeout
        try:
            matches = await asyncio.wait_for(
                self._run_ripgrep(
                    search_paths=[search_dir],
                    base_path=search_dir,
                    max_matches=total_max,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            msg = (
                f"Search timed out after {self.timeout_s:.0f}s. "
                f"Consider narrowing your search with dir_path or include_pattern."
            )
            return ToolResult(
                llm_content=msg,
                return_display="Search timed out.",
                error=ToolError(
                    message=msg, type=ToolErrorType.GREP_EXECUTION_ERROR,
                ),
            )
        except FileNotFoundError as e:
            return ToolResult(
                llm_content=str(e),
                return_display="ripgrep not found.",
                error=ToolError(
                    message=str(e), type=ToolErrorType.RIPGREP_NOT_FOUND,
                ),
            )
        except OSError as e:
            msg = f"Error during ripgrep execution: {e}"
            return ToolResult(
                llm_content=msg,
                return_display=f"Error: {e}",
                error=ToolError(
                    message=msg, type=ToolErrorType.GREP_EXECUTION_ERROR,
                ),
            )

        # Post-search file filtering (mirrors TS fileDiscoveryService.filterFiles)
        matches = self._filter_ignored_files(matches)

        # Auto-context enrichment for low match counts
        try:
            matches = await self._enrich_with_auto_context(
                matches, search_dir,
            )
        except Exception:
            # Enrichment is best-effort; keep original matches on failure
            logger.debug("Auto-context enrichment failed", exc_info=True)

        search_location = f'in path "{self.params.dir_path or "."}"'
        return format_grep_results(matches, self.params, search_location, total_max)


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@rip_grep.register(kind="tool", readonly=True)
async def execute(
    pattern: str,
    dir_path: Optional[str] = None,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    names_only: bool = False,
    case_sensitive: bool = False,
    fixed_strings: bool = False,
    context: Optional[int] = None,
    after: Optional[int] = None,
    before: Optional[int] = None,
    no_ignore: bool = False,
    max_matches_per_file: Optional[int] = None,
    total_max_matches: int = DEFAULT_TOTAL_MAX_MATCHES,
) -> ToolResult:
    """Search file contents using ripgrep with structured JSON output.

    A faster alternative to the standard grep primitive, backed by the ``rg``
    binary. Supports regex and literal string matching, context lines,
    per-file and total match limits, and automatic context enrichment for
    low-match-count queries.

    Args:
        pattern: Regular expression pattern to search for (or literal if fixed_strings).
        dir_path: Directory to search in (relative to workspace root). Defaults to ".".
        include_pattern: File glob to include (e.g. "*.py", "*.{ts,tsx}").
        exclude_pattern: Regex pattern to exclude from results.
        names_only: If True, only return matching file paths, not line content.
        case_sensitive: If True, search case-sensitively. Defaults to False.
        fixed_strings: If True, treat pattern as a literal string. Defaults to False.
        context: Show this many lines of context around each match.
        after: Show this many lines after each match.
        before: Show this many lines before each match.
        no_ignore: If True, do not respect .gitignore or default excludes.
        max_matches_per_file: Cap matches per file.
        total_max_matches: Cap total matches returned. Defaults to 100.

    Returns:
        Matching lines grouped by file, or matching file paths if names_only.
    """
    ctx = rip_grep.context
    invocation = RipGrepToolInvocation(
        params=RipGrepToolParams(
            pattern=pattern,
            dir_path=dir_path,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            names_only=names_only,
            case_sensitive=case_sensitive,
            fixed_strings=fixed_strings,
            context=context,
            after=after,
            before=before,
            no_ignore=no_ignore,
            max_matches_per_file=max_matches_per_file,
            total_max_matches=total_max_matches,
        ),
        target_dir=ctx.get("target_dir"),
        config=ctx.get("config"),
        timeout_s=ctx.get("timeout_s", DEFAULT_SEARCH_TIMEOUT_S),
    )
    return await invocation.execute()
