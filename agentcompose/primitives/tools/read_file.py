"""
Full-featured file reading primitive.

Ported from the Google ADK TypeScript ReadFileTool/ReadFileToolInvocation.
Provides file reading with:
- Path validation and workspace access control
- 1-based line range selection (start_line / end_line)
- Automatic truncation with LLM-friendly continuation guidance
- MIME type detection for binary vs text handling
- File ignore pattern support
- Configurable max line output
- Dual LLM/human formatted results
"""

from __future__ import annotations

import fnmatch
import logging
import mimetypes
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

read_file = PrimitiveRegistry("read_file")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_LINES = 500
DEFAULT_MAX_READ_LENGTH = 30_000

# Known binary extensions that should not be read as text
_BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pyc", ".pyo", ".class", ".o", ".obj",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of read file errors."""

    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    IS_DIRECTORY = "is_directory"
    BINARY_FILE = "binary_file"
    IGNORED_FILE = "ignored_file"
    INVALID_PARAMS = "invalid_params"
    READ_ERROR = "read_error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReadFileToolParams:
    """Parameters for a read file invocation."""

    file_path: str
    start_line: Optional[int] = None  # 1-based
    end_line: Optional[int] = None    # 1-based, inclusive


@dataclass
class ToolLocation:
    """A location within a file."""

    path: str
    line: Optional[int] = None


@dataclass
class ToolError:
    """Structured error from a tool execution."""

    message: str
    type: ToolErrorType


@dataclass
class FileContent:
    """Intermediate result from processing file content."""

    llm_content: str
    return_display: str = ""
    error: Optional[str] = None
    error_type: Optional[ToolErrorType] = None
    is_truncated: bool = False
    lines_shown: Optional[tuple[int, int]] = None  # (start, end) 1-based
    original_line_count: Optional[int] = None
    mime_type: Optional[str] = None


@dataclass
class ToolResult:
    """Result formatted for consumption by an LLM agent and human display."""

    llm_content: str
    return_display: str = ""
    error: Optional[ToolError] = None


@dataclass
class PolicyUpdateOptions:
    """Options for updating read permissions after confirmation."""

    args_pattern: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ReadFileConfig(Protocol):
    """Protocol for read file configuration providers."""

    def get_target_dir(self) -> str: ...

    def validate_path_access(self, path: str, mode: str = "read") -> Optional[str]:
        """Return an error message if path is not accessible, or None if OK."""
        ...

    def get_ignore_patterns(self) -> list[str]:
        """Return a list of glob patterns for files to ignore."""
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
    remaining = max_length - len(filename) - 4  # 4 for ".../""
    if remaining <= 0:
        return f".../{filename}"
    prefix = str(Path(*parts[:2]))
    if len(prefix) > remaining:
        prefix = prefix[:remaining]
    return f"{prefix}/.../{filename}"


def get_mime_type(file_path: str) -> Optional[str]:
    """Get the MIME type for a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def is_binary_file(file_path: str) -> bool:
    """Check if a file is likely binary based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _BINARY_EXTENSIONS


def should_ignore_file(file_path: str, ignore_patterns: list[str]) -> bool:
    """Check if a file matches any of the ignore patterns."""
    name = os.path.basename(file_path)
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def get_programming_language(file_path: str) -> Optional[str]:
    """Infer programming language from file extension."""
    ext_map: dict[str, str] = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescriptreact", ".jsx": "javascriptreact",
        ".java": "java", ".go": "go", ".rs": "rust",
        ".rb": "ruby", ".php": "php", ".c": "c", ".cpp": "cpp",
        ".h": "c", ".hpp": "cpp", ".cs": "csharp",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
        ".sql": "sql", ".r": "r", ".R": "r",
        ".lua": "lua", ".pl": "perl", ".pm": "perl",
        ".html": "html", ".css": "css", ".scss": "scss",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".xml": "xml", ".md": "markdown",
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext_map.get(ext)


# ---------------------------------------------------------------------------
# File content processing
# ---------------------------------------------------------------------------


def process_file_content(
    file_path: str,
    target_dir: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_length: int = DEFAULT_MAX_READ_LENGTH,
) -> FileContent:
    """Read and process a file's content, handling truncation and line ranges.

    Args:
        file_path: Absolute path to the file.
        target_dir: The workspace root for relative path display.
        start_line: 1-based start line (inclusive). None starts from beginning.
        end_line: 1-based end line (inclusive). None reads to the end.
        max_lines: Maximum number of lines before truncation.
        max_length: Maximum character count before truncation.

    Returns:
        A FileContent with the processed text and metadata.
    """
    relative_path = make_relative(file_path, target_dir)

    # Check if file exists
    if not os.path.exists(file_path):
        return FileContent(
            llm_content=f"File not found: {relative_path}",
            return_display="File not found.",
            error=f"File not found: {relative_path}",
            error_type=ToolErrorType.FILE_NOT_FOUND,
        )

    # Check if it's a directory
    if os.path.isdir(file_path):
        return FileContent(
            llm_content=f"Path is a directory, not a file: {relative_path}",
            return_display="Path is a directory.",
            error=f"Path is a directory: {relative_path}",
            error_type=ToolErrorType.IS_DIRECTORY,
        )

    # Check if binary
    if is_binary_file(file_path):
        mime = get_mime_type(file_path) or "application/octet-stream"
        return FileContent(
            llm_content=(
                f"Binary file ({mime}): {relative_path}\n"
                f"Cannot display binary content as text."
            ),
            return_display=f"Binary file: {relative_path}",
            error=f"Binary file: {relative_path}",
            error_type=ToolErrorType.BINARY_FILE,
            mime_type=mime,
        )

    # Read the file
    try:
        with open(file_path, "r", errors="replace") as f:
            all_lines = f.readlines()
    except PermissionError:
        return FileContent(
            llm_content=f"Permission denied: {relative_path}",
            return_display="Permission denied.",
            error=f"Permission denied: {relative_path}",
            error_type=ToolErrorType.PERMISSION_DENIED,
        )
    except OSError as e:
        return FileContent(
            llm_content=f"Error reading file: {e}",
            return_display="Error reading file.",
            error=str(e),
            error_type=ToolErrorType.READ_ERROR,
        )

    total_lines = len(all_lines)

    # Apply line range (1-based to 0-based)
    actual_start = 1
    actual_end = total_lines

    if start_line is not None:
        actual_start = max(1, start_line)
    if end_line is not None:
        actual_end = min(total_lines, end_line)

    selected = all_lines[actual_start - 1 : actual_end]

    # Apply max_lines truncation
    is_truncated = False
    if len(selected) > max_lines:
        selected = selected[:max_lines]
        actual_end = actual_start + max_lines - 1
        is_truncated = True

    # Add line numbers and join
    numbered_lines: list[str] = []
    for i, line in enumerate(selected, start=actual_start):
        # Strip trailing newline for consistent formatting, then re-add
        numbered_lines.append(f"{i}\t{line.rstrip()}")

    content = "\n".join(numbered_lines)

    # Apply character-level truncation
    if len(content) > max_length:
        content = content[:max_length]
        is_truncated = True

    # Build display path
    display = shorten_path(relative_path)

    return FileContent(
        llm_content=content,
        return_display=display,
        is_truncated=is_truncated,
        lines_shown=(actual_start, actual_end),
        original_line_count=total_lines,
        mime_type=get_mime_type(file_path),
    )


# ---------------------------------------------------------------------------
# ReadFileToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class ReadFileToolInvocation:
    """Manages a single file read operation with full lifecycle support.

    Mirrors the TypeScript ReadFileToolInvocation class. Handles:
    - Path resolution relative to target directory
    - Parameter validation (empty path, line ranges)
    - Workspace access control via pluggable config
    - File ignore pattern filtering
    - Content processing with truncation and line numbering
    - LLM-friendly truncation messages with continuation guidance
    - Dual LLM/human result formatting
    """

    def __init__(
        self,
        params: ReadFileToolParams,
        *,
        target_dir: Optional[str] = None,
        config: Optional[ReadFileConfig] = None,
        max_lines: int = DEFAULT_MAX_LINES,
        max_length: int = DEFAULT_MAX_READ_LENGTH,
    ):
        self.params = params
        self.target_dir = target_dir or os.getcwd()
        self.config = config
        self.max_lines = max_lines
        self.max_length = max_length
        self.resolved_path = os.path.abspath(
            os.path.join(self.target_dir, self.params.file_path)
        )

    # -- Display helpers ---------------------------------------------------

    def get_description(self) -> str:
        """Short display string for the file being read."""
        relative = make_relative(self.resolved_path, self.target_dir)
        return shorten_path(relative)

    def tool_locations(self) -> list[ToolLocation]:
        """Return locations referenced by this invocation."""
        return [
            ToolLocation(path=self.resolved_path, line=self.params.start_line)
        ]

    def get_policy_update_options(self) -> PolicyUpdateOptions:
        """Build policy update options for this file path."""
        # Build a glob pattern from the file path for policy matching
        return PolicyUpdateOptions(args_pattern=self.params.file_path)

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution.

        Returns an error message string, or None if valid.
        """
        if not self.params.file_path.strip():
            return "The 'file_path' parameter must be non-empty."

        # Path access validation
        if self.config:
            error = self.config.validate_path_access(self.resolved_path, "read")
            if error:
                return error

        # Line range validation
        if self.params.start_line is not None and self.params.start_line < 1:
            return "start_line must be at least 1"
        if self.params.end_line is not None and self.params.end_line < 1:
            return "end_line must be at least 1"
        if (
            self.params.start_line is not None
            and self.params.end_line is not None
            and self.params.start_line > self.params.end_line
        ):
            return "start_line cannot be greater than end_line"

        # Ignore pattern check
        if self.config:
            patterns = self.config.get_ignore_patterns()
            if should_ignore_file(self.resolved_path, patterns):
                return (
                    f"File path '{self.resolved_path}' is ignored "
                    f"by configured ignore patterns."
                )

        return None

    # -- Main execution ----------------------------------------------------

    def execute(self) -> ToolResult:
        """Read the file and return a formatted ToolResult.

        Returns:
            ToolResult with llm_content and return_display.
        """
        # Validate
        validation_error = self.validate()
        if validation_error:
            return ToolResult(
                llm_content=validation_error,
                return_display="Validation failed.",
                error=ToolError(
                    message=validation_error,
                    type=ToolErrorType.PATH_NOT_IN_WORKSPACE,
                ),
            )

        # Process file content
        result = process_file_content(
            file_path=self.resolved_path,
            target_dir=self.target_dir,
            start_line=self.params.start_line,
            end_line=self.params.end_line,
            max_lines=self.max_lines,
            max_length=self.max_length,
        )

        # Handle errors from processing
        if result.error:
            return ToolResult(
                llm_content=result.llm_content,
                return_display=result.return_display or "Error reading file",
                error=ToolError(
                    message=result.error,
                    type=result.error_type or ToolErrorType.READ_ERROR,
                ),
            )

        # Build LLM content with truncation guidance
        llm_content: str
        if result.is_truncated and result.lines_shown and result.original_line_count:
            start, end = result.lines_shown
            total = result.original_line_count
            llm_content = (
                f"IMPORTANT: The file content has been truncated.\n"
                f"Status: Showing lines {start}-{end} of {total} total lines.\n"
                f"Action: To read more of the file, use the 'start_line' and "
                f"'end_line' parameters in a subsequent 'read_file' call. "
                f"For example, to read the next section, use "
                f"start_line: {end + 1}.\n"
                f"\n"
                f"--- FILE CONTENT (truncated) ---\n"
                f"{result.llm_content}"
            )
        else:
            llm_content = result.llm_content

        return ToolResult(
            llm_content=llm_content,
            return_display=result.return_display or "",
        )


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@read_file.register(kind="tool")
def execute(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    target_dir: Optional[str] = None,
    config: Optional[ReadFileConfig] = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_length: int = DEFAULT_MAX_READ_LENGTH,
) -> ToolResult:
    """Read a file with line range selection, truncation, and workspace validation.

    This is a convenience wrapper around ReadFileToolInvocation for use as a
    registered primitive. For full control, instantiate ReadFileToolInvocation
    directly.

    Args:
        file_path: Path to the file to read (absolute or relative to target_dir).
        start_line: 1-based start line (inclusive). None starts from beginning.
        end_line: 1-based end line (inclusive). None reads to the end.
        target_dir: Workspace root directory. Defaults to cwd.
        config: Optional configuration provider for path validation and ignore patterns.
        max_lines: Maximum number of lines before truncation.
        max_length: Maximum character count before truncation.

    Returns:
        A ToolResult with llm_content (for the model) and return_display (for the user).
    """
    invocation = ReadFileToolInvocation(
        params=ReadFileToolParams(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
        ),
        target_dir=target_dir,
        config=config,
        max_lines=max_lines,
        max_length=max_length,
    )
    return invocation.execute()
