"""
Full-featured file writing primitive.

Ported from the Google ADK TypeScript WriteFileTool/WriteFileToolInvocation.
Provides file writing with:
- Path validation and workspace access control
- Automatic parent directory creation
- New file creation vs overwrite detection
- Line ending preservation (LF/CRLF)
- Omission placeholder detection (rejects incomplete content)
- Unified diff generation for LLM consumption
- Diff context snippet for verification without a follow-up read
- Detailed error classification (permission, disk space, directory target)
- ModifyContext integration for user-review workflows
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

write_file = PrimitiveRegistry("write_file")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Patterns that indicate the model omitted code with a placeholder comment
_OMISSION_PATTERNS: list[re.Pattern] = [
    re.compile(r"//\s*\.{3}\s*(?:rest|remaining|other|existing)\b", re.IGNORECASE),
    re.compile(r"#\s*\.{3}\s*(?:rest|remaining|other|existing)\b", re.IGNORECASE),
    re.compile(r"//\s*\.\.\.\s*$", re.MULTILINE),
    re.compile(r"#\s*\.\.\.\s*$", re.MULTILINE),
    re.compile(r"/\*\s*\.{3}\s*\*/", re.IGNORECASE),
    re.compile(
        r"(?://|#|/\*)\s*(?:rest of|remaining|other|existing)\s+\w+",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of write file errors."""

    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    FILE_WRITE_FAILURE = "file_write_failure"
    PERMISSION_DENIED = "permission_denied"
    NO_SPACE_LEFT = "no_space_left"
    TARGET_IS_DIRECTORY = "target_is_directory"
    INVALID_PARAMS = "invalid_params"
    OMISSION_DETECTED = "omission_detected"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WriteFileToolParams:
    """Parameters for a write file invocation."""

    file_path: str
    content: str
    modified_by_user: bool = False
    ai_proposed_content: Optional[str] = None


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
class FileDiff:
    """Diff information for display."""

    file_diff: str
    file_name: str
    file_path: str
    original_content: str
    new_content: str
    is_new_file: bool
    additions: int = 0
    deletions: int = 0


@dataclass
class ToolResult:
    """Result formatted for consumption by an LLM agent and human display."""

    llm_content: str
    return_display: str = ""
    error: Optional[ToolError] = None
    file_diff: Optional[FileDiff] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class WriteFileConfig(Protocol):
    """Protocol for write file configuration providers."""

    def get_target_dir(self) -> str: ...

    def validate_path_access(self, path: str, mode: str = "write") -> Optional[str]:
        """Return an error message if path is not accessible, or None if OK."""
        ...


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def detect_line_ending(text: str) -> str:
    """Detect the dominant line ending in a text string.

    Returns ``'\\r\\n'`` if CRLF is more common, ``'\\n'`` otherwise.
    """
    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    return "\r\n" if crlf_count > lf_count else "\n"


def detect_omission_placeholders(content: str) -> list[str]:
    """Detect omission placeholder comments in content.

    Returns a list of matched placeholder strings. An empty list means
    the content is complete.
    """
    found: list[str] = []
    for pattern in _OMISSION_PATTERNS:
        for match in pattern.finditer(content):
            found.append(match.group(0))
    return found


def create_unified_diff(
    file_name: str,
    old_content: str,
    new_content: str,
    old_label: str = "Current",
    new_label: str = "Proposed",
) -> str:
    """Generate a unified diff string between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{file_name} ({old_label})",
        tofile=f"{file_name} ({new_label})",
    )
    return "".join(diff_lines)


def get_diff_stat(
    old_content: str,
    new_content: str,
) -> tuple[int, int]:
    """Compute (additions, deletions) between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    additions = 0
    deletions = 0
    for tag, _, _, _, _ in difflib.SequenceMatcher(
        None, old_lines, new_lines
    ).get_opcodes():
        if tag == "insert":
            additions += 1
        elif tag == "delete":
            deletions += 1
        elif tag == "replace":
            additions += 1
            deletions += 1
    return additions, deletions


def get_diff_context_snippet(
    old_content: str,
    new_content: str,
    context_lines: int = 5,
) -> str:
    """Generate a concise diff snippet with surrounding context.

    Shows a few lines around each change so the LLM can verify the write
    without needing a follow-up read_file call.
    """
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        n=context_lines,
    )
    return "".join(diff)


def make_relative(file_path: str, base_dir: str) -> str:
    """Make a path relative to the base directory."""
    try:
        return os.path.relpath(file_path, base_dir)
    except ValueError:
        return file_path


def shorten_path(file_path: str, max_length: int = 60) -> str:
    """Shorten a path for display."""
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
# WriteFileToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class WriteFileToolInvocation:
    """Manages a single file write operation with full lifecycle support.

    Mirrors the TypeScript WriteFileToolInvocation class. Handles:
    - Path resolution relative to target directory
    - Parameter validation (empty path, directory target, omission detection)
    - Workspace access control via pluggable config
    - Automatic parent directory creation
    - Line ending detection and preservation
    - Diff generation for LLM consumption and human display
    - Detailed error classification
    """

    def __init__(
        self,
        params: WriteFileToolParams,
        *,
        target_dir: Optional[str] = None,
        config: Optional[WriteFileConfig] = None,
    ):
        self.params = params
        self.target_dir = target_dir or os.getcwd()
        self.config = config
        self.resolved_path = os.path.abspath(
            os.path.join(self.target_dir, self.params.file_path)
        )

    # -- Display helpers ---------------------------------------------------

    def get_description(self) -> str:
        """Short display string for the file being written."""
        relative = make_relative(self.resolved_path, self.target_dir)
        return f"Writing to {shorten_path(relative)}"

    def tool_locations(self) -> list[ToolLocation]:
        """Return locations referenced by this invocation."""
        return [ToolLocation(path=self.resolved_path)]

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution.

        Returns an error message string, or None if valid.
        """
        if not self.params.file_path or not self.params.file_path.strip():
            return 'Missing or empty "file_path".'

        # Path access validation
        if self.config:
            error = self.config.validate_path_access(self.resolved_path, "write")
            if error:
                return error

        # Check if target is a directory
        if os.path.exists(self.resolved_path):
            try:
                st = os.lstat(self.resolved_path)
                if stat.S_ISDIR(st.st_mode):
                    return f"Path is a directory, not a file: {self.resolved_path}"
            except OSError as e:
                return (
                    f"Error accessing path properties for validation: "
                    f"{self.resolved_path}. Reason: {e}"
                )

        # Omission placeholder detection
        placeholders = detect_omission_placeholders(self.params.content)
        if placeholders:
            return (
                "`content` contains an omission placeholder "
                "(for example 'rest of methods ...'). "
                "Provide complete file content."
            )

        return None

    # -- Main execution ----------------------------------------------------

    def execute(self) -> ToolResult:
        """Write the file and return a formatted ToolResult.

        Returns:
            ToolResult with llm_content, diff info, and error details.
        """
        # Validate
        validation_error = self.validate()
        if validation_error:
            error_type = ToolErrorType.INVALID_PARAMS
            if "workspace" in validation_error.lower():
                error_type = ToolErrorType.PATH_NOT_IN_WORKSPACE
            elif "directory" in validation_error.lower():
                error_type = ToolErrorType.TARGET_IS_DIRECTORY
            elif "omission" in validation_error.lower():
                error_type = ToolErrorType.OMISSION_DETECTED
            return ToolResult(
                llm_content=validation_error,
                return_display="Validation failed.",
                error=ToolError(message=validation_error, type=error_type),
            )

        # Read existing content (if any) for diff generation
        original_content = ""
        file_exists = False
        try:
            with open(self.resolved_path, "r", encoding="utf-8", errors="replace") as f:
                original_content = f.read()
            file_exists = True
        except FileNotFoundError:
            file_exists = False
        except OSError as e:
            # File exists but unreadable — continue with empty original
            file_exists = True
            logger.debug("Could not read existing file %s: %s", self.resolved_path, e)

        is_new_file = not file_exists

        # Determine line ending to preserve
        final_content = self.params.content
        if not is_new_file and original_content:
            if detect_line_ending(original_content) == "\r\n":
                # Normalise to CRLF to match existing file
                final_content = re.sub(r"\r?\n", "\r\n", final_content)

        # Ensure parent directory exists
        dir_name = os.path.dirname(self.resolved_path)
        try:
            os.makedirs(dir_name, exist_ok=True)
        except OSError as e:
            msg = f"Error creating directory '{dir_name}': {e}"
            return ToolResult(
                llm_content=msg,
                return_display=msg,
                error=ToolError(message=msg, type=ToolErrorType.FILE_WRITE_FAILURE),
            )

        # Write the file
        try:
            with open(self.resolved_path, "w", encoding="utf-8") as f:
                f.write(final_content)
        except PermissionError as e:
            msg = f"Permission denied writing to file: {self.resolved_path} ({e})"
            return ToolResult(
                llm_content=msg,
                return_display=msg,
                error=ToolError(message=msg, type=ToolErrorType.PERMISSION_DENIED),
            )
        except OSError as e:
            error_type = ToolErrorType.FILE_WRITE_FAILURE
            msg = f"Error writing to file '{self.resolved_path}': {e}"

            if hasattr(e, "errno"):
                import errno
                if e.errno == errno.ENOSPC:
                    msg = f"No space left on device: {self.resolved_path}"
                    error_type = ToolErrorType.NO_SPACE_LEFT
                elif e.errno == errno.EISDIR:
                    msg = f"Target is a directory, not a file: {self.resolved_path}"
                    error_type = ToolErrorType.TARGET_IS_DIRECTORY

            return ToolResult(
                llm_content=msg,
                return_display=msg,
                error=ToolError(message=msg, type=error_type),
            )

        # Generate diff
        file_name = os.path.basename(self.resolved_path)
        file_diff_str = create_unified_diff(
            file_name, original_content, final_content, "Original", "Written",
        )
        additions, deletions = get_diff_stat(original_content, final_content)

        # Build LLM content
        llm_parts: list[str] = []
        if is_new_file:
            llm_parts.append(
                f"Successfully created and wrote to new file: {self.resolved_path}."
            )
        else:
            llm_parts.append(
                f"Successfully overwrote file: {self.resolved_path}."
            )

        if self.params.modified_by_user:
            llm_parts.append(
                f"User modified the `content` to be: {self.params.content}"
            )

        # Append a diff snippet so the LLM can verify without a follow-up read
        snippet = get_diff_context_snippet(
            "" if is_new_file else original_content,
            final_content,
            context_lines=5,
        )
        if snippet:
            llm_parts.append(f"Here is the updated code:\n{snippet}")

        diff_info = FileDiff(
            file_diff=file_diff_str,
            file_name=file_name,
            file_path=self.resolved_path,
            original_content=original_content,
            new_content=final_content,
            is_new_file=is_new_file,
            additions=additions,
            deletions=deletions,
        )

        return ToolResult(
            llm_content=" ".join(llm_parts),
            return_display=file_diff_str or f"Wrote {len(final_content)} bytes.",
            file_diff=diff_info,
        )


# ---------------------------------------------------------------------------
# ModifyContext for user-review workflow
# ---------------------------------------------------------------------------


class WriteFileModifyContext:
    """ModifyContext implementation for the write_file tool.

    Enables the modify-with-editor workflow from ``modifable_tool.py``.
    """

    def __init__(self, target_dir: str):
        self.target_dir = target_dir

    def get_file_path(self, params: WriteFileToolParams) -> str:
        return params.file_path

    async def get_current_content(self, params: WriteFileToolParams) -> str:
        resolved = os.path.abspath(
            os.path.join(self.target_dir, params.file_path)
        )
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    async def get_proposed_content(self, params: WriteFileToolParams) -> str:
        return params.content

    def create_updated_params(
        self,
        _old_content: str,
        modified_proposed_content: str,
        original_params: WriteFileToolParams,
    ) -> WriteFileToolParams:
        return WriteFileToolParams(
            file_path=original_params.file_path,
            content=modified_proposed_content,
            modified_by_user=True,
            ai_proposed_content=original_params.content,
        )


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@write_file.register(kind="tool", readonly=False)
def execute(
    file_path: str,
    content: str,
) -> ToolResult:
    """Write content to a file, creating it if it doesn't exist.

    Overwrites the file if it already exists. Automatically creates parent
    directories as needed. Preserves existing line endings (LF/CRLF).
    Rejects content that contains omission placeholders.

    Args:
        file_path: Path to the file to write (absolute or relative to workspace root).
        content: The complete content to write to the file.

    Returns:
        Success message with a diff snippet, or an error message.
    """
    ctx = write_file.context
    invocation = WriteFileToolInvocation(
        params=WriteFileToolParams(
            file_path=file_path,
            content=content,
        ),
        target_dir=ctx.get("target_dir"),
        config=ctx.get("config"),
    )
    return invocation.execute()
