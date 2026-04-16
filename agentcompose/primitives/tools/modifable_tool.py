"""
Modifiable tool support.

Ported from the Google ADK TypeScript modifiable-tool module.
Provides a protocol for tools that support a "modify" operation — launching
an external diff editor so the user can review and edit proposed content
before it is applied.

Workflow:
1. The tool computes proposed content (e.g. a file write or edit).
2. Current and proposed content are written to secure temp files.
3. An external diff editor is opened for the user.
4. After the user saves and closes, the (possibly modified) content is read
   back and used to build updated tool parameters.
5. Temp files are cleaned up.
"""

from __future__ import annotations

import difflib
import logging
import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Editor types
# ---------------------------------------------------------------------------


class EditorType(str, Enum):
    """Supported external diff editor types."""

    VSCODE = "vscode"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ModifyContext(Protocol[T]):
    """Protocol that tools implement to support the modify workflow.

    The type parameter ``T`` is the tool's own params type (e.g.
    ``WriteFileToolParams``).
    """

    def get_file_path(self, params: T) -> str:
        """Return the file path that this invocation targets."""
        ...

    async def get_current_content(self, params: T) -> str:
        """Return the current on-disk content of the file."""
        ...

    async def get_proposed_content(self, params: T) -> str:
        """Return the content the tool *would* write without user review."""
        ...

    def create_updated_params(
        self,
        old_content: str,
        modified_proposed_content: str,
        original_params: T,
    ) -> T:
        """Build new tool params that reflect the user's edits."""
        ...


class ModifiableDeclarativeTool(Protocol[T]):
    """A declarative tool that supports a modify (user-review) operation."""

    def get_modify_context(self) -> ModifyContext[T]:
        """Return the ModifyContext for this tool."""
        ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModifyResult(Generic[T]):
    """Result of the modify-with-editor workflow."""

    updated_params: T
    updated_diff: str


_UNSET = object()


@dataclass
class ModifyContentOverrides:
    """Optional overrides for the modify workflow.

    Useful for testing or when the caller already has the content and
    doesn't need the ModifyContext to fetch it.

    Fields default to ``_UNSET`` (not provided). Setting a field to ``None``
    means "override with empty string" — matching the TypeScript behaviour
    where ``currentContent: null`` is a valid override.
    """

    current_content: Any = _UNSET
    proposed_content: Any = _UNSET


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------


def is_modifiable_declarative_tool(tool: Any) -> bool:
    """Check if a tool object implements the modifiable protocol."""
    return hasattr(tool, "get_modify_context") and callable(
        tool.get_modify_context
    )


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------


@dataclass
class _TempFiles:
    old_path: str
    new_path: str
    dir_path: str


def _create_temp_files(
    current_content: str,
    proposed_content: str,
    file_path: str,
) -> _TempFiles:
    """Write current and proposed content to secure temp files for diffing."""
    diff_dir = tempfile.mkdtemp(prefix="agentcompose-tool-modify-")

    try:
        os.chmod(diff_dir, stat.S_IRWXU)  # 0o700
    except OSError:
        logger.error(
            "Error setting permissions on temp diff directory: %s", diff_dir
        )
        raise

    ext = Path(file_path).suffix
    stem = Path(file_path).stem
    timestamp = int(time.time() * 1000)

    old_path = os.path.join(
        diff_dir, f"agentcompose-modify-{stem}-old-{timestamp}{ext}"
    )
    new_path = os.path.join(
        diff_dir, f"agentcompose-modify-{stem}-new-{timestamp}{ext}"
    )

    for fpath, content in ((old_path, current_content), (new_path, proposed_content)):
        fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)

    return _TempFiles(old_path=old_path, new_path=new_path, dir_path=diff_dir)


def _delete_temp_files(temps: _TempFiles) -> None:
    """Best-effort cleanup of temp diff files and directory."""
    for fpath in (temps.old_path, temps.new_path):
        try:
            os.unlink(fpath)
        except OSError:
            logger.error("Error deleting temp diff file: %s", fpath)

    try:
        os.rmdir(temps.dir_path)
    except OSError:
        logger.error("Error deleting temp diff directory: %s", temps.dir_path)


# ---------------------------------------------------------------------------
# Diff generation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# External editor launch
# ---------------------------------------------------------------------------


def _open_diff(old_path: str, new_path: str, editor_type: EditorType) -> None:
    """Open an external diff editor and block until the user closes it.

    Raises:
        FileNotFoundError: If the editor binary is not found.
        subprocess.CalledProcessError: If the editor exits with an error.
    """
    if editor_type == EditorType.VSCODE:
        subprocess.run(
            ["code", "--wait", "--diff", old_path, new_path],
            check=True,
        )
    else:
        # Fall back to $EDITOR, then vimdiff, then diff
        editor = os.environ.get("EDITOR")
        if editor and "vim" in editor:
            subprocess.run(
                [editor, "-d", old_path, new_path],
                check=True,
            )
        elif editor:
            subprocess.run(
                [editor, old_path, new_path],
                check=True,
            )
        else:
            # Try vimdiff, fall back to printing the diff
            try:
                subprocess.run(
                    ["vimdiff", old_path, new_path],
                    check=True,
                )
            except FileNotFoundError:
                diff_output = create_unified_diff(
                    os.path.basename(new_path),
                    open(old_path).read(),
                    open(new_path).read(),
                )
                print(diff_output)
                print(
                    f"\nNo diff editor found. Edit the proposed file directly:\n"
                    f"  {new_path}\n"
                    f"Press Enter when done..."
                )
                input()


# ---------------------------------------------------------------------------
# Updated params extraction
# ---------------------------------------------------------------------------


def _get_updated_params(
    old_path: str,
    new_path: str,
    original_params: T,
    modify_context: ModifyContext[T],
) -> ModifyResult[T]:
    """Read back temp files after user editing, build updated params and diff."""
    try:
        old_content = open(old_path, "r", encoding="utf-8").read()
    except FileNotFoundError:
        old_content = ""

    try:
        new_content = open(new_path, "r", encoding="utf-8").read()
    except FileNotFoundError:
        new_content = ""

    updated_params = modify_context.create_updated_params(
        old_content, new_content, original_params,
    )

    file_name = os.path.basename(
        modify_context.get_file_path(original_params)
    )
    updated_diff = create_unified_diff(file_name, old_content, new_content)

    return ModifyResult(
        updated_params=updated_params,
        updated_diff=updated_diff,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def modify_with_editor(
    original_params: T,
    modify_context: ModifyContext[T],
    editor_type: EditorType = EditorType.SYSTEM,
    overrides: Optional[ModifyContentOverrides] = None,
) -> ModifyResult[T]:
    """Launch an external diff editor for the user to review proposed changes.

    This is the primary entry point for the modify workflow:
    1. Fetches current and proposed content (or uses overrides).
    2. Writes both to secure temp files.
    3. Opens the external diff editor and waits for the user.
    4. Reads back the (possibly modified) proposed content.
    5. Builds updated tool params and a unified diff.
    6. Cleans up temp files.

    Args:
        original_params: The tool's original parameters.
        modify_context: Protocol implementation from the tool.
        editor_type: Which diff editor to launch.
        overrides: Optional pre-fetched content (skips async fetch).

    Returns:
        A ModifyResult with the updated params and a unified diff string.
    """
    current_content: str
    if overrides is not None and overrides.current_content is not _UNSET:
        current_content = overrides.current_content or ""
    else:
        current_content = await modify_context.get_current_content(original_params)

    proposed_content: str
    if overrides is not None and overrides.proposed_content is not _UNSET:
        proposed_content = overrides.proposed_content or ""
    else:
        proposed_content = await modify_context.get_proposed_content(original_params)

    temps = _create_temp_files(
        current_content,
        proposed_content,
        modify_context.get_file_path(original_params),
    )

    try:
        _open_diff(temps.old_path, temps.new_path, editor_type)
        return _get_updated_params(
            temps.old_path,
            temps.new_path,
            original_params,
            modify_context,
        )
    finally:
        _delete_temp_files(temps)
