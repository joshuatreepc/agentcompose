"""File read/write primitives."""

import os
from dataclasses import dataclass
from typing import Optional

from agentcompose.core import PrimitiveRegistry

files = PrimitiveRegistry("files")

DEFAULT_MAX_READ_LENGTH = 30_000


@dataclass
class ReadResult:
    """Result of a file read operation."""

    content: str
    path: str
    lines: int
    truncated: bool


@dataclass
class WriteResult:
    """Result of a file write operation."""

    path: str
    bytes_written: int
    created: bool


@files.register(kind="tool")
def read_file(
    path: str,
    offset: int = 0,
    limit: Optional[int] = None,
    max_length: int = DEFAULT_MAX_READ_LENGTH,
) -> ReadResult:
    """Read the contents of a file.

    Args:
        path: Path to the file to read.
        offset: Line number to start reading from (0-based).
        limit: Maximum number of lines to read. None reads all lines.
        max_length: Max characters before truncation.

    Returns:
        A ReadResult with the file content and metadata.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    selected = lines[offset:] if limit is None else lines[offset : offset + limit]
    content = "".join(selected)

    truncated = len(content) > max_length
    if truncated:
        content = (
            f"{content[:max_length]}\n\n"
            f"[truncated: {len(content) - max_length} characters removed]"
        )

    return ReadResult(
        content=content,
        path=os.path.abspath(path),
        lines=len(selected),
        truncated=truncated,
    )


@files.register(kind="tool")
def write_file(
    path: str,
    content: str,
    create_dirs: bool = True,
) -> WriteResult:
    """Write content to a file, creating it if it doesn't exist.

    Args:
        path: Path to the file to write.
        content: The content to write.
        create_dirs: Whether to create parent directories if they don't exist.

    Returns:
        A WriteResult with the path and bytes written.
    """
    created = not os.path.exists(path)

    if create_dirs:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(path, "w") as f:
        bytes_written = f.write(content)

    return WriteResult(
        path=os.path.abspath(path),
        bytes_written=bytes_written,
        created=created,
    )
