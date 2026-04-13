"""Built-in tool primitives for agent workflows."""

from .bash import bash, BashResult
from .files import files, ReadResult, WriteResult
from .read_file import read_file, ReadFileToolInvocation, ReadFileToolParams
from .shell import shell, ShellResult, ShellToolInvocation, ShellToolParams, ToolResult

__all__ = [
    "bash",
    "BashResult",
    "files",
    "ReadResult",
    "WriteResult",
    "read_file",
    "ReadFileToolInvocation",
    "ReadFileToolParams",
    "shell",
    "ShellResult",
    "ShellToolInvocation",
    "ShellToolParams",
    "ToolResult",
]
