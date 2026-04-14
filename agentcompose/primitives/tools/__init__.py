"""Built-in tool primitives for agent workflows."""

from .ask_user import ask_user, AskUserInvocation, Question
from .bash import bash, BashResult
from .files import files, ReadResult, WriteResult
from .grep import grep, GrepToolInvocation, GrepToolParams, GrepMatch
from .read_file import read_file, ReadFileToolInvocation, ReadFileToolParams
from .shell import shell, ShellResult, ShellToolInvocation, ShellToolParams, ToolResult
from .web_fetch import web_fetch, WebFetchToolInvocation, WebFetchToolParams
from .write_todos import write_todos, WriteTodosInvocation, Todo

__all__ = [
    "ask_user",
    "AskUserInvocation",
    "Question",
    "bash",
    "BashResult",
    "files",
    "ReadResult",
    "WriteResult",
    "grep",
    "GrepToolInvocation",
    "GrepToolParams",
    "GrepMatch",
    "read_file",
    "ReadFileToolInvocation",
    "ReadFileToolParams",
    "shell",
    "ShellResult",
    "ShellToolInvocation",
    "ShellToolParams",
    "ToolResult",
    "web_fetch",
    "WebFetchToolInvocation",
    "WebFetchToolParams",
    "write_todos",
    "WriteTodosInvocation",
    "Todo",
]
