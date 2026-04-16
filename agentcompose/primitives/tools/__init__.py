"""Built-in tool primitives for agent workflows."""

from .ask_user import ask_user, AskUserInvocation, Question
from .bash import bash, BashResult
from .files import files, ReadResult, WriteResult
from .glob import glob_tool, GlobToolInvocation, GlobToolParams, GlobEntry
from .grep import grep, GrepToolInvocation, GrepToolParams, GrepMatch
from .modifable_tool import (
    ModifyContext,
    ModifyContentOverrides,
    ModifyResult,
    EditorType,
    is_modifiable_declarative_tool,
    modify_with_editor,
)
from .read_file import read_file, ReadFileToolInvocation, ReadFileToolParams
from .rip_grep import rip_grep, RipGrepToolInvocation, RipGrepToolParams
from .shell import shell, ShellResult, ShellToolInvocation, ShellToolParams, ToolResult
from .web_fetch import web_fetch, WebFetchToolInvocation, WebFetchToolParams
from .web_search import web_search, WebSearchToolInvocation, WebSearchToolParams
from .write_file import write_file, WriteFileToolInvocation, WriteFileToolParams
from .write_todos import write_todos, WriteTodosInvocation, Todo

__all__ = [
    # ask_user
    "ask_user",
    "AskUserInvocation",
    "Question",
    # bash
    "bash",
    "BashResult",
    # files
    "files",
    "ReadResult",
    "WriteResult",
    # glob
    "glob_tool",
    "GlobToolInvocation",
    "GlobToolParams",
    "GlobEntry",
    # grep
    "grep",
    "GrepToolInvocation",
    "GrepToolParams",
    "GrepMatch",
    # modifable_tool
    "ModifyContext",
    "ModifyContentOverrides",
    "ModifyResult",
    "EditorType",
    "is_modifiable_declarative_tool",
    "modify_with_editor",
    # read_file
    "read_file",
    "ReadFileToolInvocation",
    "ReadFileToolParams",
    # rip_grep
    "rip_grep",
    "RipGrepToolInvocation",
    "RipGrepToolParams",
    # shell
    "shell",
    "ShellResult",
    "ShellToolInvocation",
    "ShellToolParams",
    "ToolResult",
    # web_fetch
    "web_fetch",
    "WebFetchToolInvocation",
    "WebFetchToolParams",
    # web_search
    "web_search",
    "WebSearchToolInvocation",
    "WebSearchToolParams",
    # write_file
    "write_file",
    "WriteFileToolInvocation",
    "WriteFileToolParams",
    # write_todos
    "write_todos",
    "WriteTodosInvocation",
    "Todo",
]
