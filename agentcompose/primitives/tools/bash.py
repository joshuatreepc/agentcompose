"""Bash command execution primitives."""

import asyncio
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional

from agentcompose.core import PrimitiveRegistry

bash = PrimitiveRegistry("bash")

DEFAULT_MAX_OUTPUT_LENGTH = 30_000


@dataclass
class BashResult:
    """Result of a bash command execution."""

    stdout: str
    stderr: str
    exit_code: int
    tee_files: list[dict[str, str]] = field(default_factory=list)


BeforeBashHook = Callable[[str], Optional[str]]
"""Takes a command string, returns a modified command or None to keep original."""

AfterBashHook = Callable[[str, BashResult], Optional[BashResult]]
"""Takes (command, result), returns a modified result or None to keep original."""


def _truncate_output(
    output: str, max_length: int, stream_name: str
) -> str:
    """Truncate output if it exceeds max_length, appending a notice."""
    if len(output) <= max_length:
        return output
    truncated_length = len(output) - max_length
    return (
        f"{output[:max_length]}\n\n"
        f"[{stream_name} truncated: {truncated_length} characters removed]"
    )


@bash.register(kind="tool")
def execute(
    command: str,
    cwd: str = ".",
    timeout: float = 120.0,
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    before_hook: Optional[BeforeBashHook] = None,
    after_hook: Optional[AfterBashHook] = None,
) -> BashResult:
    """Execute a bash command and return stdout, stderr, and exit code.

    Args:
        command: The bash command to execute.
        cwd: Working directory for execution.
        timeout: Maximum seconds to wait before killing the process.
        max_output_length: Max characters for stdout/stderr before truncation.
        before_hook: Optional callback that can modify the command before execution.
        after_hook: Optional callback that can modify the result after execution.

    Returns:
        A BashResult with stdout, stderr, and exit_code.
    """
    if before_hook:
        modified = before_hook(command)
        if modified is not None:
            command = modified

    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = BashResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        result = BashResult(
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            exit_code=124,
        )

    result.stdout = _truncate_output(result.stdout, max_output_length, "stdout")
    result.stderr = _truncate_output(result.stderr, max_output_length, "stderr")

    if after_hook:
        modified = after_hook(command, result)
        if modified is not None:
            result = modified

    return result


@bash.register(kind="tool")
async def execute_async(
    command: str,
    cwd: str = ".",
    timeout: float = 120.0,
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    before_hook: Optional[BeforeBashHook] = None,
    after_hook: Optional[AfterBashHook] = None,
) -> BashResult:
    """Execute a bash command asynchronously and return stdout, stderr, and exit code.

    Args:
        command: The bash command to execute.
        cwd: Working directory for execution.
        timeout: Maximum seconds to wait before killing the process.
        max_output_length: Max characters for stdout/stderr before truncation.
        before_hook: Optional callback that can modify the command before execution.
        after_hook: Optional callback that can modify the result after execution.

    Returns:
        A BashResult with stdout, stderr, and exit_code.
    """
    if before_hook:
        modified = before_hook(command)
        if modified is not None:
            command = modified

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        result = BashResult(
            stdout=stdout_bytes.decode(),
            stderr=stderr_bytes.decode(),
            exit_code=proc.returncode or 0,
        )
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        result = BashResult(
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            exit_code=124,
        )

    result.stdout = _truncate_output(result.stdout, max_output_length, "stdout")
    result.stderr = _truncate_output(result.stderr, max_output_length, "stderr")

    if after_hook:
        modified = after_hook(command, result)
        if modified is not None:
            result = modified

    return result
