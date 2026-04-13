"""
Full-featured shell execution primitive.

Ported from the Google ADK TypeScript ShellTool/ShellToolInvocation.
Provides sandboxed command execution with:
- Streaming output with configurable update intervals
- Background process support with PID tracking via pgrep
- Inactivity-based timeout with per-chunk reset
- Pluggable sandbox permissions and confirmation flows
- Path validation and access control
- Command parsing for root command extraction
- Proactive permission suggestions for known tools
- Output formatting for both LLM consumption and human display
- Sandbox denial detection and expansion requests
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal as signal_mod
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

shell = PrimitiveRegistry("shell")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_UPDATE_INTERVAL_S = 1.0
BACKGROUND_DELAY_S = 0.2
DEFAULT_INACTIVITY_TIMEOUT_S = 120.0
DEFAULT_MAX_OUTPUT_LENGTH = 30_000
SHOW_NL_DESCRIPTION_THRESHOLD = 150


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfirmationOutcome(Enum):
    """Possible outcomes of a tool confirmation prompt."""

    CANCEL = auto()
    PROCEED_ONCE = auto()
    PROCEED_ALWAYS = auto()
    PROCEED_ALWAYS_AND_SAVE = auto()


class ToolErrorType(str, Enum):
    """Classification of tool errors."""

    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    SHELL_EXECUTE_ERROR = "shell_execute_error"
    SANDBOX_EXPANSION_REQUIRED = "sandbox_expansion_required"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SandboxPermissions:
    """Permissions that can be granted to a sandboxed command."""

    network: bool = False
    file_system: Optional[FileSystemPermissions] = None


@dataclass
class FileSystemPermissions:
    """Read/write path lists for sandbox file system access."""

    read: list[str] = field(default_factory=list)
    write: list[str] = field(default_factory=list)


@dataclass
class ShellToolParams:
    """Parameters for a shell tool invocation."""

    command: str
    description: Optional[str] = None
    dir_path: Optional[str] = None
    is_background: bool = False
    delay_s: Optional[float] = None
    additional_permissions: Optional[SandboxPermissions] = None


@dataclass
class ToolError:
    """Structured error from a tool execution."""

    message: str
    type: ToolErrorType


@dataclass
class BackgroundExecutionData:
    """Metadata about a backgrounded or failed command."""

    pid: Optional[int] = None
    command: Optional[str] = None
    initial_output: Optional[str] = None
    exit_code: Optional[int] = None
    is_error: bool = False


@dataclass
class ShellResult:
    """Full result of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int
    pid: Optional[int] = None
    signal_name: Optional[str] = None
    aborted: bool = False
    backgrounded: bool = False
    background_pids: list[int] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result formatted for consumption by an LLM agent and human display."""

    llm_content: str
    return_display: str = ""
    error: Optional[ToolError] = None
    data: Optional[BackgroundExecutionData] = None


@dataclass
class ConfirmationDetails:
    """Details presented to the user when confirmation is required."""

    type: str  # "exec" or "sandbox_expansion"
    title: str
    command: str
    root_command: str
    root_commands: list[str] = field(default_factory=list)
    additional_permissions: Optional[SandboxPermissions] = None


@dataclass
class PolicyUpdateOptions:
    """Options for updating the command allow-list after confirmation."""

    command_prefix: list[str] | str = field(default_factory=list)
    allow_redirection: Optional[bool] = None


@dataclass
class CommandDetail:
    """Parsed information about a single command in a pipeline."""

    name: str
    args: list[str] = field(default_factory=list)


@dataclass
class ParsedCommand:
    """Result of parsing a shell command string."""

    details: list[CommandDetail] = field(default_factory=list)
    has_error: bool = False


# ---------------------------------------------------------------------------
# Protocols — plug in your own sandbox, confirmation bus, or config
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxManager(Protocol):
    """Protocol for sandbox implementations that can parse denial messages."""

    def parse_denials(self, result: ShellResult) -> Optional[SandboxPermissions]:
        """Inspect a failed command result and return permissions that were denied."""
        ...


@runtime_checkable
class ConfirmationBus(Protocol):
    """Protocol for confirmation UIs (CLI prompt, web dialog, etc.)."""

    async def confirm(self, details: ConfirmationDetails) -> ConfirmationOutcome:
        """Present confirmation to the user and return their decision."""
        ...


@runtime_checkable
class ShellConfig(Protocol):
    """Protocol for shell configuration providers."""

    def get_target_dir(self) -> str: ...
    def get_sandbox_enabled(self) -> bool: ...
    def get_inactivity_timeout(self) -> float: ...

    def validate_path_access(self, path: str) -> Optional[str]:
        """Return an error message if path is not accessible, or None if OK."""
        ...

    def is_path_allowed(self, path: str) -> bool: ...


OutputCallback = Callable[[str], None]
"""Called with cumulative output as it arrives."""


# ---------------------------------------------------------------------------
# Command parsing utilities
# ---------------------------------------------------------------------------

# Patterns that indicate redirections
_REDIRECTION_PATTERN = re.compile(r"(?<!\w)[012]?>{1,2}|<{1,2}")

# Shell operators that separate commands
_COMMAND_SEPARATORS = re.compile(r"\s*(?:&&|\|\|?|;)\s*")


def has_redirection(command: str) -> bool:
    """Check whether a command string contains shell redirections."""
    return bool(_REDIRECTION_PATTERN.search(command))


def strip_shell_wrapper(command: str) -> str:
    """Remove common shell wrappers (env vars, subshell parens, etc.)."""
    stripped = command.strip()
    # Remove leading environment variable assignments like VAR=val
    while re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+", stripped):
        stripped = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+", "", stripped, count=1)
    return stripped


def get_command_roots(command: str) -> list[str]:
    """Extract the root command names from a (possibly compound) command string.

    For ``'cd /tmp && npm install | tee log.txt'`` returns ``['cd', 'npm', 'tee']``.
    """
    stripped = strip_shell_wrapper(command)
    parts = _COMMAND_SEPARATORS.split(stripped)
    roots: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            tokens = shlex.split(part)
        except ValueError:
            tokens = part.split()
        if tokens:
            # Skip env-var-style prefixes
            for tok in tokens:
                if "=" in tok and tok.split("=")[0].isidentifier():
                    continue
                roots.append(os.path.basename(tok))
                break
    return roots


def normalize_command(command: str) -> str:
    """Normalize a command name (strip path, lowercase)."""
    return os.path.basename(command).lower()


def parse_command_details(command: str) -> ParsedCommand:
    """Parse a command string into structured details per pipeline stage."""
    stripped = strip_shell_wrapper(command)
    parts = _COMMAND_SEPARATORS.split(stripped)
    details: list[CommandDetail] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            tokens = shlex.split(part)
        except ValueError:
            details.append(CommandDetail(name=part.split()[0] if part.split() else part))
            continue
        if tokens:
            details.append(CommandDetail(name=tokens[0], args=tokens[1:]))
    return ParsedCommand(details=details)


# ---------------------------------------------------------------------------
# Proactive permission suggestions
# ---------------------------------------------------------------------------

# Known commands that typically require network access
_NETWORK_COMMANDS: dict[str, Optional[set[str]]] = {
    "npm": {"install", "ci", "update", "publish", "pack", "audit", "fund"},
    "yarn": {"install", "add", "upgrade", "publish"},
    "pnpm": {"install", "add", "update", "publish"},
    "pip": {"install", "download"},
    "pip3": {"install", "download"},
    "uv": {"pip", "sync", "lock", "add"},
    "cargo": {"build", "install", "update", "fetch"},
    "go": {"get", "install", "mod"},
    "curl": None,  # always needs network
    "wget": None,
    "git": {"clone", "fetch", "pull", "push"},
    "docker": {"pull", "push", "build"},
    "brew": {"install", "update", "upgrade"},
    "apt": {"install", "update", "upgrade"},
    "apt-get": {"install", "update", "upgrade"},
}


def is_network_reliant_command(
    root_command: str, sub_command: Optional[str] = None
) -> bool:
    """Check if a command (and optionally its sub-command) typically needs network."""
    normalized = normalize_command(root_command)
    if normalized not in _NETWORK_COMMANDS:
        return False
    sub_commands = _NETWORK_COMMANDS[normalized]
    if sub_commands is None:
        return True
    return sub_command is not None and sub_command in sub_commands


def get_proactive_permissions(root_command: str) -> Optional[SandboxPermissions]:
    """Return suggested sandbox permissions for known commands, or None."""
    normalized = normalize_command(root_command)
    if normalized in _NETWORK_COMMANDS:
        return SandboxPermissions(network=True)
    return None


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

# Directories that should never be auto-expanded into
_SENSITIVE_DIRS: set[str] = {
    os.path.expanduser("~"),
    str(Path.home().parent),
    os.sep,
    os.path.join(os.sep, "etc"),
    os.path.join(os.sep, "usr"),
    os.path.join(os.sep, "var"),
    os.path.join(os.sep, "bin"),
    os.path.join(os.sep, "sbin"),
    os.path.join(os.sep, "lib"),
    os.path.join(os.sep, "root"),
    os.path.join(os.sep, "home"),
    os.path.join(os.sep, "Users"),
}


def is_subpath(parent: str, child: str) -> bool:
    """Return True if *child* is equal to or nested under *parent*."""
    try:
        Path(child).relative_to(parent)
        return True
    except ValueError:
        return False


def simplify_paths(paths: set[str]) -> list[str]:
    """De-duplicate and consolidate a set of paths.

    - Removes redundant child paths already covered by a parent.
    - Collapses clusters of 3+ siblings into their shared parent directory
      (unless the parent is a sensitive system directory).
    """
    if not paths:
        return []

    raw = sorted(paths, key=len)

    # 1. Remove redundant sub-paths
    non_redundant: list[str] = []
    for p in raw:
        if not any(is_subpath(s, p) for s in non_redundant):
            non_redundant.append(p)

    # 2. Consolidate clusters of siblings sharing a parent
    parent_groups: dict[str, list[str]] = {}
    for p in non_redundant:
        parent = os.path.dirname(p)
        parent_groups.setdefault(parent, []).append(p)

    final: set[str] = set()
    for parent, children in parent_groups.items():
        if len(children) >= 3 and len(parent) > 1 and parent not in _SENSITIVE_DIRS:
            final.add(parent)
        else:
            final.update(children)

    # 3. Final redundancy pass after consolidation
    result: list[str] = []
    for p in sorted(final, key=len):
        if not any(is_subpath(s, p) for s in result):
            result.append(p)

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def truncate_output(output: str, max_length: int, stream_name: str) -> str:
    """Truncate output exceeding *max_length*, appending a notice."""
    if len(output) <= max_length:
        return output
    removed = len(output) - max_length
    return f"{output[:max_length]}\n\n[{stream_name} truncated: {removed} characters removed]"


def format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# pgrep helpers
# ---------------------------------------------------------------------------


def _wrap_command_for_pgrep(command: str, temp_path: str) -> str:
    """Wrap a command in a subshell to capture background PIDs via pgrep."""
    trimmed = command.strip()
    if not trimmed:
        return ""
    if trimmed.endswith("\\"):
        trimmed += " "
    return (
        f"(\n{trimmed}\n); "
        f"__code=$?; pgrep -g 0 >{temp_path} 2>&1; exit $__code;"
    )


def _read_background_pids(temp_path: str, own_pid: Optional[int]) -> list[int]:
    """Read background PIDs written by the pgrep wrapper."""
    pids: list[int] = []
    try:
        with open(temp_path) as f:
            for line in f:
                line = line.strip()
                if not line or not line.isdigit():
                    continue
                pid = int(line)
                if pid != own_pid:
                    pids.append(pid)
    except FileNotFoundError:
        pass
    return pids


# ---------------------------------------------------------------------------
# ShellToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class ShellToolInvocation:
    """Manages a single shell command execution with full lifecycle support.

    Mirrors the TypeScript ShellToolInvocation class. Handles:
    - Pre-execution validation (empty command, path access)
    - Sandbox permission checks and proactive expansion
    - Confirmation flow via a pluggable ConfirmationBus
    - Streaming output with inactivity timeout
    - Background process management with PID tracking
    - Post-execution sandbox denial detection
    - LLM-formatted and human-formatted result output
    """

    def __init__(
        self,
        params: ShellToolParams,
        *,
        target_dir: Optional[str] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        confirmation_bus: Optional[ConfirmationBus] = None,
        config: Optional[ShellConfig] = None,
        max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    ):
        self.params = params
        self.target_dir = target_dir or os.getcwd()
        self.sandbox_manager = sandbox_manager
        self.confirmation_bus = confirmation_bus
        self.config = config
        self.max_output_length = max_output_length
        self._proactive_permissions_confirmed: Optional[SandboxPermissions] = None

    # -- Display helpers ---------------------------------------------------

    def get_contextual_details(self) -> str:
        """Build a human-readable context string for this invocation."""
        parts: list[str] = []
        if self.params.dir_path:
            parts.append(f"[in {self.params.dir_path}]")
        else:
            parts.append(f"[current working directory {self.target_dir}]")
        if self.params.description:
            parts.append(f"({self.params.description.replace(chr(10), ' ')})")
        if self.params.is_background:
            parts.append("[background]")
        return " ".join(parts)

    def get_display_title(self) -> str:
        return self.params.command

    def get_description(self) -> str:
        desc = (self.params.description or "").strip()
        if len(self.params.command) <= SHOW_NL_DESCRIPTION_THRESHOLD or not desc:
            return self.params.command
        return desc

    # -- Policy / confirmation helpers ------------------------------------

    def get_policy_update_options(
        self, outcome: ConfirmationOutcome
    ) -> Optional[PolicyUpdateOptions]:
        """Compute policy updates to apply after the user confirms."""
        if outcome not in (
            ConfirmationOutcome.PROCEED_ALWAYS,
            ConfirmationOutcome.PROCEED_ALWAYS_AND_SAVE,
        ):
            return None

        command = strip_shell_wrapper(self.params.command)
        root_commands = list(dict.fromkeys(get_command_roots(command)))
        allow_redirection = True if has_redirection(command) else None

        if root_commands:
            return PolicyUpdateOptions(
                command_prefix=root_commands, allow_redirection=allow_redirection
            )
        return PolicyUpdateOptions(
            command_prefix=self.params.command, allow_redirection=allow_redirection
        )

    def get_confirmation_details(
        self,
        proactive_permissions: Optional[SandboxPermissions] = None,
    ) -> ConfirmationDetails:
        """Build confirmation details for the user prompt."""
        command = strip_shell_wrapper(self.params.command)
        parsed = parse_command_details(command)

        if parsed.has_error or not parsed.details:
            fallback = command.strip().split()[0] if command.strip() else "shell command"
            root_display = fallback
            if has_redirection(command):
                root_display += ", redirection"
        else:
            root_display = ", ".join(d.name for d in parsed.details)

        root_commands = list(dict.fromkeys(get_command_roots(command)))

        effective_perms = (
            self.params.additional_permissions or proactive_permissions
        )

        if effective_perms:
            title = (
                "Sandbox Expansion Request (Recommended)"
                if proactive_permissions
                else "Sandbox Expansion Request"
            )
            return ConfirmationDetails(
                type="sandbox_expansion",
                title=title,
                command=self.params.command,
                root_command=root_display,
                root_commands=root_commands,
                additional_permissions=effective_perms,
            )

        return ConfirmationDetails(
            type="exec",
            title="Confirm Shell Command",
            command=self.params.command,
            root_command=root_display,
            root_commands=root_commands,
        )

    async def should_confirm(self) -> Optional[ConfirmationDetails]:
        """Determine whether this invocation needs user confirmation.

        Returns ConfirmationDetails if confirmation is needed, None otherwise.
        Checks for explicit additional_permissions first, then proactive
        expansion for known network-heavy commands.
        """
        if self.params.additional_permissions:
            return self.get_confirmation_details()

        if self.config and self.config.get_sandbox_enabled():
            command = strip_shell_wrapper(self.params.command)
            roots = get_command_roots(command)
            if roots:
                root_command = normalize_command(roots[0])
                proactive = get_proactive_permissions(root_command)
                if proactive:
                    parsed = parse_command_details(command)
                    sub_command = (
                        parsed.details[0].args[0]
                        if parsed.details and parsed.details[0].args
                        else None
                    )
                    if is_network_reliant_command(root_command, sub_command):
                        return self.get_confirmation_details(proactive)

        return None

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution.

        Returns an error message string, or None if valid.
        """
        if not self.params.command.strip():
            return "Command cannot be empty."

        if self.params.dir_path and self.config:
            resolved = os.path.join(self.target_dir, self.params.dir_path)
            resolved = os.path.abspath(resolved)
            return self.config.validate_path_access(resolved)

        return None

    # -- Main execution ----------------------------------------------------

    async def execute(
        self,
        *,
        on_output: Optional[OutputCallback] = None,
        abort_event: Optional[asyncio.Event] = None,
    ) -> ToolResult:
        """Execute the shell command and return a formatted ToolResult.

        Args:
            on_output: Callback for streaming output updates.
            abort_event: If set, the command is cancelled.

        Returns:
            ToolResult with llm_content and return_display.
        """
        # Validate first
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

        # Confirmation flow
        if self.confirmation_bus:
            confirm_details = await self.should_confirm()
            if confirm_details:
                outcome = await self.confirmation_bus.confirm(confirm_details)
                if outcome == ConfirmationOutcome.CANCEL:
                    return ToolResult(
                        llm_content="Command was cancelled by user.",
                        return_display="Command cancelled by user.",
                    )
                if confirm_details.additional_permissions:
                    self._proactive_permissions_confirmed = (
                        confirm_details.additional_permissions
                    )

        command = strip_shell_wrapper(self.params.command)

        # Resolve working directory
        if self.params.dir_path:
            cwd = os.path.abspath(os.path.join(self.target_dir, self.params.dir_path))
        else:
            cwd = self.target_dir

        # Set up pgrep temp file for background PID detection
        temp_fd, temp_path = tempfile.mkstemp(prefix="shell_pgrep_", suffix=".tmp")
        os.close(temp_fd)
        wrapped = _wrap_command_for_pgrep(command, temp_path)

        timeout = (
            self.config.get_inactivity_timeout()
            if self.config
            else DEFAULT_INACTIVITY_TIMEOUT_S
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", wrapped,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

            assert proc.stdout is not None
            assert proc.stderr is not None

            pid = proc.pid
            cumulative_stdout = ""
            stderr_chunks: list[str] = []
            aborted = False
            is_binary = False
            bytes_received = 0

            async def _read_stream(
                stream: asyncio.StreamReader,
                is_stdout: bool,
            ) -> None:
                nonlocal cumulative_stdout, aborted, is_binary, bytes_received
                last_update = asyncio.get_event_loop().time()

                while True:
                    # Check abort
                    if abort_event and abort_event.is_set():
                        aborted = True
                        try:
                            os.killpg(os.getpgid(proc.pid), signal_mod.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            proc.kill()
                        return

                    # Read with inactivity timeout
                    try:
                        chunk = await asyncio.wait_for(
                            stream.read(4096), timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        aborted = True
                        try:
                            os.killpg(os.getpgid(proc.pid), signal_mod.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            proc.kill()
                        return

                    if not chunk:
                        break

                    # Binary detection: check for null bytes in first chunk
                    if is_stdout and not cumulative_stdout and b"\x00" in chunk[:512]:
                        is_binary = True
                        bytes_received += len(chunk)
                        if on_output:
                            on_output("[Binary output detected. Halting stream...]")
                        continue

                    if is_binary and is_stdout:
                        bytes_received += len(chunk)
                        now = asyncio.get_event_loop().time()
                        if on_output and (now - last_update >= OUTPUT_UPDATE_INTERVAL_S):
                            on_output(
                                f"[Receiving binary output... "
                                f"{format_bytes(bytes_received)} received]"
                            )
                            last_update = now
                        continue

                    text = chunk.decode(errors="replace")
                    if is_stdout:
                        cumulative_stdout += text
                        now = asyncio.get_event_loop().time()
                        if on_output and (now - last_update >= OUTPUT_UPDATE_INTERVAL_S):
                            on_output(cumulative_stdout)
                            last_update = now
                    else:
                        stderr_chunks.append(text)

            # Background mode: wait briefly then return if still running
            if self.params.is_background:
                stdout_task = asyncio.create_task(
                    _read_stream(proc.stdout, is_stdout=True)
                )
                stderr_task = asyncio.create_task(
                    _read_stream(proc.stderr, is_stdout=False)
                )

                delay = self.params.delay_s or BACKGROUND_DELAY_S
                await asyncio.wait(
                    {stdout_task, stderr_task},
                    timeout=delay,
                )

                if proc.returncode is None:
                    # Still running — return early with background info
                    bg_msg = (
                        f"Command is running in background. PID: {pid}. "
                        f"Initial output:\n{cumulative_stdout}"
                    )
                    return ToolResult(
                        llm_content=bg_msg,
                        return_display=f"Background process started with PID {pid}.",
                        data=BackgroundExecutionData(
                            pid=pid,
                            command=self.params.command,
                            initial_output=cumulative_stdout,
                        ),
                    )

                # Command completed within the delay
                await asyncio.gather(stdout_task, stderr_task)
            else:
                # Foreground: read until completion
                await asyncio.gather(
                    _read_stream(proc.stdout, is_stdout=True),
                    _read_stream(proc.stderr, is_stdout=False),
                )

            await proc.wait()

            # Final output callback
            if on_output and cumulative_stdout:
                on_output(cumulative_stdout)

            # Build the raw ShellResult
            signal_name = None
            exit_code = proc.returncode or 0
            if proc.returncode is not None and proc.returncode < 0:
                try:
                    signal_name = signal_mod.Signals(-proc.returncode).name
                except (ValueError, AttributeError):
                    signal_name = f"signal {-proc.returncode}"

            background_pids = _read_background_pids(temp_path, pid)

            stderr_text = "".join(stderr_chunks)
            shell_result = ShellResult(
                stdout=truncate_output(
                    cumulative_stdout, self.max_output_length, "stdout"
                ),
                stderr=truncate_output(
                    stderr_text, self.max_output_length, "stderr"
                ),
                exit_code=exit_code,
                pid=pid,
                signal_name=signal_name,
                aborted=aborted,
                background_pids=background_pids,
            )

            # -- Format the ToolResult --

            timeout_message = ""
            data: Optional[BackgroundExecutionData] = None

            # LLM content
            if shell_result.aborted:
                if not (abort_event and abort_event.is_set()):
                    timeout_message = (
                        f"Command was automatically cancelled because it exceeded "
                        f"the timeout of {timeout / 60:.1f} minutes without output."
                    )
                    llm_content = timeout_message
                else:
                    llm_content = "Command was cancelled by user before it could complete."

                if shell_result.stdout.strip():
                    llm_content += (
                        f" Below is the output before it was cancelled:\n"
                        f"{shell_result.stdout}"
                    )
                else:
                    llm_content += " There was no output before it was cancelled."

            elif self.params.is_background or shell_result.backgrounded:
                llm_content = (
                    f"Command moved to background (PID: {shell_result.pid}). "
                    f"Output hidden."
                )
                data = BackgroundExecutionData(
                    pid=shell_result.pid,
                    command=self.params.command,
                    initial_output=shell_result.stdout,
                )
            else:
                parts = [f"Output: {shell_result.stdout or '(empty)'}"]

                if shell_result.stderr:
                    parts.append(f"Error: {shell_result.stderr}")

                if shell_result.exit_code != 0:
                    parts.append(f"Exit Code: {shell_result.exit_code}")
                    data = BackgroundExecutionData(
                        exit_code=shell_result.exit_code, is_error=True
                    )

                if shell_result.signal_name:
                    parts.append(f"Signal: {shell_result.signal_name}")

                if shell_result.background_pids:
                    parts.append(
                        f"Background PIDs: "
                        f"{', '.join(str(p) for p in shell_result.background_pids)}"
                    )

                if shell_result.pid:
                    parts.append(f"Process Group PGID: {shell_result.pid}")

                llm_content = "\n".join(parts)

            # Human display
            if self.params.is_background or shell_result.backgrounded:
                return_display = (
                    f"Command moved to background (PID: {shell_result.pid}). "
                    f"Output hidden."
                )
            elif shell_result.aborted:
                cancel_msg = timeout_message or "Command cancelled by user."
                if shell_result.stdout.strip():
                    return_display = (
                        f"{cancel_msg}\n\nOutput before cancellation:\n"
                        f"{shell_result.stdout}"
                    )
                else:
                    return_display = cancel_msg
            elif shell_result.stdout.strip():
                return_display = shell_result.stdout
            elif shell_result.signal_name:
                return_display = (
                    f"Command terminated by signal: {shell_result.signal_name}"
                )
            elif shell_result.stderr:
                return_display = f"Command failed: {shell_result.stderr}"
            elif shell_result.exit_code != 0:
                return_display = (
                    f"Command exited with code: {shell_result.exit_code}"
                )
            else:
                return_display = ""

            # -- Sandbox denial detection --
            error: Optional[ToolError] = None

            if shell_result.stderr:
                error = ToolError(
                    message=shell_result.stderr,
                    type=ToolErrorType.SHELL_EXECUTE_ERROR,
                )

            if self.sandbox_manager and (
                shell_result.stderr
                or shell_result.signal_name
                or shell_result.exit_code != 0
                or shell_result.aborted
            ):
                denial = self.sandbox_manager.parse_denials(shell_result)
                if denial:
                    expansion = self._build_expansion_request(
                        denial, shell_result
                    )
                    if expansion:
                        return ToolResult(
                            llm_content="Sandbox expansion required",
                            return_display=return_display,
                            error=ToolError(
                                message=str(expansion),
                                type=ToolErrorType.SANDBOX_EXPANSION_REQUIRED,
                            ),
                        )

            return ToolResult(
                llm_content=llm_content,
                return_display=return_display,
                error=error,
                data=data,
            )

        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _build_expansion_request(
        self,
        denial: SandboxPermissions,
        result: ShellResult,
    ) -> Optional[ConfirmationDetails]:
        """Inspect a sandbox denial and build an expansion request if new perms needed."""
        command = strip_shell_wrapper(self.params.command)
        roots = get_command_roots(command)
        root_display = roots[0] if roots else "shell"

        read_paths: set[str] = set()
        write_paths: set[str] = set()

        if self.params.additional_permissions:
            ap = self.params.additional_permissions
            if ap.file_system:
                read_paths.update(ap.file_system.read)
                write_paths.update(ap.file_system.write)

        # Add proactive suggestions
        proactive = get_proactive_permissions(root_display)
        if proactive and proactive.file_system:
            read_paths.update(proactive.file_system.read)
            write_paths.update(proactive.file_system.write)

        # Resolve denial file paths
        if denial.file_system:
            for p in denial.file_system.read:
                resolved = self._resolve_denial_path(p)
                if resolved:
                    read_paths.add(resolved)
            for p in denial.file_system.write:
                resolved = self._resolve_denial_path(p)
                if resolved:
                    write_paths.add(resolved)

        simplified_read = simplify_paths(read_paths)
        simplified_write = simplify_paths(write_paths)

        # Check if we actually have new permissions beyond what was already requested
        orig_read = len(
            (self.params.additional_permissions.file_system.read
             if self.params.additional_permissions and self.params.additional_permissions.file_system
             else [])
        )
        orig_write = len(
            (self.params.additional_permissions.file_system.write
             if self.params.additional_permissions and self.params.additional_permissions.file_system
             else [])
        )
        orig_network = bool(
            self.params.additional_permissions and self.params.additional_permissions.network
        )

        has_new = (
            len(simplified_read) > orig_read
            or len(simplified_write) > orig_write
            or (denial.network and not orig_network)
        )

        if not has_new:
            return None

        return ConfirmationDetails(
            type="sandbox_expansion",
            title="Sandbox Expansion Request",
            command=self.params.command,
            root_command=root_display,
            additional_permissions=SandboxPermissions(
                network=denial.network or orig_network,
                file_system=FileSystemPermissions(
                    read=simplified_read,
                    write=simplified_write,
                )
                if simplified_read or simplified_write
                else None,
            ),
        )

    @staticmethod
    def _resolve_denial_path(p: str) -> Optional[str]:
        """Resolve a denied path to an existing parent directory."""
        current = os.path.expanduser(p)
        try:
            if os.path.exists(current) and os.path.isfile(current):
                current = os.path.dirname(current)
        except OSError:
            pass
        while len(current) > 1:
            if os.path.exists(current):
                return current
            current = os.path.dirname(current)
        return None


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@shell.register(kind="tool")
async def execute(
    command: str,
    cwd: Optional[str] = None,
    description: Optional[str] = None,
    is_background: bool = False,
    timeout: float = DEFAULT_INACTIVITY_TIMEOUT_S,
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH,
    on_output: Optional[OutputCallback] = None,
    abort_event: Optional[asyncio.Event] = None,
    sandbox_manager: Optional[SandboxManager] = None,
    confirmation_bus: Optional[ConfirmationBus] = None,
    config: Optional[ShellConfig] = None,
) -> ToolResult:
    """Execute a shell command with full lifecycle support.

    This is a convenience wrapper around ShellToolInvocation for use as a
    registered primitive. For full control, instantiate ShellToolInvocation
    directly.

    Args:
        command: The shell command to execute.
        cwd: Working directory. Defaults to the current directory.
        description: Human-readable description of what the command does.
        is_background: If True, return early and let the process run.
        timeout: Seconds of inactivity before the process is killed.
        max_output_length: Max characters for stdout/stderr before truncation.
        on_output: Callback invoked with cumulative stdout as data arrives.
        abort_event: If set, the command is cancelled.
        sandbox_manager: Optional sandbox for denial detection.
        confirmation_bus: Optional confirmation UI for permission prompts.
        config: Optional shell configuration provider.

    Returns:
        A ToolResult with llm_content (for the model) and return_display (for the user).
    """
    invocation = ShellToolInvocation(
        params=ShellToolParams(
            command=command,
            description=description,
            dir_path=cwd,
            is_background=is_background,
        ),
        target_dir=cwd or os.getcwd(),
        sandbox_manager=sandbox_manager,
        confirmation_bus=confirmation_bus,
        config=config,
        max_output_length=max_output_length,
    )
    return await invocation.execute(
        on_output=on_output,
        abort_event=abort_event,
    )
