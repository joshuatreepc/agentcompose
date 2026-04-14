"""
Guardrail system for agentcompose tool primitives.

Provides input/output guardrails that can be attached to any tool via
the registry context system. Guardrails run before and/or after tool
execution to enforce safety policies.

Architecture:
- Guardrail: A check that inspects tool input or output and returns pass/fail.
- GuardrailPipeline: An ordered sequence of guardrails for a given stage.
- Stages: pre_flight → input → (tool executes) → output
- Tool-level vs agent-level: tool guardrails wrap individual tool calls,
  agent guardrails wrap the entire agent interaction.

Integration with agentcompose:
- Guardrails are injected via registry.configure(guardrails=...)
- The before_hook / after_hook pattern on bash/shell tools can use guardrails
- GuardrailAgent wraps any adapter (PydanticAI, etc.) to apply agent-level guards

Example:
    >>> from agentcompose.primitives.tools.guard_rail import (
    ...     Guardrail, GuardrailPipeline, GuardrailAgent,
    ...     path_allowlist, command_denylist, prompt_injection_check,
    ... )
    >>>
    >>> # Build a pipeline
    >>> pipeline = GuardrailPipeline()
    >>> pipeline.add_input(command_denylist(["rm", "curl", "wget"]))
    >>> pipeline.add_input(path_allowlist(["/home/user/project"]))
    >>> pipeline.add_output(prompt_injection_check())
    >>>
    >>> # Attach to tools via context
    >>> shell.configure(guardrails=pipeline)
    >>> bash.configure(guardrails=pipeline)
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class GuardrailStage(str, Enum):
    """When a guardrail runs in the tool lifecycle."""

    PRE_FLIGHT = "pre_flight"  # Before any processing
    INPUT = "input"            # Before tool execution, after param parsing
    OUTPUT = "output"          # After tool execution, before returning result


class GuardrailAction(str, Enum):
    """What to do when a guardrail triggers."""

    ALLOW = "allow"            # Check passed
    BLOCK = "block"            # Halt execution, return error
    WARN = "warn"              # Log warning but continue
    REDACT = "redact"          # Modify the content and continue


@dataclass
class GuardrailResult:
    """Result of a single guardrail check."""

    action: GuardrailAction
    guardrail_name: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    redacted_content: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.action == GuardrailAction.ALLOW

    @property
    def triggered(self) -> bool:
        return self.action in (GuardrailAction.BLOCK, GuardrailAction.WARN, GuardrailAction.REDACT)


@dataclass
class GuardrailContext:
    """Context passed to guardrail checks."""

    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_output: Optional[str] = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guardrail protocol and base class
# ---------------------------------------------------------------------------


@runtime_checkable
class GuardrailCheck(Protocol):
    """Protocol for guardrail implementations."""

    name: str

    def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        """Run the guardrail check.

        Args:
            content: The text to check (tool input or output).
            context: Additional context about the tool call.

        Returns:
            A GuardrailResult indicating pass/fail.
        """
        ...


@runtime_checkable
class AsyncGuardrailCheck(Protocol):
    """Protocol for async guardrail implementations (e.g., LLM-based checks)."""

    name: str

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        ...


class Guardrail:
    """Base class for guardrail implementations.

    Subclass this and implement ``_check`` to create custom guardrails.
    """

    def __init__(self, name: str):
        self.name = name

    def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        start = time.monotonic()
        result = self._check(content, context)
        result.elapsed_s = time.monotonic() - start
        result.guardrail_name = self.name
        return result

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        """Override this in subclasses."""
        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


# ---------------------------------------------------------------------------
# GuardrailPipeline — ordered sequence of checks
# ---------------------------------------------------------------------------


@dataclass
class PipelineStage:
    """A collection of guardrails for a specific stage."""

    guardrails: list[Guardrail] = field(default_factory=list)


class GuardrailPipeline:
    """Ordered pipeline of guardrail checks across stages.

    Guardrails are organized into three stages:
    - pre_flight: Runs before any tool processing (e.g., rate limiting)
    - input: Runs before tool execution (e.g., command validation)
    - output: Runs after tool execution (e.g., output sanitization)

    Within each stage, guardrails run in order. If any guardrail returns
    BLOCK, the pipeline short-circuits and returns that result.

    Example:
        >>> pipeline = GuardrailPipeline()
        >>> pipeline.add_input(command_denylist(["rm -rf"]))
        >>> pipeline.add_output(secret_redaction(["API_KEY"]))
        >>>
        >>> # Run manually
        >>> results = pipeline.run_input("rm -rf /", context)
        >>> if any(r.action == GuardrailAction.BLOCK for r in results):
        ...     print("Blocked!")
    """

    def __init__(self) -> None:
        self.pre_flight = PipelineStage()
        self.input = PipelineStage()
        self.output = PipelineStage()

    def add_pre_flight(self, guardrail: Guardrail) -> "GuardrailPipeline":
        self.pre_flight.guardrails.append(guardrail)
        return self

    def add_input(self, guardrail: Guardrail) -> "GuardrailPipeline":
        self.input.guardrails.append(guardrail)
        return self

    def add_output(self, guardrail: Guardrail) -> "GuardrailPipeline":
        self.output.guardrails.append(guardrail)
        return self

    def run_stage(
        self,
        stage: PipelineStage,
        content: str,
        context: GuardrailContext,
        *,
        stop_on_block: bool = True,
    ) -> list[GuardrailResult]:
        """Run all guardrails in a stage.

        Args:
            stage: The pipeline stage to run.
            content: Text to check.
            context: Tool call context.
            stop_on_block: If True, stop at the first BLOCK result.

        Returns:
            List of results from each guardrail.
        """
        results: list[GuardrailResult] = []
        for guardrail in stage.guardrails:
            result = guardrail.check(content, context)
            results.append(result)

            if result.action == GuardrailAction.BLOCK and stop_on_block:
                break

            # Apply redaction for subsequent checks
            if result.action == GuardrailAction.REDACT and result.redacted_content is not None:
                content = result.redacted_content

        return results

    def run_pre_flight(self, content: str, context: GuardrailContext) -> list[GuardrailResult]:
        return self.run_stage(self.pre_flight, content, context)

    def run_input(self, content: str, context: GuardrailContext) -> list[GuardrailResult]:
        return self.run_stage(self.input, content, context)

    def run_output(self, content: str, context: GuardrailContext) -> list[GuardrailResult]:
        return self.run_stage(self.output, content, context)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GuardrailPipeline":
        """Build a pipeline from a configuration dict.

        Config format:
            {
                "pre_flight": [{"type": "rate_limit", "max_per_minute": 30}],
                "input": [
                    {"type": "command_denylist", "commands": ["rm", "curl"]},
                    {"type": "path_allowlist", "paths": ["/home/user/project"]},
                ],
                "output": [
                    {"type": "secret_redaction", "patterns": ["API_KEY"]},
                ]
            }
        """
        pipeline = cls()
        for stage_name, stage_attr in [
            ("pre_flight", pipeline.pre_flight),
            ("input", pipeline.input),
            ("output", pipeline.output),
        ]:
            for guard_config in config.get(stage_name, []):
                guard_type = guard_config.get("type", "")
                factory = _GUARDRAIL_REGISTRY.get(guard_type)
                if factory is None:
                    logger.warning("Unknown guardrail type: %s", guard_type)
                    continue
                stage_attr.guardrails.append(factory(guard_config))
        return pipeline


# ---------------------------------------------------------------------------
# Built-in guardrails
# ---------------------------------------------------------------------------


class CommandDenylist(Guardrail):
    """Block shell commands that match a denylist.

    Checks the root command of a shell invocation against a list of
    denied commands. Supports exact matches and prefix matches.

    Example:
        >>> guard = command_denylist(["rm", "curl", "wget", "sudo"])
        >>> result = guard.check("rm -rf /", context)
        >>> result.action
        <GuardrailAction.BLOCK: 'block'>
    """

    def __init__(self, commands: list[str]):
        super().__init__("Command Denylist")
        self.commands = [c.lower() for c in commands]

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        # Extract root commands from the content
        parts = re.split(r"\s*(?:&&|\|\|?|;)\s*", content.strip())
        for part in parts:
            tokens = part.strip().split()
            if not tokens:
                continue
            # Skip env var assignments
            root = None
            for tok in tokens:
                if "=" in tok and tok.split("=")[0].isidentifier():
                    continue
                root = os.path.basename(tok).lower()
                break

            if root and root in self.commands:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name=self.name,
                    message=f"Command '{root}' is not allowed.",
                    details={"blocked_command": root, "full_input": content},
                )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class CommandAllowlist(Guardrail):
    """Only allow shell commands that match an allowlist.

    The inverse of CommandDenylist — blocks everything not explicitly listed.

    Example:
        >>> guard = command_allowlist(["ls", "cat", "grep", "find", "git"])
        >>> result = guard.check("curl http://evil.com", context)
        >>> result.action
        <GuardrailAction.BLOCK: 'block'>
    """

    def __init__(self, commands: list[str]):
        super().__init__("Command Allowlist")
        self.commands = {c.lower() for c in commands}

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        parts = re.split(r"\s*(?:&&|\|\|?|;)\s*", content.strip())
        for part in parts:
            tokens = part.strip().split()
            if not tokens:
                continue
            root = None
            for tok in tokens:
                if "=" in tok and tok.split("=")[0].isidentifier():
                    continue
                root = os.path.basename(tok).lower()
                break

            if root and root not in self.commands:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name=self.name,
                    message=f"Command '{root}' is not in the allowlist.",
                    details={"blocked_command": root, "allowed": sorted(self.commands)},
                )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class PathAllowlist(Guardrail):
    """Block file operations outside allowed directory trees.

    Checks file paths in tool arguments against a list of allowed
    parent directories.

    Example:
        >>> guard = path_allowlist(["/home/user/project"])
        >>> result = guard.check("/etc/passwd", context)
        >>> result.action
        <GuardrailAction.BLOCK: 'block'>
    """

    def __init__(self, allowed_paths: list[str]):
        super().__init__("Path Allowlist")
        self.allowed_paths = [os.path.abspath(p) for p in allowed_paths]

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        # Check path-like arguments from the tool context
        paths_to_check: list[str] = []

        for key in ("file_path", "path", "cwd", "dir_path"):
            val = context.tool_args.get(key)
            if val and isinstance(val, str):
                paths_to_check.append(val)

        # Also check the content itself if it looks like a path
        if content.startswith("/") or content.startswith("~"):
            paths_to_check.append(content)

        for check_path in paths_to_check:
            abs_path = os.path.abspath(os.path.expanduser(check_path))
            if not any(
                abs_path == allowed or abs_path.startswith(allowed + os.sep)
                for allowed in self.allowed_paths
            ):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name=self.name,
                    message=f"Path '{check_path}' is outside allowed directories.",
                    details={
                        "blocked_path": abs_path,
                        "allowed_paths": self.allowed_paths,
                    },
                )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class PathDenylist(Guardrail):
    """Block file operations targeting specific paths or patterns.

    Example:
        >>> guard = path_denylist(["/etc", "/root", "*.env", "*.pem"])
        >>> result = guard.check(".env", context)
        >>> result.action
        <GuardrailAction.BLOCK: 'block'>
    """

    def __init__(self, denied_paths: list[str]):
        super().__init__("Path Denylist")
        # Separate glob patterns from absolute paths
        self.denied_dirs = [os.path.abspath(p) for p in denied_paths if not p.startswith("*")]
        self.denied_patterns = [p for p in denied_paths if p.startswith("*") or "*" in p]

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        paths_to_check: list[str] = []
        for key in ("file_path", "path", "cwd", "dir_path"):
            val = context.tool_args.get(key)
            if val and isinstance(val, str):
                paths_to_check.append(val)

        for check_path in paths_to_check:
            abs_path = os.path.abspath(os.path.expanduser(check_path))
            basename = os.path.basename(check_path)

            # Check directory deny list
            for denied in self.denied_dirs:
                if abs_path == denied or abs_path.startswith(denied + os.sep):
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        guardrail_name=self.name,
                        message=f"Path '{check_path}' is denied.",
                        details={"blocked_path": abs_path, "matched_rule": denied},
                    )

            # Check glob patterns
            for pattern in self.denied_patterns:
                if fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(check_path, pattern):
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        guardrail_name=self.name,
                        message=f"Path '{check_path}' matches denied pattern '{pattern}'.",
                        details={"blocked_path": check_path, "matched_pattern": pattern},
                    )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class SecretRedaction(Guardrail):
    """Redact secrets and sensitive patterns from tool output.

    Scans output for patterns like API keys, tokens, and passwords,
    replacing them with [REDACTED].

    Example:
        >>> guard = secret_redaction()
        >>> result = guard.check("API_KEY=sk-abc123def456", context)
        >>> result.redacted_content
        'API_KEY=[REDACTED]'
    """

    # Common secret patterns
    DEFAULT_PATTERNS = [
        (r"(?i)(api[_-]?key|secret|token|password|passwd|pwd)\s*[=:]\s*\S+", "credential assignment"),
        (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
        (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
        (r"github_pat_[a-zA-Z0-9_]{22,}", "GitHub fine-grained PAT"),
        (r"xox[bprs]-[a-zA-Z0-9\-]+", "Slack token"),
        (r"AKIA[0-9A-Z]{16}", "AWS access key"),
        (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Private key"),
        (r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}", "JWT token"),
    ]

    def __init__(self, extra_patterns: Optional[list[str]] = None):
        super().__init__("Secret Redaction")
        self.patterns: list[tuple[re.Pattern, str]] = []
        for pattern, desc in self.DEFAULT_PATTERNS:
            self.patterns.append((re.compile(pattern), desc))
        if extra_patterns:
            for p in extra_patterns:
                self.patterns.append((re.compile(p), "custom pattern"))

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        redacted = content
        found: list[str] = []

        for pattern, desc in self.patterns:
            if pattern.search(redacted):
                found.append(desc)
                redacted = pattern.sub("[REDACTED]", redacted)

        if found:
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                guardrail_name=self.name,
                message=f"Redacted {len(found)} secret(s): {', '.join(found)}",
                details={"redacted_types": found},
                redacted_content=redacted,
            )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class MaxOutputLength(Guardrail):
    """Block or truncate tool output that exceeds a length limit.

    Example:
        >>> guard = max_output_length(50000, action="redact")
    """

    def __init__(self, max_length: int = 50_000, action: str = "warn"):
        super().__init__("Max Output Length")
        self.max_length = max_length
        self.on_exceed = GuardrailAction(action)

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        if len(content) <= self.max_length:
            return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)

        if self.on_exceed == GuardrailAction.REDACT:
            truncated = content[: self.max_length] + "\n\n[Output truncated by guardrail]"
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                guardrail_name=self.name,
                message=f"Output truncated from {len(content)} to {self.max_length} chars.",
                redacted_content=truncated,
            )

        return GuardrailResult(
            action=self.on_exceed,
            guardrail_name=self.name,
            message=f"Output length {len(content)} exceeds limit {self.max_length}.",
        )


class RateLimit(Guardrail):
    """Limit tool execution frequency.

    Tracks call timestamps and blocks if the rate is exceeded.

    Example:
        >>> guard = rate_limit(max_calls=10, window_s=60)
    """

    def __init__(self, max_calls: int = 30, window_s: float = 60.0):
        super().__init__("Rate Limit")
        self.max_calls = max_calls
        self.window_s = window_s
        self._timestamps: list[float] = []

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        now = time.monotonic()
        cutoff = now - self.window_s

        # Prune old timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.max_calls:
            wait = self._timestamps[0] + self.window_s - now
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                guardrail_name=self.name,
                message=f"Rate limit exceeded ({self.max_calls} calls per {self.window_s}s). Wait {wait:.1f}s.",
                details={"wait_s": wait, "current_count": len(self._timestamps)},
            )

        self._timestamps.append(now)
        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class ContentPattern(Guardrail):
    """Block content matching specific regex patterns.

    Useful for catching prompt injection attempts, dangerous commands,
    or policy-violating content.

    Example:
        >>> guard = content_pattern(
        ...     patterns=[r"ignore previous instructions", r"you are now"],
        ...     name="Prompt Injection Detection",
        ... )
    """

    def __init__(self, patterns: list[str], name: str = "Content Pattern"):
        super().__init__(name)
        self.patterns = [(re.compile(p, re.IGNORECASE), p) for p in patterns]

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        for compiled, raw in self.patterns:
            match = compiled.search(content)
            if match:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name=self.name,
                    message=f"Content matched blocked pattern.",
                    details={"pattern": raw, "match": match.group()},
                )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


class NetworkDenylist(Guardrail):
    """Block commands that access denied network hosts or URLs.

    Example:
        >>> guard = network_denylist(["*.internal.corp", "10.0.0.*"])
    """

    def __init__(self, denied_hosts: list[str]):
        super().__init__("Network Denylist")
        self.denied_hosts = denied_hosts

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        # Check for URLs in content
        urls = re.findall(r"https?://([^/\s:]+)", content)
        for host in urls:
            for pattern in self.denied_hosts:
                if fnmatch.fnmatch(host.lower(), pattern.lower()):
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        guardrail_name=self.name,
                        message=f"Access to host '{host}' is denied.",
                        details={"blocked_host": host, "matched_pattern": pattern},
                    )

        return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)


# ---------------------------------------------------------------------------
# Factory functions — convenient constructors
# ---------------------------------------------------------------------------


def command_denylist(commands: list[str]) -> CommandDenylist:
    """Create a guardrail that blocks specific commands."""
    return CommandDenylist(commands)


def command_allowlist(commands: list[str]) -> CommandAllowlist:
    """Create a guardrail that only allows specific commands."""
    return CommandAllowlist(commands)


def path_allowlist(paths: list[str]) -> PathAllowlist:
    """Create a guardrail that restricts file access to specific directories."""
    return PathAllowlist(paths)


def path_denylist(paths: list[str]) -> PathDenylist:
    """Create a guardrail that blocks access to specific paths or patterns."""
    return PathDenylist(paths)


def secret_redaction(extra_patterns: Optional[list[str]] = None) -> SecretRedaction:
    """Create a guardrail that redacts secrets from output."""
    return SecretRedaction(extra_patterns)


def max_output_length(length: int = 50_000, action: str = "warn") -> MaxOutputLength:
    """Create a guardrail that limits output length."""
    return MaxOutputLength(length, action)


def rate_limit(max_calls: int = 30, window_s: float = 60.0) -> RateLimit:
    """Create a guardrail that limits tool call frequency."""
    return RateLimit(max_calls, window_s)


def content_pattern(patterns: list[str], name: str = "Content Pattern") -> ContentPattern:
    """Create a guardrail that blocks content matching regex patterns."""
    return ContentPattern(patterns, name)


def prompt_injection_check() -> ContentPattern:
    """Create a guardrail that detects common prompt injection patterns."""
    return ContentPattern(
        patterns=[
            r"ignore (?:all )?previous instructions",
            r"ignore (?:all )?(?:above|prior) (?:instructions|prompts|context)",
            r"you are now (?:a |an )?",
            r"forget (?:all |everything )?(?:you were|your|previous)",
            r"new instructions:",
            r"system prompt:",
            r"override (?:all )?(?:safety|security|guardrail)",
            r"disregard (?:all )?(?:safety|rules|guidelines|instructions)",
            r"pretend (?:you are|to be)",
            r"act as (?:if |though )?(?:you (?:are|have))",
            r"jailbreak",
            r"DAN mode",
        ],
        name="Prompt Injection Detection",
    )


def network_denylist(hosts: list[str]) -> NetworkDenylist:
    """Create a guardrail that blocks access to specific network hosts."""
    return NetworkDenylist(hosts)


class PIIDetection(Guardrail):
    """Detect and redact personally identifiable information in tool output.

    Detects:
    - Email addresses (RFC-compliant pattern)
    - Phone numbers (US/international formats)
    - Social Security Numbers (XXX-XX-XXXX)
    - Credit card numbers (major card brands)
    - IP addresses (IPv4)
    - Street addresses (common US patterns)
    - Names near PII context clues (near "name:", "patient:", etc.)

    Patterns ported from datacompose's PySpark PII transformers,
    adapted for plain-text guardrail use.

    Example:
        >>> guard = pii_detection()
        >>> result = guard.check("Contact john.doe@gmail.com at 555-123-4567", ctx)
        >>> result.action
        <GuardrailAction.REDACT: 'redact'>
        >>> result.redacted_content
        'Contact [EMAIL REDACTED] at [PHONE REDACTED]'
    """

    # Pattern definitions: (compiled regex, PII type label, replacement)
    DEFAULT_PII_PATTERNS: list[tuple[str, str, str]] = [
        # Email addresses (from datacompose emails.extract_email)
        (
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "email",
            "[EMAIL REDACTED]",
        ),
        # US phone numbers — multiple formats
        # (555) 123-4567, 555-123-4567, 555.123.4567, +1 555 123 4567
        (
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "phone",
            "[PHONE REDACTED]",
        ),
        # International phone with country code (+44 20 7946 0958)
        (
            r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            "international_phone",
            "[PHONE REDACTED]",
        ),
        # Social Security Numbers (123-45-6789 or 123 45 6789)
        (
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            "ssn",
            "[SSN REDACTED]",
        ),
        # Credit card numbers (major brands)
        # Visa: 4xxx, MC: 5[1-5]xx, Amex: 3[47]xx, Discover: 6011/65xx
        (
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "credit_card",
            "[CREDIT CARD REDACTED]",
        ),
        # Amex (15 digits: 3[47]xx-xxxxxx-xxxxx)
        (
            r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",
            "credit_card",
            "[CREDIT CARD REDACTED]",
        ),
        # IPv4 addresses (but not version numbers like 1.2.3)
        (
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
            "ip_address",
            "[IP REDACTED]",
        ),
        # US street addresses (123 Main St, 456 Oak Avenue, etc.)
        (
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Ln|Lane|Rd|Road|Ct|Court|Pl|Place|Way|Cir(?:cle)?|Terr(?:ace)?|Pkwy|Parkway)\b",
            "address",
            "[ADDRESS REDACTED]",
        ),
        # Date of birth patterns near context clues (DOB: 01/15/1990, born 1990-01-15)
        (
            r"(?i)(?:DOB|date of birth|born|birthday)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
            "date_of_birth",
            "[DOB REDACTED]",
        ),
    ]

    def __init__(
        self,
        extra_patterns: Optional[list[tuple[str, str, str]]] = None,
        detect_only: Optional[set[str]] = None,
        action_on_detect: str = "redact",
    ):
        """Initialize PII detection guardrail.

        Args:
            extra_patterns: Additional (regex, label, replacement) tuples.
            detect_only: If set, only detect these PII types (e.g. {"email", "ssn"}).
            action_on_detect: "redact" to replace PII, "block" to halt entirely.
        """
        super().__init__("PII Detection")
        self.action_on_detect = GuardrailAction(action_on_detect)
        self.detect_only = detect_only

        all_patterns = list(self.DEFAULT_PII_PATTERNS)
        if extra_patterns:
            all_patterns.extend(extra_patterns)

        self.patterns: list[tuple[re.Pattern, str, str]] = []
        for regex_str, label, replacement in all_patterns:
            if detect_only and label not in detect_only:
                continue
            self.patterns.append((re.compile(regex_str), label, replacement))

    def _check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        redacted = content
        found: list[dict[str, Any]] = []

        for pattern, label, replacement in self.patterns:
            matches = list(pattern.finditer(redacted))
            if matches:
                for match in matches:
                    found.append({
                        "type": label,
                        "match": match.group(),
                        "position": match.start(),
                    })
                redacted = pattern.sub(replacement, redacted)

        if not found:
            return GuardrailResult(action=GuardrailAction.ALLOW, guardrail_name=self.name)

        pii_types = list(dict.fromkeys(f["type"] for f in found))
        message = f"Detected {len(found)} PII instance(s): {', '.join(pii_types)}"

        if self.action_on_detect == GuardrailAction.BLOCK:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                guardrail_name=self.name,
                message=message,
                details={"pii_found": found, "pii_types": pii_types},
            )

        return GuardrailResult(
            action=GuardrailAction.REDACT,
            guardrail_name=self.name,
            message=message,
            details={"pii_found": found, "pii_types": pii_types},
            redacted_content=redacted,
        )


def pii_detection(
    extra_patterns: Optional[list[tuple[str, str, str]]] = None,
    detect_only: Optional[set[str]] = None,
    action: str = "redact",
) -> PIIDetection:
    """Create a guardrail that detects and redacts PII.

    Args:
        extra_patterns: Additional (regex, label, replacement) tuples.
        detect_only: Only detect specific PII types (e.g. {"email", "phone", "ssn"}).
            Available types: email, phone, international_phone, ssn, credit_card,
            ip_address, address, date_of_birth.
        action: "redact" (default) or "block".

    Example:
        >>> guard = pii_detection()
        >>> guard = pii_detection(detect_only={"email", "ssn"})
        >>> guard = pii_detection(action="block")  # halt instead of redact
    """
    return PIIDetection(extra_patterns, detect_only, action)


# ---------------------------------------------------------------------------
# Guardrail registry for config-driven instantiation
# ---------------------------------------------------------------------------

_GUARDRAIL_REGISTRY: dict[str, Callable[[dict[str, Any]], Guardrail]] = {
    "command_denylist": lambda c: CommandDenylist(c.get("commands", [])),
    "command_allowlist": lambda c: CommandAllowlist(c.get("commands", [])),
    "path_allowlist": lambda c: PathAllowlist(c.get("paths", [])),
    "path_denylist": lambda c: PathDenylist(c.get("paths", [])),
    "secret_redaction": lambda c: SecretRedaction(c.get("extra_patterns")),
    "max_output_length": lambda c: MaxOutputLength(c.get("max_length", 50_000), c.get("action", "warn")),
    "rate_limit": lambda c: RateLimit(c.get("max_calls", 30), c.get("window_s", 60.0)),
    "content_pattern": lambda c: ContentPattern(c.get("patterns", []), c.get("name", "Content Pattern")),
    "prompt_injection": lambda c: prompt_injection_check(),
    "network_denylist": lambda c: NetworkDenylist(c.get("hosts", [])),
    "pii_detection": lambda c: PIIDetection(
        extra_patterns=c.get("extra_patterns"),
        detect_only=set(c["detect_only"]) if c.get("detect_only") else None,
        action_on_detect=c.get("action", "redact"),
    ),
}


def register_guardrail_type(name: str, factory: Callable[[dict[str, Any]], Guardrail]) -> None:
    """Register a custom guardrail type for config-driven pipelines.

    Args:
        name: Type name used in config files.
        factory: Callable that takes a config dict and returns a Guardrail.
    """
    _GUARDRAIL_REGISTRY[name] = factory


# ---------------------------------------------------------------------------
# Hook integration — bridges guardrails into before_hook / after_hook
# ---------------------------------------------------------------------------


def create_before_hook(pipeline: GuardrailPipeline, tool_name: str = "") -> Callable[[str], Optional[str]]:
    """Create a before_hook for bash/shell that runs input guardrails.

    Returns a function compatible with the bash.configure(before_hook=...) pattern.
    If a guardrail blocks, raises ValueError. If it redacts, returns the modified command.

    Example:
        >>> pipeline = GuardrailPipeline()
        >>> pipeline.add_input(command_denylist(["rm"]))
        >>> bash.configure(before_hook=create_before_hook(pipeline, "bash"))
    """

    def hook(command: str) -> Optional[str]:
        context = GuardrailContext(
            tool_name=tool_name,
            tool_args={"command": command},
        )

        # Run pre_flight + input stages
        for stage in [pipeline.pre_flight, pipeline.input]:
            results = pipeline.run_stage(stage, command, context)
            for result in results:
                if result.action == GuardrailAction.BLOCK:
                    raise ValueError(
                        f"Guardrail '{result.guardrail_name}' blocked: {result.message}"
                    )
                if result.action == GuardrailAction.REDACT and result.redacted_content is not None:
                    command = result.redacted_content

        return command

    return hook


def create_after_hook(pipeline: GuardrailPipeline, tool_name: str = "") -> Callable:
    """Create an after_hook for bash that runs output guardrails.

    Returns a function compatible with the bash.configure(after_hook=...) pattern.

    Example:
        >>> pipeline = GuardrailPipeline()
        >>> pipeline.add_output(secret_redaction())
        >>> bash.configure(after_hook=create_after_hook(pipeline, "bash"))
    """

    def hook(command: str, result: Any) -> Optional[Any]:
        output = getattr(result, "stdout", str(result))
        context = GuardrailContext(
            tool_name=tool_name,
            tool_args={"command": command},
            tool_output=output,
        )

        results = pipeline.run_output(output, context)
        modified = False
        for gr_result in results:
            if gr_result.action == GuardrailAction.BLOCK:
                # Replace output with error message
                result.stdout = f"[BLOCKED by {gr_result.guardrail_name}]: {gr_result.message}"
                result.stderr = ""
                return result
            if gr_result.action == GuardrailAction.REDACT and gr_result.redacted_content is not None:
                result.stdout = gr_result.redacted_content
                modified = True

        return result if modified else None

    return hook
