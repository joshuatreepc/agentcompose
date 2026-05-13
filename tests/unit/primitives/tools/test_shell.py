"""Unit tests for the shell primitive.

Focuses on the helpers (command parsing, path utilities, output formatting,
permission heuristics) and a small set of end-to-end execution tests for the
ShellToolInvocation.
"""

import asyncio
import os

import pytest

from agentcompose.primitives.tools.shell import (
    CommandDetail,
    ParsedCommand,
    SandboxPermissions,
    ShellToolInvocation,
    ShellToolParams,
    ToolErrorType,
    _SENSITIVE_DIRS,
    format_bytes,
    get_command_roots,
    get_proactive_permissions,
    has_redirection,
    is_network_reliant_command,
    is_subpath,
    normalize_command,
    parse_command_details,
    shell,
    simplify_paths,
    strip_shell_wrapper,
    truncate_output,
)


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------


class TestHasRedirection:
    def test_simple_redirect(self):
        assert has_redirection("ls > out.txt") is True

    def test_double_redirect(self):
        assert has_redirection("ls >> out.txt") is True

    def test_input_redirect(self):
        assert has_redirection("cat < input") is True

    def test_no_redirect(self):
        assert has_redirection("echo hello") is False


class TestStripShellWrapper:
    def test_removes_env_var_prefix(self):
        assert strip_shell_wrapper("FOO=bar npm install") == "npm install"

    def test_removes_multiple_env_vars(self):
        assert strip_shell_wrapper("X=1 Y=2 cmd arg") == "cmd arg"

    def test_no_change_when_clean(self):
        assert strip_shell_wrapper("echo hello") == "echo hello"


class TestGetCommandRoots:
    def test_simple_command(self):
        assert get_command_roots("echo hello") == ["echo"]

    def test_pipeline(self):
        assert get_command_roots("ls | grep foo") == ["ls", "grep"]

    def test_chained_with_and(self):
        assert get_command_roots("cd /tmp && npm install") == ["cd", "npm"]

    def test_basename_strip(self):
        assert get_command_roots("/usr/bin/python script.py") == ["python"]

    def test_skips_env_var_prefix(self):
        assert get_command_roots("FOO=bar curl https://x.com") == ["curl"]


class TestNormalizeCommand:
    def test_strips_path_and_lowercases(self):
        assert normalize_command("/usr/bin/Python") == "python"


class TestParseCommandDetails:
    def test_single_command(self):
        parsed = parse_command_details("ls -la")
        assert len(parsed.details) == 1
        assert parsed.details[0].name == "ls"
        assert parsed.details[0].args == ["-la"]

    def test_pipeline(self):
        parsed = parse_command_details("cat foo | grep bar")
        assert len(parsed.details) == 2
        assert parsed.details[0].name == "cat"
        assert parsed.details[1].name == "grep"


# ---------------------------------------------------------------------------
# Network heuristics
# ---------------------------------------------------------------------------


class TestNetworkHeuristics:
    def test_curl_always_network(self):
        assert is_network_reliant_command("curl") is True

    def test_npm_install_needs_network(self):
        assert is_network_reliant_command("npm", "install") is True

    def test_npm_run_does_not(self):
        assert is_network_reliant_command("npm", "run") is False

    def test_unknown_command(self):
        assert is_network_reliant_command("ls") is False


class TestProactivePermissions:
    def test_curl_suggests_network(self):
        perms = get_proactive_permissions("curl")
        assert perms is not None
        assert perms.network is True

    def test_unknown_command_no_perms(self):
        assert get_proactive_permissions("ls") is None


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


class TestIsSubpath:
    def test_direct_child(self):
        assert is_subpath("/a", "/a/b") is True

    def test_same_path(self):
        assert is_subpath("/a", "/a") is True

    def test_unrelated(self):
        assert is_subpath("/a", "/b") is False


class TestSimplifyPaths:
    def test_empty(self):
        assert simplify_paths(set()) == []

    def test_redundant_child_removed(self):
        result = simplify_paths({"/a", "/a/b", "/a/b/c"})
        assert result == ["/a"]

    def test_three_siblings_consolidate_to_parent(self):
        # Three siblings under non-sensitive parent should collapse
        result = simplify_paths({"/proj/a", "/proj/b", "/proj/c"})
        assert result == ["/proj"]

    def test_two_siblings_stay_listed(self):
        result = simplify_paths({"/proj/a", "/proj/b"})
        assert sorted(result) == ["/proj/a", "/proj/b"]

    def test_sensitive_parent_not_consolidated(self):
        sensitive = next(iter(_SENSITIVE_DIRS))
        children = {os.path.join(sensitive, x) for x in ("a", "b", "c")}
        result = simplify_paths(children)
        # Should NOT collapse to the sensitive parent
        assert sensitive not in result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_below_threshold(self):
        assert truncate_output("hi", 100, "stdout") == "hi"

    def test_above_threshold(self):
        out = truncate_output("a" * 200, 50, "stdout")
        assert "150 characters removed" in out


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(512) == "512 B"

    def test_kilobytes(self):
        assert "KB" in format_bytes(2048)

    def test_megabytes(self):
        assert "MB" in format_bytes(2 * 1024 * 1024)


# ---------------------------------------------------------------------------
# ShellToolInvocation
# ---------------------------------------------------------------------------


class TestShellInvocationValidation:
    def test_empty_command_rejected(self, tmp_path):
        inv = ShellToolInvocation(
            ShellToolParams(command="   "),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        assert result.error is not None


class TestShellInvocationExecute:
    def test_runs_simple_echo(self, tmp_path):
        inv = ShellToolInvocation(
            ShellToolParams(command="echo hello"),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        assert result.error is None
        assert "hello" in result.llm_content

    def test_nonzero_exit_reported(self, tmp_path):
        inv = ShellToolInvocation(
            ShellToolParams(command="exit 7"),
            target_dir=str(tmp_path),
        )
        result = asyncio.run(inv.execute())
        assert "Exit Code: 7" in result.llm_content

    def test_get_description_uses_command_under_threshold(self, tmp_path):
        inv = ShellToolInvocation(
            ShellToolParams(command="ls", description="list files"),
            target_dir=str(tmp_path),
        )
        # Short command — description shows the command itself
        assert inv.get_description() == "ls"

    def test_get_description_uses_description_when_long(self, tmp_path):
        long_cmd = "echo " + "x" * 200
        inv = ShellToolInvocation(
            ShellToolParams(command=long_cmd, description="prints x's"),
            target_dir=str(tmp_path),
        )
        assert inv.get_description() == "prints x's"

    def test_contextual_details_includes_dir(self, tmp_path):
        inv = ShellToolInvocation(
            ShellToolParams(command="ls", dir_path="src"),
            target_dir=str(tmp_path),
        )
        details = inv.get_contextual_details()
        assert "src" in details

    def test_background_marker_in_details(self, tmp_path):
        inv = ShellToolInvocation(
            ShellToolParams(command="ls", is_background=True),
            target_dir=str(tmp_path),
        )
        assert "[background]" in inv.get_contextual_details()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestShellRegistry:
    def test_registered_as_tool(self):
        primitive = shell.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
