"""Unit tests for the bash primitive."""

import asyncio

import pytest

from agentcompose.primitives.tools.bash import (
    BashResult,
    DEFAULT_MAX_OUTPUT_LENGTH,
    _truncate_output,
    bash,
)


class TestTruncateOutput:
    def test_below_threshold_unchanged(self):
        assert _truncate_output("hi", 100, "stdout") == "hi"

    def test_above_threshold_truncated_with_notice(self):
        out = _truncate_output("a" * 200, 50, "stdout")
        assert out.startswith("a" * 50)
        assert "150 characters removed" in out
        assert "[stdout truncated:" in out


class TestBashSync:
    def test_simple_echo(self):
        result = bash.execute("echo hello")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    def test_nonzero_exit(self):
        result = bash.execute("exit 7")
        assert result.exit_code == 7

    def test_stderr_capture(self):
        result = bash.execute("echo oops 1>&2")
        assert result.exit_code == 0
        assert result.stderr.strip() == "oops"

    def test_cwd(self, tmp_path):
        result = bash.execute("pwd", cwd=str(tmp_path))
        assert result.exit_code == 0
        # On macOS, /tmp paths may be /private/tmp via symlink — accept either.
        assert str(tmp_path) in result.stdout or str(tmp_path).replace("/private", "") in result.stdout

    def test_timeout(self):
        result = bash.execute("sleep 5", timeout=0.1)
        assert result.exit_code == 124
        assert "timed out" in result.stderr

    def test_before_hook_modifies_command(self):
        bash.configure(before_hook=lambda cmd: "echo modified")
        try:
            result = bash.execute("echo original")
            assert result.stdout.strip() == "modified"
        finally:
            bash.configure(before_hook=None)

    def test_after_hook_modifies_result(self):
        def post(cmd, result):
            return BashResult(
                stdout="REPLACED",
                stderr=result.stderr,
                exit_code=result.exit_code,
            )

        bash.configure(after_hook=post)
        try:
            result = bash.execute("echo something")
            assert result.stdout == "REPLACED"
        finally:
            bash.configure(after_hook=None)

    def test_max_output_length_truncates(self):
        # Generate output > 200 chars but limit max_output_length to 50
        bash.configure(max_output_length=50)
        try:
            result = bash.execute("printf '%.0s.' {1..500}")
            assert "truncated" in result.stdout
        finally:
            bash.configure(max_output_length=DEFAULT_MAX_OUTPUT_LENGTH)


class TestBashAsync:
    def test_async_echo(self):
        result = asyncio.run(bash.execute_async("echo async"))
        assert result.exit_code == 0
        assert result.stdout.strip() == "async"

    def test_async_nonzero_exit(self):
        result = asyncio.run(bash.execute_async("exit 3"))
        assert result.exit_code == 3

    def test_async_timeout(self):
        result = asyncio.run(bash.execute_async("sleep 5", timeout=0.1))
        assert result.exit_code == 124
        assert "timed out" in result.stderr


class TestBashAsyncHooks:
    def test_async_before_hook_modifies_command(self):
        bash.configure(before_hook=lambda cmd: "echo modified-async")
        try:
            result = asyncio.run(bash.execute_async("echo original"))
            assert result.stdout.strip() == "modified-async"
        finally:
            bash.configure(before_hook=None)

    def test_async_after_hook_modifies_result(self):
        def post(cmd, result):
            return BashResult(
                stdout="ASYNC_REPLACED",
                stderr=result.stderr,
                exit_code=result.exit_code,
            )

        bash.configure(after_hook=post)
        try:
            result = asyncio.run(bash.execute_async("echo something"))
            assert result.stdout == "ASYNC_REPLACED"
        finally:
            bash.configure(after_hook=None)

    def test_async_before_hook_returning_none_keeps_command(self):
        bash.configure(before_hook=lambda cmd: None)
        try:
            result = asyncio.run(bash.execute_async("echo kept"))
            assert result.stdout.strip() == "kept"
        finally:
            bash.configure(before_hook=None)


class TestBashRegistry:
    def test_sync_registered_as_tool(self):
        primitive = bash.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is False

    def test_async_registered_as_tool(self):
        primitive = bash.get("execute_async")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is False
