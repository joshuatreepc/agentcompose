"""Unit tests for the modifable_tool module."""

import asyncio
import os
import stat
import sys
from dataclasses import dataclass

import pytest

from agentcompose.primitives.tools.modifable_tool import (
    EditorType,
    ModifyContentOverrides,
    ModifyResult,
    _create_temp_files,
    _delete_temp_files,
    _get_updated_params,
    create_unified_diff,
    is_modifiable_declarative_tool,
    modify_with_editor,
)

# Real submodule (needed for monkeypatching private helpers).
_modify_mod = sys.modules["agentcompose.primitives.tools.modifable_tool"]


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Params:
    file_path: str
    content: str


class _FakeModifyContext:
    """Minimal ModifyContext implementation for testing."""

    def __init__(self, current="OLD\n", proposed="NEW\n"):
        self._current = current
        self._proposed = proposed
        self.get_current_called = 0
        self.get_proposed_called = 0

    def get_file_path(self, params):
        return params.file_path

    async def get_current_content(self, params):
        self.get_current_called += 1
        return self._current

    async def get_proposed_content(self, params):
        self.get_proposed_called += 1
        return self._proposed

    def create_updated_params(self, old_content, modified_proposed_content, original_params):
        return _Params(
            file_path=original_params.file_path,
            content=modified_proposed_content,
        )


# ---------------------------------------------------------------------------
# EditorType
# ---------------------------------------------------------------------------


class TestEditorType:
    def test_values(self):
        assert EditorType.VSCODE == "vscode"
        assert EditorType.SYSTEM == "system"


# ---------------------------------------------------------------------------
# is_modifiable_declarative_tool
# ---------------------------------------------------------------------------


class TestIsModifiableDeclarativeTool:
    def test_object_with_callable_attr_returns_true(self):
        class Tool:
            def get_modify_context(self):
                return None

        assert is_modifiable_declarative_tool(Tool()) is True

    def test_missing_attr_returns_false(self):
        class NotATool:
            pass

        assert is_modifiable_declarative_tool(NotATool()) is False

    def test_non_callable_attr_returns_false(self):
        class WrongShape:
            get_modify_context = "not callable"

        assert is_modifiable_declarative_tool(WrongShape()) is False


# ---------------------------------------------------------------------------
# ModifyContentOverrides
# ---------------------------------------------------------------------------


class TestModifyContentOverrides:
    def test_defaults_to_unset(self):
        # Two instances should share the same _UNSET sentinel
        o1 = ModifyContentOverrides()
        o2 = ModifyContentOverrides()
        assert o1.current_content is o2.current_content
        assert o1.proposed_content is o2.proposed_content

    def test_explicit_values(self):
        o = ModifyContentOverrides(current_content="A", proposed_content="B")
        assert o.current_content == "A"
        assert o.proposed_content == "B"


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------


class TestCreateTempFiles:
    def test_creates_files_with_content(self):
        temps = _create_temp_files("OLD", "NEW", "/path/to/foo.txt")
        try:
            assert os.path.exists(temps.old_path)
            assert os.path.exists(temps.new_path)
            assert open(temps.old_path).read() == "OLD"
            assert open(temps.new_path).read() == "NEW"
        finally:
            _delete_temp_files(temps)

    def test_temp_dir_is_0700(self):
        temps = _create_temp_files("a", "b", "/path/foo.txt")
        try:
            mode = stat.S_IMODE(os.stat(temps.dir_path).st_mode)
            assert mode == 0o700
        finally:
            _delete_temp_files(temps)

    def test_temp_files_are_0600(self):
        temps = _create_temp_files("a", "b", "/path/foo.txt")
        try:
            for fpath in (temps.old_path, temps.new_path):
                mode = stat.S_IMODE(os.stat(fpath).st_mode)
                assert mode == 0o600
        finally:
            _delete_temp_files(temps)

    def test_filenames_preserve_extension_and_stem(self):
        temps = _create_temp_files("a", "b", "/x/myfile.py")
        try:
            assert temps.old_path.endswith(".py")
            assert temps.new_path.endswith(".py")
            assert "myfile" in os.path.basename(temps.old_path)
            assert "myfile" in os.path.basename(temps.new_path)
            assert "-old-" in temps.old_path
            assert "-new-" in temps.new_path
        finally:
            _delete_temp_files(temps)

    def test_no_extension_handled(self):
        temps = _create_temp_files("a", "b", "/x/README")
        try:
            assert os.path.exists(temps.old_path)
        finally:
            _delete_temp_files(temps)


class TestDeleteTempFiles:
    def test_removes_files_and_dir(self):
        temps = _create_temp_files("a", "b", "/x/foo.txt")
        _delete_temp_files(temps)
        assert not os.path.exists(temps.old_path)
        assert not os.path.exists(temps.new_path)
        assert not os.path.exists(temps.dir_path)

    def test_idempotent_on_missing_files(self):
        temps = _create_temp_files("a", "b", "/x/foo.txt")
        _delete_temp_files(temps)
        # Second call should not raise even though files are gone
        _delete_temp_files(temps)


# ---------------------------------------------------------------------------
# create_unified_diff
# ---------------------------------------------------------------------------


class TestCreateUnifiedDiff:
    def test_returns_empty_for_equal_content(self):
        out = create_unified_diff("f.txt", "same\n", "same\n")
        assert out == ""

    def test_shows_added_line(self):
        out = create_unified_diff("f.txt", "a\n", "a\nb\n")
        assert "+b" in out

    def test_shows_removed_line(self):
        out = create_unified_diff("f.txt", "a\nb\n", "a\n")
        assert "-b" in out

    def test_labels_appear_in_diff(self):
        out = create_unified_diff(
            "f.txt", "a\n", "b\n", old_label="Before", new_label="After"
        )
        assert "Before" in out
        assert "After" in out


# ---------------------------------------------------------------------------
# _get_updated_params
# ---------------------------------------------------------------------------


class TestGetUpdatedParams:
    def test_reads_back_temp_files_and_builds_result(self):
        temps = _create_temp_files("OLD\n", "NEW\n", "/x/foo.txt")
        try:
            params = _Params(file_path="/x/foo.txt", content="original")
            ctx = _FakeModifyContext()
            result = _get_updated_params(
                temps.old_path, temps.new_path, params, ctx
            )
            assert isinstance(result, ModifyResult)
            assert result.updated_params.content == "NEW\n"
            assert "NEW" in result.updated_diff or "+NEW" in result.updated_diff
        finally:
            _delete_temp_files(temps)

    def test_missing_files_treated_as_empty(self, tmp_path):
        # Build paths that don't exist
        missing_old = str(tmp_path / "missing_old.txt")
        missing_new = str(tmp_path / "missing_new.txt")
        params = _Params(file_path="/x/foo.txt", content="original")
        ctx = _FakeModifyContext()
        result = _get_updated_params(missing_old, missing_new, params, ctx)
        # Both should be empty; the unified diff between empty strings is empty
        assert result.updated_params.content == ""
        assert result.updated_diff == ""


# ---------------------------------------------------------------------------
# modify_with_editor
# ---------------------------------------------------------------------------


class TestOpenDiff:
    def test_vscode_invokes_code_command(self, monkeypatch):
        captured = {}

        def fake_run(cmd, check=False):
            captured["cmd"] = cmd
            captured["check"] = check

        monkeypatch.setattr(_modify_mod.subprocess, "run", fake_run)
        _modify_mod._open_diff("/tmp/a", "/tmp/b", EditorType.VSCODE)
        assert captured["cmd"][0] == "code"
        assert "--wait" in captured["cmd"]
        assert "--diff" in captured["cmd"]
        assert captured["cmd"][-2:] == ["/tmp/a", "/tmp/b"]

    def test_system_with_vim_editor_uses_diff_mode(self, monkeypatch):
        captured = {}

        def fake_run(cmd, check=False):
            captured["cmd"] = cmd

        monkeypatch.setenv("EDITOR", "/usr/bin/vim")
        monkeypatch.setattr(_modify_mod.subprocess, "run", fake_run)
        _modify_mod._open_diff("/tmp/a", "/tmp/b", EditorType.SYSTEM)
        assert "-d" in captured["cmd"]
        assert captured["cmd"][-2:] == ["/tmp/a", "/tmp/b"]

    def test_system_with_other_editor_no_diff_flag(self, monkeypatch):
        captured = {}

        def fake_run(cmd, check=False):
            captured["cmd"] = cmd

        monkeypatch.setenv("EDITOR", "/usr/bin/nano")
        monkeypatch.setattr(_modify_mod.subprocess, "run", fake_run)
        _modify_mod._open_diff("/tmp/a", "/tmp/b", EditorType.SYSTEM)
        # nano isn't vim — no -d flag
        assert "-d" not in captured["cmd"]
        assert captured["cmd"][0] == "/usr/bin/nano"

    def test_system_no_editor_tries_vimdiff(self, monkeypatch):
        captured = {}

        def fake_run(cmd, check=False):
            captured["cmd"] = cmd

        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setattr(_modify_mod.subprocess, "run", fake_run)
        _modify_mod._open_diff("/tmp/a", "/tmp/b", EditorType.SYSTEM)
        assert captured["cmd"][0] == "vimdiff"


class TestCreateTempFilesError:
    def test_chmod_failure_raises(self, monkeypatch):
        def boom(path, mode):
            raise OSError("permission denied")

        monkeypatch.setattr(_modify_mod.os, "chmod", boom)
        with pytest.raises(OSError):
            _create_temp_files("a", "b", "/x/y.txt")


class TestModifyWithEditor:
    def test_uses_modify_context_when_no_overrides(self, monkeypatch):
        ctx = _FakeModifyContext(current="A\n", proposed="B\n")
        params = _Params(file_path="/x/foo.txt", content="initial")

        # Mock _open_diff to avoid launching an editor; simulate no user edits
        monkeypatch.setattr(_modify_mod, "_open_diff", lambda o, n, e: None)

        result = asyncio.run(modify_with_editor(params, ctx))
        assert ctx.get_current_called == 1
        assert ctx.get_proposed_called == 1
        assert result.updated_params.content == "B\n"

    def test_overrides_skip_fetch(self, monkeypatch):
        ctx = _FakeModifyContext()
        params = _Params(file_path="/x/foo.txt", content="initial")
        monkeypatch.setattr(_modify_mod, "_open_diff", lambda o, n, e: None)

        overrides = ModifyContentOverrides(
            current_content="OVERRIDE_OLD",
            proposed_content="OVERRIDE_NEW",
        )
        result = asyncio.run(modify_with_editor(params, ctx, overrides=overrides))
        # The fake context should NOT have been called
        assert ctx.get_current_called == 0
        assert ctx.get_proposed_called == 0
        assert result.updated_params.content == "OVERRIDE_NEW"

    def test_user_edits_temp_file_reflected_in_params(self, monkeypatch):
        ctx = _FakeModifyContext(current="A\n", proposed="B\n")
        params = _Params(file_path="/x/foo.txt", content="initial")

        def fake_open_diff(old_path, new_path, editor_type):
            # Simulate the user editing the proposed temp file
            with open(new_path, "w") as f:
                f.write("USER_EDITED\n")

        monkeypatch.setattr(_modify_mod, "_open_diff", fake_open_diff)

        result = asyncio.run(modify_with_editor(params, ctx))
        assert result.updated_params.content == "USER_EDITED\n"
        # Diff should reflect the user's edit
        assert "USER_EDITED" in result.updated_diff

    def test_cleanup_happens_on_success(self, monkeypatch):
        ctx = _FakeModifyContext()
        params = _Params(file_path="/x/foo.txt", content="x")

        captured = {}

        def fake_open_diff(old_path, new_path, editor_type):
            captured["dir"] = os.path.dirname(old_path)

        monkeypatch.setattr(_modify_mod, "_open_diff", fake_open_diff)
        asyncio.run(modify_with_editor(params, ctx))
        # Temp directory should have been cleaned up
        assert not os.path.exists(captured["dir"])

    def test_cleanup_happens_on_editor_error(self, monkeypatch):
        ctx = _FakeModifyContext()
        params = _Params(file_path="/x/foo.txt", content="x")

        captured = {}

        def boom(old_path, new_path, editor_type):
            captured["dir"] = os.path.dirname(old_path)
            raise RuntimeError("editor failed")

        monkeypatch.setattr(_modify_mod, "_open_diff", boom)
        with pytest.raises(RuntimeError):
            asyncio.run(modify_with_editor(params, ctx))
        assert not os.path.exists(captured["dir"])

    def test_editor_type_propagated(self, monkeypatch):
        ctx = _FakeModifyContext()
        params = _Params(file_path="/x/foo.txt", content="x")

        seen = {}

        def fake_open_diff(old_path, new_path, editor_type):
            seen["editor_type"] = editor_type

        monkeypatch.setattr(_modify_mod, "_open_diff", fake_open_diff)
        asyncio.run(modify_with_editor(params, ctx, editor_type=EditorType.VSCODE))
        assert seen["editor_type"] == EditorType.VSCODE

    def test_none_override_treated_as_empty(self, monkeypatch):
        ctx = _FakeModifyContext()
        params = _Params(file_path="/x/foo.txt", content="x")
        monkeypatch.setattr(_modify_mod, "_open_diff", lambda o, n, e: None)

        overrides = ModifyContentOverrides(
            current_content=None,
            proposed_content=None,
        )
        result = asyncio.run(modify_with_editor(params, ctx, overrides=overrides))
        # Both fetched as empty string per source: `overrides.current_content or ""`
        assert result.updated_params.content == ""
        assert ctx.get_current_called == 0
        assert ctx.get_proposed_called == 0
