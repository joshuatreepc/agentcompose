"""Unit tests for the write_file primitive."""

import errno
import os

from agentcompose.primitives.tools.write_file import (
    ToolErrorType,
    WriteFileModifyContext,
    WriteFileToolInvocation,
    WriteFileToolParams,
    create_unified_diff,
    detect_line_ending,
    detect_omission_placeholders,
    get_diff_context_snippet,
    get_diff_stat,
    make_relative,
    shorten_path,
    write_file,
)


class TestDetectLineEnding:
    def test_lf_dominant(self):
        assert detect_line_ending("a\nb\nc\n") == "\n"

    def test_crlf_dominant(self):
        assert detect_line_ending("a\r\nb\r\nc\r\n") == "\r\n"

    def test_mixed_more_lf(self):
        assert detect_line_ending("a\nb\nc\r\n") == "\n"

    def test_empty(self):
        assert detect_line_ending("") == "\n"


class TestDetectOmissionPlaceholders:
    def test_clean_content_returns_empty(self):
        assert detect_omission_placeholders("def foo():\n    return 1\n") == []

    def test_double_slash_rest(self):
        found = detect_omission_placeholders("// ... rest of code\n")
        assert len(found) > 0

    def test_hash_rest(self):
        found = detect_omission_placeholders("# ... rest of methods\n")
        assert len(found) > 0

    def test_block_comment_ellipsis(self):
        found = detect_omission_placeholders("/* ... */")
        assert len(found) > 0

    def test_remaining_comment(self):
        found = detect_omission_placeholders("// remaining methods unchanged")
        assert len(found) > 0

    def test_does_not_flag_real_ellipsis_in_string(self):
        # Strings with literal "..." not in a comment should not match
        assert detect_omission_placeholders('msg = "loading..."') == []


class TestUnifiedDiff:
    def test_no_change_produces_empty_diff(self):
        assert create_unified_diff("x.py", "a\n", "a\n") == ""

    def test_change_produces_diff(self):
        diff = create_unified_diff("x.py", "a\n", "b\n")
        assert "-a" in diff
        assert "+b" in diff


class TestDiffStat:
    def test_pure_addition(self):
        adds, dels = get_diff_stat("", "a\nb\n")
        assert adds >= 1
        assert dels == 0

    def test_pure_deletion(self):
        adds, dels = get_diff_stat("a\nb\n", "")
        assert dels >= 1
        assert adds == 0

    def test_replace_counts_both(self):
        adds, dels = get_diff_stat("a\n", "b\n")
        assert adds >= 1 and dels >= 1


# ---------------------------------------------------------------------------
# WriteFileToolInvocation
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_make_relative_within_base(self, tmp_path):
        assert make_relative(str(tmp_path / "a.txt"), str(tmp_path)) == "a.txt"

    def test_make_relative_value_error_fallback(self, monkeypatch):
        monkeypatch.setattr(os.path, "relpath",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
        assert make_relative("/foo/bar", "/baz") == "/foo/bar"

    def test_shorten_path_short_unchanged(self):
        assert shorten_path("a/b.py") == "a/b.py"

    def test_shorten_path_long_truncated(self):
        long = "/".join(["d"] * 50) + "/f.py"
        out = shorten_path(long, max_length=40)
        assert "/.../" in out
        assert out.endswith("f.py")

    def test_shorten_path_remaining_le_zero(self):
        long = "/".join(["d"] * 50) + "/longfilename.py"
        out = shorten_path(long, max_length=5)
        assert out == ".../longfilename.py"

    def test_shorten_path_two_parts_unchanged(self):
        assert shorten_path("d/f.py", max_length=3) == "d/f.py"


class TestDiffContextSnippet:
    def test_contains_changes(self):
        out = get_diff_context_snippet("a\nb\nc\n", "a\nB\nc\n")
        assert "-b" in out
        assert "+B" in out

    def test_no_change_empty(self):
        assert get_diff_context_snippet("a\n", "a\n") == ""


class TestWriteFileToolInvocation:
    def test_creates_new_file(self, tmp_path):
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="new.txt", content="hello\n"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is None
        assert (tmp_path / "new.txt").read_text() == "hello\n"
        assert result.file_diff is not None
        assert result.file_diff.is_new_file is True

    def test_overwrites_existing_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("old\n")
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="new\n"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is None
        assert f.read_text() == "new\n"
        assert result.file_diff.is_new_file is False

    def test_creates_parent_directories(self, tmp_path):
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="a/b/c.txt", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is None
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "x"

    def test_rejects_empty_file_path(self, tmp_path):
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.INVALID_PARAMS

    def test_rejects_directory_target(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="subdir", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.TARGET_IS_DIRECTORY

    def test_rejects_omission_placeholder(self, tmp_path):
        inv = WriteFileToolInvocation(
            WriteFileToolParams(
                file_path="bad.py",
                content="def foo():\n    pass\n# ... rest of methods\n",
            ),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.OMISSION_DETECTED

    def test_diff_metadata_populated(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("alpha\n")
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="beta\n"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        diff = result.file_diff
        assert diff is not None
        assert diff.original_content == "alpha\n"
        assert diff.new_content == "beta\n"
        assert diff.additions >= 1
        assert diff.deletions >= 1

    def test_config_blocks_write_outside_workspace(self, tmp_path):
        class BlockingConfig:
            def get_target_dir(self):
                return str(tmp_path)

            def validate_path_access(self, path, mode="write"):
                return "Path is outside the workspace."

        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="hi"),
            target_dir=str(tmp_path),
            config=BlockingConfig(),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_IN_WORKSPACE

    def test_get_description_includes_path(self, tmp_path):
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="hello.txt", content="x"),
            target_dir=str(tmp_path),
        )
        assert "hello.txt" in inv.get_description()


# ---------------------------------------------------------------------------
# WriteFileModifyContext
# ---------------------------------------------------------------------------


class TestWriteFileErrorPaths:
    def test_makedirs_failure_returns_error(self, tmp_path, monkeypatch):
        def boom(path, exist_ok=False):
            raise OSError("dir creation failed")

        monkeypatch.setattr("os.makedirs", boom)
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="subdir/x.txt", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.FILE_WRITE_FAILURE

    def test_permission_error_on_write(self, tmp_path, monkeypatch):
        original_open = open

        def fake_open(path, *args, **kwargs):
            if "w" in (args[0] if args else kwargs.get("mode", "")):
                raise PermissionError("denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PERMISSION_DENIED

    def test_no_space_left_errno(self, tmp_path, monkeypatch):
        original_open = open

        def fake_open(path, *args, **kwargs):
            if "w" in (args[0] if args else kwargs.get("mode", "")):
                err = OSError("no space")
                err.errno = errno.ENOSPC
                raise err
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error.type == ToolErrorType.NO_SPACE_LEFT

    def test_eisdir_errno_on_write(self, tmp_path, monkeypatch):
        original_open = open

        def fake_open(path, *args, **kwargs):
            if "w" in (args[0] if args else kwargs.get("mode", "")):
                err = OSError("is a dir")
                err.errno = errno.EISDIR
                raise err
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="x"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error.type == ToolErrorType.TARGET_IS_DIRECTORY

    def test_existing_file_unreadable_continues(self, tmp_path, monkeypatch):
        # Create a file, then fake OSError on read open
        f = tmp_path / "x.txt"
        f.write_text("orig")
        original_open = open
        read_attempts = []

        def fake_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "")
            if str(path) == str(f) and "r" in mode and "w" not in mode and len(read_attempts) == 0:
                read_attempts.append(1)
                raise OSError("unreadable")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        inv = WriteFileToolInvocation(
            WriteFileToolParams(file_path="x.txt", content="new"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        # Should succeed despite unreadable original
        assert result.error is None
        assert f.read_text() == "new"


class TestWriteFileModifyContextSync:
    """Test the modify-context async methods using asyncio.run."""

    def test_get_file_path_returns_param(self, tmp_path):
        ctx = WriteFileModifyContext(target_dir=str(tmp_path))
        params = WriteFileToolParams(file_path="x.txt", content="")
        assert ctx.get_file_path(params) == "x.txt"

    def test_create_updated_params_marks_modified_by_user(self, tmp_path):
        ctx = WriteFileModifyContext(target_dir=str(tmp_path))
        original = WriteFileToolParams(file_path="x.txt", content="ai content")
        updated = ctx.create_updated_params("orig", "user content", original)
        assert updated.modified_by_user is True
        assert updated.content == "user content"
        assert updated.ai_proposed_content == "ai content"
        assert updated.file_path == "x.txt"

    def test_get_current_content_for_missing_file(self, tmp_path):
        import asyncio
        ctx = WriteFileModifyContext(target_dir=str(tmp_path))
        params = WriteFileToolParams(file_path="missing.txt", content="")
        assert asyncio.run(ctx.get_current_content(params)) == ""

    def test_get_current_content_reads_existing(self, tmp_path):
        import asyncio
        f = tmp_path / "x.txt"
        f.write_text("content here")
        ctx = WriteFileModifyContext(target_dir=str(tmp_path))
        params = WriteFileToolParams(file_path="x.txt", content="")
        assert asyncio.run(ctx.get_current_content(params)) == "content here"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestWriteFileRegistry:
    def test_execute_via_registry(self, tmp_path):
        write_file.configure(target_dir=str(tmp_path))
        try:
            result = write_file.execute("note.txt", "body\n")
            assert result.error is None
            assert (tmp_path / "note.txt").read_text() == "body\n"
        finally:
            write_file.configure(target_dir=None)

    def test_registered_as_tool_not_readonly(self):
        primitive = write_file.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is False
