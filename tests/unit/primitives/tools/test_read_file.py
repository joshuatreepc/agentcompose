"""Unit tests for the read_file primitive."""

import os

import pytest

from agentcompose.primitives.tools.read_file import (
    ReadFileToolInvocation,
    ReadFileToolParams,
    ToolErrorType,
    get_programming_language,
    is_binary_file,
    make_relative,
    process_file_content,
    read_file,
    should_ignore_file,
    shorten_path,
)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


class TestPathUtilities:
    def test_make_relative_within_base(self, tmp_path):
        f = tmp_path / "a" / "b.txt"
        assert make_relative(str(f), str(tmp_path)) == os.path.join("a", "b.txt")

    def test_shorten_path_short_enough(self):
        assert shorten_path("a/b/c.py", max_length=60) == "a/b/c.py"

    def test_shorten_path_truncates_long_paths(self):
        long = "/".join(["dir"] * 30) + "/file.py"
        out = shorten_path(long, max_length=40)
        assert "/.../" in out
        assert out.endswith("file.py")
        assert len(out) <= 40 + len("file.py")

    def test_shorten_path_one_level(self):
        # 2-part paths should be returned unchanged regardless of length
        assert shorten_path("dir/file.py", max_length=5) == "dir/file.py"

    def test_shorten_path_remaining_le_zero(self):
        # If max_length < filename+4, falls to .../filename form
        long = "/".join(["dir"] * 10) + "/file.py"
        out = shorten_path(long, max_length=5)
        assert out == ".../file.py"

    def test_make_relative_falls_back_on_value_error(self, monkeypatch):
        def boom(*a, **kw):
            raise ValueError("different drives")

        monkeypatch.setattr(os.path, "relpath", boom)
        assert make_relative("/x/y", "/a/b") == "/x/y"


class TestBinaryDetection:
    def test_jpg_is_binary(self):
        assert is_binary_file("photo.jpg") is True

    def test_pyc_is_binary(self):
        assert is_binary_file("module.pyc") is True

    def test_py_is_not_binary(self):
        assert is_binary_file("module.py") is False

    def test_no_extension_not_binary(self):
        assert is_binary_file("Makefile") is False

    def test_uppercase_extension_detected(self):
        assert is_binary_file("photo.PNG") is True


class TestIgnorePatterns:
    def test_basename_match(self):
        assert should_ignore_file("/proj/secret.env", ["*.env"]) is True

    def test_path_match(self):
        assert should_ignore_file("/proj/build/x.js", ["*build*"]) is True

    def test_no_match(self):
        assert should_ignore_file("/proj/main.py", ["*.env"]) is False

    def test_empty_pattern_list(self):
        assert should_ignore_file("anything.py", []) is False


class TestLanguageInference:
    def test_python(self):
        assert get_programming_language("script.py") == "python"

    def test_typescript_react(self):
        assert get_programming_language("Component.tsx") == "typescriptreact"

    def test_unknown_extension(self):
        assert get_programming_language("file.xyz") is None

    def test_no_extension(self):
        assert get_programming_language("Dockerfile") is None


# ---------------------------------------------------------------------------
# process_file_content — the read engine
# ---------------------------------------------------------------------------


class TestProcessFileContent:
    def test_reads_full_file_with_line_numbers(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        result = process_file_content(str(f), str(tmp_path))
        assert result.error is None
        assert "1\talpha" in result.llm_content
        assert "2\tbeta" in result.llm_content
        assert "3\tgamma" in result.llm_content
        assert result.original_line_count == 3

    def test_missing_file(self, tmp_path):
        result = process_file_content(str(tmp_path / "missing.txt"), str(tmp_path))
        assert result.error_type == ToolErrorType.FILE_NOT_FOUND

    def test_directory_target(self, tmp_path):
        result = process_file_content(str(tmp_path), str(tmp_path))
        assert result.error_type == ToolErrorType.IS_DIRECTORY

    def test_binary_file_rejected(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = process_file_content(str(f), str(tmp_path))
        assert result.error_type == ToolErrorType.BINARY_FILE
        assert "Binary file" in result.llm_content

    def test_line_range_inclusive(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = process_file_content(str(f), str(tmp_path), start_line=3, end_line=5)
        assert "3\tline3" in result.llm_content
        assert "4\tline4" in result.llm_content
        assert "5\tline5" in result.llm_content
        assert "6\tline6" not in result.llm_content
        assert result.lines_shown == (3, 5)

    def test_max_lines_truncates(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        result = process_file_content(str(f), str(tmp_path), max_lines=5)
        assert result.is_truncated
        assert result.lines_shown == (1, 5)

    def test_max_length_truncates(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("a" * 1000)
        result = process_file_content(str(f), str(tmp_path), max_length=50)
        assert result.is_truncated
        assert len(result.llm_content) <= 50

    def test_start_line_clamped_to_one(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("alpha\nbeta\n")
        result = process_file_content(str(f), str(tmp_path), start_line=0)
        assert "1\talpha" in result.llm_content

    def test_permission_denied_returns_error(self, tmp_path, monkeypatch):
        f = tmp_path / "secret.txt"
        f.write_text("x")
        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == str(f):
                raise PermissionError("no access")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        result = process_file_content(str(f), str(tmp_path))
        assert result.error_type == ToolErrorType.PERMISSION_DENIED

    def test_os_error_returns_read_error(self, tmp_path, monkeypatch):
        f = tmp_path / "x.txt"
        f.write_text("x")
        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == str(f):
                raise OSError("io broken")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        result = process_file_content(str(f), str(tmp_path))
        assert result.error_type == ToolErrorType.READ_ERROR


# ---------------------------------------------------------------------------
# ReadFileToolInvocation
# ---------------------------------------------------------------------------


class TestReadFileToolInvocation:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello\nworld\n")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="hello.txt"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is None
        assert "1\thello" in result.llm_content

    def test_validation_rejects_empty_path(self, tmp_path):
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path=""),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None

    def test_validation_rejects_negative_start_line(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("a")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="x.txt", start_line=0),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert "start_line" in result.error.message

    def test_validation_rejects_negative_end_line(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("a\n")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="x.txt", end_line=0),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert "end_line" in result.error.message

    def test_get_policy_update_options_uses_file_path(self, tmp_path):
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="some/path.py"),
            target_dir=str(tmp_path),
        )
        opts = inv.get_policy_update_options()
        assert opts.args_pattern == "some/path.py"

    def test_validation_rejects_inverted_range(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("a")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="x.txt", start_line=5, end_line=2),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert "greater than" in result.error.message

    def test_truncation_includes_continuation_guidance(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("\n".join(f"l{i}" for i in range(1, 21)))
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="x.txt"),
            target_dir=str(tmp_path),
            max_lines=5,
        )
        result = inv.execute()
        assert "truncated" in result.llm_content.lower()
        assert "start_line: 6" in result.llm_content

    def test_get_description_short_path(self, tmp_path):
        f = tmp_path / "z.txt"
        f.write_text("")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="z.txt"),
            target_dir=str(tmp_path),
        )
        assert inv.get_description() == "z.txt"

    def test_tool_locations_returns_resolved_path(self, tmp_path):
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="z.txt", start_line=3),
            target_dir=str(tmp_path),
        )
        locations = inv.tool_locations()
        assert len(locations) == 1
        assert locations[0].path == os.path.join(str(tmp_path), "z.txt")
        assert locations[0].line == 3

    def test_config_path_access_blocks(self, tmp_path):
        class BlockingConfig:
            def get_target_dir(self):
                return str(tmp_path)

            def validate_path_access(self, path, mode="read"):
                return "Path is outside the workspace."

            def get_ignore_patterns(self):
                return []

        f = tmp_path / "x.txt"
        f.write_text("hi")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="x.txt"),
            target_dir=str(tmp_path),
            config=BlockingConfig(),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_IN_WORKSPACE

    def test_config_ignore_patterns_block(self, tmp_path):
        class IgnoreConfig:
            def get_target_dir(self):
                return str(tmp_path)

            def validate_path_access(self, path, mode="read"):
                return None

            def get_ignore_patterns(self):
                return ["*.env"]

        f = tmp_path / "secret.env"
        f.write_text("X=1")
        inv = ReadFileToolInvocation(
            ReadFileToolParams(file_path="secret.env"),
            target_dir=str(tmp_path),
            config=IgnoreConfig(),
        )
        result = inv.execute()
        assert result.error is not None
        assert "ignored" in result.error.message.lower()


class TestReadFileRegistry:
    def test_execute_via_registry(self, tmp_path, monkeypatch):
        f = tmp_path / "h.txt"
        f.write_text("hello\n")
        # configure target_dir for this call
        read_file.configure(target_dir=str(tmp_path))
        try:
            result = read_file.execute("h.txt")
            assert "hello" in result.llm_content
        finally:
            read_file.configure(target_dir=None)

    def test_registered_as_tool_readonly(self):
        primitive = read_file.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is True
