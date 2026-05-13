"""Unit tests for the glob primitive."""

import os
import time

from agentcompose.primitives.tools.glob import (
    DEFAULT_EXCLUDES,
    GlobEntry,
    GlobToolInvocation,
    GlobToolParams,
    ToolErrorType,
    glob_tool,
    sort_file_entries,
)


def _touch(path, *, mtime=None, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if mtime is not None:
        os.utime(str(path), (mtime, mtime))


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSortFileEntries:
    def test_recent_files_first_then_alpha(self):
        now = time.time()
        recent_a = GlobEntry("/p/a.py", "a.py", mtime=now - 60)
        recent_b = GlobEntry("/p/b.py", "b.py", mtime=now - 30)
        old_z = GlobEntry("/p/z.py", "z.py", mtime=now - 86400 * 5)
        old_m = GlobEntry("/p/m.py", "m.py", mtime=now - 86400 * 5)

        sorted_entries = sort_file_entries(
            [old_z, recent_a, old_m, recent_b], now,
        )
        # recent files first, ordered newest-first
        assert sorted_entries[0].relative_path == "b.py"
        assert sorted_entries[1].relative_path == "a.py"
        # then old files alphabetically
        assert sorted_entries[2].relative_path == "m.py"
        assert sorted_entries[3].relative_path == "z.py"

    def test_empty_list(self):
        assert sort_file_entries([], time.time()) == []


# ---------------------------------------------------------------------------
# GlobToolInvocation
# ---------------------------------------------------------------------------


class TestGlobToolInvocation:
    def test_finds_files_recursively(self, tmp_path):
        _touch(tmp_path / "a.py")
        _touch(tmp_path / "sub" / "b.py")
        _touch(tmp_path / "sub" / "c.txt")
        # case_sensitive=True uses pathlib.glob, which handles `**/*.py` correctly
        inv = GlobToolInvocation(
            GlobToolParams(pattern="**/*.py", case_sensitive=True),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        rel_paths = {e.relative_path for e in result.entries or []}
        assert "a.py" in rel_paths
        assert os.path.join("sub", "b.py") in rel_paths
        assert os.path.join("sub", "c.txt") not in rel_paths

    def test_returns_no_match_message(self, tmp_path):
        _touch(tmp_path / "a.py")
        inv = GlobToolInvocation(
            GlobToolParams(pattern="**/*.go"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.entries == []
        assert "No files found" in result.llm_content

    def test_excludes_default_dirs(self, tmp_path):
        _touch(tmp_path / "node_modules" / "lib.py")
        _touch(tmp_path / ".git" / "config.py")
        _touch(tmp_path / "__pycache__" / "x.py")
        _touch(tmp_path / "src" / "main.py")

        inv = GlobToolInvocation(
            GlobToolParams(pattern="**/*.py"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        rel_paths = {e.relative_path for e in result.entries or []}
        assert os.path.join("src", "main.py") in rel_paths
        assert not any("node_modules" in p for p in rel_paths)
        assert not any(".git" in p for p in rel_paths)
        assert not any("__pycache__" in p for p in rel_paths)

    def test_case_insensitive_default(self, tmp_path):
        _touch(tmp_path / "sub" / "Hello.PY")
        # `**/*.py` with case-insensitive walk matches nested files (the walk
        # joins rel_root + filename so the path separator is present).
        inv = GlobToolInvocation(
            GlobToolParams(pattern="**/*.py", case_sensitive=False),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        rel_paths = {e.relative_path for e in result.entries or []}
        assert os.path.join("sub", "Hello.PY") in rel_paths

    def test_case_sensitive_flag_filters(self, tmp_path):
        _touch(tmp_path / "sub" / "a.PY")
        _touch(tmp_path / "sub" / "b.py")
        inv = GlobToolInvocation(
            GlobToolParams(pattern="**/*.py", case_sensitive=True),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        rel_paths = {e.relative_path for e in result.entries or []}
        assert os.path.join("sub", "b.py") in rel_paths
        assert os.path.join("sub", "a.PY") not in rel_paths

    def test_dir_path_scopes_search(self, tmp_path):
        _touch(tmp_path / "out.py")
        _touch(tmp_path / "src" / "in.py")
        inv = GlobToolInvocation(
            GlobToolParams(pattern="**/*.py", dir_path="src", case_sensitive=True),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        rel_paths = {e.relative_path for e in result.entries or []}
        assert "in.py" in rel_paths
        assert "out.py" not in rel_paths

    def test_empty_pattern_rejected(self, tmp_path):
        inv = GlobToolInvocation(
            GlobToolParams(pattern=""),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert "empty" in result.error.message.lower()

    def test_missing_dir_path_errors(self, tmp_path):
        inv = GlobToolInvocation(
            GlobToolParams(pattern="*.py", dir_path="does_not_exist"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_FOUND

    def test_dir_path_pointing_to_file_errors(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("")
        inv = GlobToolInvocation(
            GlobToolParams(pattern="*.py", dir_path="x.txt"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_A_DIRECTORY

    def test_exact_file_match(self, tmp_path):
        _touch(tmp_path / "specific.txt", content="hello")
        inv = GlobToolInvocation(
            GlobToolParams(pattern="specific.txt"),
            target_dir=str(tmp_path),
        )
        result = inv.execute()
        rel_paths = {e.relative_path for e in result.entries or []}
        assert "specific.txt" in rel_paths

    def test_get_description(self, tmp_path):
        inv = GlobToolInvocation(
            GlobToolParams(pattern="*.py"),
            target_dir=str(tmp_path),
        )
        assert "*.py" in inv.get_description()

    def test_get_description_with_dir_path(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        inv = GlobToolInvocation(
            GlobToolParams(pattern="*.py", dir_path="src"),
            target_dir=str(tmp_path),
        )
        assert "src" in inv.get_description()

    def test_config_path_access_blocks(self, tmp_path):
        class BlockingConfig:
            def get_target_dir(self):
                return str(tmp_path)

            def validate_path_access(self, path, mode="read"):
                return "Path is outside the workspace."

            def get_ignore_patterns(self):
                return DEFAULT_EXCLUDES

        inv = GlobToolInvocation(
            GlobToolParams(pattern="*.py"),
            target_dir=str(tmp_path),
            config=BlockingConfig(),
        )
        result = inv.execute()
        assert result.error is not None
        assert result.error.type == ToolErrorType.PATH_NOT_IN_WORKSPACE


class TestGlobRegistry:
    def test_execute_via_registry(self, tmp_path):
        _touch(tmp_path / "x.py")
        glob_tool.configure(target_dir=str(tmp_path))
        try:
            result = glob_tool.execute("*.py", case_sensitive=True)
            rel_paths = {e.relative_path for e in result.entries or []}
            assert "x.py" in rel_paths
        finally:
            glob_tool.configure(target_dir=None)

    def test_registered_as_tool_readonly(self):
        primitive = glob_tool.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is True
