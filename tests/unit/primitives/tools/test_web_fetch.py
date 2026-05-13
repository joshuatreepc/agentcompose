"""Unit tests for the web_fetch primitive.

These tests cover URL parsing, normalization, host blocking, HTML conversion,
rate limiting, and the invocation lifecycle. Network calls are mocked via
monkeypatch — no live HTTP traffic is generated.
"""

import asyncio
import sys

import pytest

from agentcompose.primitives.tools.web_fetch import (
    FetchResponse,
    ToolErrorType,
    WebFetchToolInvocation,
    WebFetchToolParams,
    _LRURateLimiter,
    convert_github_url_to_raw,
    html_to_text,
    is_blocked_host,
    normalize_url,
    parse_prompt,
    sanitize_xml,
    truncate_string,
    web_fetch,
)

# Need the actual submodule to monkeypatch private helpers (the package
# attribute is shadowed by the registry).
_web_fetch_mod = sys.modules["agentcompose.primitives.tools.web_fetch"]


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_lowercases_hostname(self):
        assert normalize_url("https://EXAMPLE.com/path") == "https://example.com/path"

    def test_strips_default_https_port(self):
        assert normalize_url("https://example.com:443/x") == "https://example.com/x"

    def test_strips_default_http_port(self):
        assert normalize_url("http://example.com:80/x") == "http://example.com/x"

    def test_keeps_non_default_port(self):
        assert "8080" in normalize_url("https://example.com:8080/x")

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_keeps_root_slash(self):
        # Root "/" should be preserved
        out = normalize_url("https://example.com/")
        assert out.endswith("example.com/") or out == "https://example.com/"


class TestParsePrompt:
    def test_extracts_https_url(self):
        urls, errors = parse_prompt("Fetch https://example.com/x for me")
        assert urls == ["https://example.com/x"]
        assert errors == []

    def test_rejects_unsupported_scheme(self):
        urls, errors = parse_prompt("ftp://example.com/file")
        assert urls == []
        assert any("Unsupported" in e for e in errors)

    def test_ignores_non_url_tokens(self):
        urls, errors = parse_prompt("Just some words here.")
        assert urls == []
        assert errors == []


class TestConvertGithubUrlToRaw:
    def test_blob_url_converted(self):
        out = convert_github_url_to_raw(
            "https://github.com/owner/repo/blob/main/path/file.py"
        )
        assert "raw.githubusercontent.com" in out
        assert "/owner/repo/main/path/file.py" in out

    def test_non_github_unchanged(self):
        url = "https://example.com/blob/x"
        assert convert_github_url_to_raw(url) == url


class TestIsBlockedHost:
    def test_localhost_blocked(self):
        assert is_blocked_host("http://localhost/x") is True

    def test_loopback_ip_blocked(self):
        assert is_blocked_host("http://127.0.0.1/x") is True

    def test_private_ip_blocked(self):
        assert is_blocked_host("http://192.168.1.1/x") is True


# ---------------------------------------------------------------------------
# HTML to text
# ---------------------------------------------------------------------------


class TestHtmlToText:
    def test_strips_tags(self):
        assert "Hello" in html_to_text("<p>Hello</p>")

    def test_skips_script_content(self):
        out = html_to_text("<script>alert('x')</script><p>visible</p>")
        assert "alert" not in out
        assert "visible" in out

    def test_skips_style_content(self):
        out = html_to_text("<style>.x{}</style><p>text</p>")
        assert "{}" not in out
        assert "text" in out

    def test_block_tags_separate_lines(self):
        out = html_to_text("<p>one</p><p>two</p>")
        assert "one" in out
        assert "two" in out


# ---------------------------------------------------------------------------
# Truncation and XML escape
# ---------------------------------------------------------------------------


class TestTruncateString:
    def test_unchanged_below_threshold(self):
        assert truncate_string("hi", 100) == "hi"

    def test_appends_warning_above_threshold(self):
        out = truncate_string("a" * 50, 10)
        assert out.startswith("a" * 10)
        assert "truncated" in out.lower()


class TestSanitizeXml:
    def test_escapes_brackets(self):
        assert sanitize_xml("<a>") == "&lt;a&gt;"

    def test_escapes_amp(self):
        assert sanitize_xml("&") == "&amp;"

    def test_escapes_quotes(self):
        out = sanitize_xml('he said "hi"')
        assert "&quot;" in out


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = _LRURateLimiter()
        for _ in range(3):
            allowed, _ = limiter.check("https://a.com/x")
            assert allowed

    def test_blocks_over_limit(self):
        limiter = _LRURateLimiter()
        for _ in range(10):
            limiter.check("https://b.com/x")
        allowed, wait = limiter.check("https://b.com/x")
        assert not allowed
        assert wait > 0


# ---------------------------------------------------------------------------
# WebFetchToolInvocation
# ---------------------------------------------------------------------------


class TestWebFetchValidation:
    def test_invalid_url_scheme(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="ftp://example.com")
        )
        assert inv.validate() is not None

    def test_url_without_netloc(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="https://")
        )
        assert inv.validate() is not None

    def test_empty_prompt(self):
        inv = WebFetchToolInvocation(WebFetchToolParams(prompt=""))
        assert inv.validate() is not None

    def test_prompt_without_urls(self):
        inv = WebFetchToolInvocation(WebFetchToolParams(prompt="just text"))
        err = inv.validate()
        assert err is not None
        assert "URL" in err

    def test_prompt_with_url_passes(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(prompt="see https://example.com/x"),
        )
        assert inv.validate() is None

    def test_url_param_passes(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="https://example.com/")
        )
        assert inv.validate() is None


class TestWebFetchExecute:
    def test_blocked_host_returns_error(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="http://localhost/x")
        )
        result = asyncio.run(inv.execute())
        assert result.error is not None
        assert result.error.type == ToolErrorType.BLOCKED_HOST

    def test_invalid_url_returns_error(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="not a url")
        )
        result = asyncio.run(inv.execute())
        assert result.error is not None

    def test_direct_text_fetch_via_mock(self, monkeypatch):
        async def fake_fetch(url, **kwargs):
            return FetchResponse(
                status=200,
                headers={"content-type": "text/plain"},
                body=b"Hello, world!",
                content_type="text/plain",
            )

        monkeypatch.setattr(_web_fetch_mod, "fetch_with_timeout", fake_fetch)
        # Avoid retry delays
        async def passthrough(fn, **kwargs):
            return await fn()
        monkeypatch.setattr(_web_fetch_mod, "retry_with_backoff", passthrough)

        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="https://example.com/x")
        )
        result = asyncio.run(inv.execute())
        assert result.error is None
        assert "Hello, world!" in result.llm_content

    def test_direct_html_fetch_converts_to_text(self, monkeypatch):
        async def fake_fetch(url, **kwargs):
            return FetchResponse(
                status=200,
                headers={"content-type": "text/html"},
                body=b"<html><body><p>hi there</p></body></html>",
                content_type="text/html",
            )

        async def passthrough(fn, **kwargs):
            return await fn()

        monkeypatch.setattr(_web_fetch_mod, "fetch_with_timeout", fake_fetch)
        monkeypatch.setattr(_web_fetch_mod, "retry_with_backoff", passthrough)

        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="https://example.com/x"),
        )
        result = asyncio.run(inv.execute())
        assert result.error is None
        assert "hi there" in result.llm_content
        assert "<html>" not in result.llm_content

    def test_direct_error_status_reported(self, monkeypatch):
        async def fake_fetch(url, **kwargs):
            return FetchResponse(
                status=500,
                headers={"content-type": "text/plain"},
                body=b"server error",
                content_type="text/plain",
            )

        async def passthrough(fn, **kwargs):
            return await fn()

        monkeypatch.setattr(_web_fetch_mod, "fetch_with_timeout", fake_fetch)
        monkeypatch.setattr(_web_fetch_mod, "retry_with_backoff", passthrough)

        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="https://example.com/x"),
        )
        result = asyncio.run(inv.execute())
        # Error status produces a content payload describing the failure
        assert "500" in result.llm_content

    def test_get_description_with_url(self):
        inv = WebFetchToolInvocation(
            WebFetchToolParams(url="https://example.com/x")
        )
        assert "example.com" in inv.get_description()


class TestWebFetchRegistry:
    def test_registered_as_tool_readonly(self):
        primitive = web_fetch.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is True
