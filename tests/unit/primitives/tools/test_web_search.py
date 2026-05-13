"""Unit tests for the web_search primitive."""

import asyncio

import pytest

from agentcompose.primitives.tools.web_search import (
    GroundingChunk,
    GroundingSupport,
    SearchResponse,
    ToolErrorType,
    WebSearchToolInvocation,
    WebSearchToolParams,
    format_source_list,
    insert_citations,
    web_search,
)


# ---------------------------------------------------------------------------
# insert_citations
# ---------------------------------------------------------------------------


class TestInsertCitations:
    def test_no_supports_returns_text_unchanged(self):
        out = insert_citations("hello world", [], [])
        assert out == "hello world"

    def test_inserts_marker_at_byte_position(self):
        text = "hello world"
        chunks = [GroundingChunk(uri="u", title="t")]
        # end_index 5 = after "hello"
        supports = [GroundingSupport(start_index=0, end_index=5, chunk_indices=[0])]
        out = insert_citations(text, chunks, supports)
        assert out == "hello[1] world"

    def test_multiple_chunk_indices_concatenate(self):
        text = "hello"
        chunks = [GroundingChunk(uri="a"), GroundingChunk(uri="b")]
        supports = [GroundingSupport(start_index=0, end_index=5, chunk_indices=[0, 1])]
        out = insert_citations(text, chunks, supports)
        assert out == "hello[1][2]"

    def test_handles_unicode_byte_positions(self):
        # "héllo" — é is 2 bytes in UTF-8, so byte position 6 = end of word
        text = "héllo"
        chunks = [GroundingChunk(uri="u")]
        supports = [GroundingSupport(start_index=0, end_index=6, chunk_indices=[0])]
        out = insert_citations(text, chunks, supports)
        assert out == "héllo[1]"

    def test_supports_without_chunk_indices_skipped(self):
        chunks = [GroundingChunk(uri="u")]
        supports = [GroundingSupport(start_index=0, end_index=3, chunk_indices=[])]
        out = insert_citations("abc def", chunks, supports)
        assert out == "abc def"


# ---------------------------------------------------------------------------
# format_source_list
# ---------------------------------------------------------------------------


class TestFormatSourceList:
    def test_empty(self):
        assert format_source_list([]) == ""

    def test_numbered_format(self):
        chunks = [
            GroundingChunk(title="First", uri="https://a.com"),
            GroundingChunk(title="Second", uri="https://b.com"),
        ]
        out = format_source_list(chunks)
        assert "[1] First (https://a.com)" in out
        assert "[2] Second (https://b.com)" in out

    def test_missing_title_uses_placeholder(self):
        out = format_source_list([GroundingChunk(uri="https://x.com")])
        assert "Untitled" in out


# ---------------------------------------------------------------------------
# WebSearchToolInvocation
# ---------------------------------------------------------------------------


class _StubBackend:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def search(self, query):
        if self.error:
            raise self.error
        return self.response


class TestWebSearchValidation:
    def test_empty_query_rejected(self):
        inv = WebSearchToolInvocation(WebSearchToolParams(query="  "))
        result = asyncio.run(inv.execute())
        assert result.error is not None
        assert result.error.type == ToolErrorType.INVALID_PARAMS

    def test_no_backend_returns_error(self):
        inv = WebSearchToolInvocation(WebSearchToolParams(query="anything"))
        result = asyncio.run(inv.execute())
        assert result.error is not None
        assert result.error.type == ToolErrorType.WEB_SEARCH_FAILED


class TestWebSearchExecute:
    def test_returns_results_with_sources(self):
        response = SearchResponse(
            text="Python is great",
            chunks=[GroundingChunk(uri="https://python.org", title="Home")],
            supports=[GroundingSupport(start_index=0, end_index=6, chunk_indices=[0])],
        )
        inv = WebSearchToolInvocation(
            WebSearchToolParams(query="python"),
            backend=_StubBackend(response=response),
        )
        result = asyncio.run(inv.execute())
        assert result.error is None
        assert "Python[1]" in result.llm_content
        assert "Sources:" in result.llm_content
        assert result.sources == response.chunks

    def test_empty_text_response(self):
        response = SearchResponse(text="")
        inv = WebSearchToolInvocation(
            WebSearchToolParams(query="x"),
            backend=_StubBackend(response=response),
        )
        result = asyncio.run(inv.execute())
        assert result.error is None
        assert "No search results" in result.llm_content

    def test_backend_exception_returns_error(self):
        inv = WebSearchToolInvocation(
            WebSearchToolParams(query="x"),
            backend=_StubBackend(error=RuntimeError("boom")),
        )
        result = asyncio.run(inv.execute())
        assert result.error is not None
        assert result.error.type == ToolErrorType.WEB_SEARCH_FAILED

    def test_get_description(self):
        inv = WebSearchToolInvocation(WebSearchToolParams(query="dogs"))
        assert "dogs" in inv.get_description()


class TestWebSearchRegistry:
    def test_registered_as_tool_readonly(self):
        primitive = web_search.get("execute")
        assert primitive is not None
        assert primitive.kind == "tool"
        assert primitive.readonly is True
