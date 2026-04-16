"""
Web search primitive.

Ported from the Google ADK TypeScript WebSearchTool/WebSearchToolInvocation.
Provides web searching with:
- Pluggable search backend via protocol
- Grounding metadata with source citations
- Inline citation marker insertion at UTF-8 byte positions
- Formatted source list appended to results
- Cancellation support
- Formatted results for LLM consumption
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

web_search = PrimitiveRegistry("web_search")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of web search errors."""

    WEB_SEARCH_FAILED = "web_search_failed"
    INVALID_PARAMS = "invalid_params"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WebSearchToolParams:
    """Parameters for a web search invocation."""

    query: str


@dataclass
class GroundingChunk:
    """A single grounding source returned by the search backend."""

    uri: str = ""
    title: str = ""


@dataclass
class GroundingSupport:
    """Maps a text segment to the grounding chunks that support it."""

    start_index: int = 0
    end_index: int = 0
    chunk_indices: list[int] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Response from a search backend.

    Mirrors the Gemini API grounding metadata structure. Search backend
    implementations should populate ``text``, and optionally
    ``chunks`` and ``supports`` for citation insertion.
    """

    text: str = ""
    chunks: list[GroundingChunk] = field(default_factory=list)
    supports: list[GroundingSupport] = field(default_factory=list)


@dataclass
class ToolError:
    """Structured error from a tool execution."""

    message: str
    type: ToolErrorType


@dataclass
class ToolResult:
    """Result formatted for consumption by an LLM agent and human display."""

    llm_content: str
    return_display: str = ""
    error: Optional[ToolError] = None
    sources: Optional[list[GroundingChunk]] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for pluggable web search backends.

    Implementations might call the Gemini API with grounding, a third-party
    search API (Serper, Brave, Tavily, etc.), or a local index.
    """

    async def search(self, query: str) -> SearchResponse:
        """Execute a search query and return structured results."""
        ...


# ---------------------------------------------------------------------------
# Citation insertion
# ---------------------------------------------------------------------------


def insert_citations(
    text: str,
    chunks: list[GroundingChunk],
    supports: list[GroundingSupport],
) -> str:
    """Insert inline citation markers into the response text.

    The Gemini API returns grounding supports with ``start_index`` and
    ``end_index`` as UTF-8 byte positions. We encode the text to bytes,
    insert markers in descending order (to avoid shifting indices), then
    decode back to a string.

    Args:
        text: The response text to annotate.
        chunks: Grounding source chunks (used for numbering).
        supports: Grounding supports mapping text segments to chunks.

    Returns:
        The text with ``[N]`` citation markers inserted.
    """
    if not chunks or not supports:
        return text

    # Build insertion list: (byte_index, marker_string)
    insertions: list[tuple[int, str]] = []
    for support in supports:
        if not support.chunk_indices:
            continue
        marker = "".join(f"[{idx + 1}]" for idx in support.chunk_indices)
        insertions.append((support.end_index, marker))

    # Sort descending so earlier insertions don't shift later indices
    insertions.sort(key=lambda x: x[0], reverse=True)

    # Work in byte space (segment indices are UTF-8 byte positions)
    text_bytes = text.encode("utf-8")
    parts: list[bytes] = []
    last_index = len(text_bytes)

    for byte_pos, marker in insertions:
        pos = min(byte_pos, last_index)
        parts.insert(0, text_bytes[pos:last_index])
        parts.insert(0, marker.encode("utf-8"))
        last_index = pos

    parts.insert(0, text_bytes[:last_index])
    return b"".join(parts).decode("utf-8")


def format_source_list(chunks: list[GroundingChunk]) -> str:
    """Format grounding chunks as a numbered source list."""
    if not chunks:
        return ""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.title or "Untitled"
        uri = chunk.uri or "No URI"
        lines.append(f"[{i}] {title} ({uri})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WebSearchToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class WebSearchToolInvocation:
    """Manages a single web search with full lifecycle support.

    Mirrors the TypeScript WebSearchToolInvocation class. Handles:
    - Parameter validation
    - Search backend invocation
    - Citation marker insertion into response text
    - Source list formatting
    - Error handling with cancellation support
    - Formatted results for LLM consumption
    """

    def __init__(
        self,
        params: WebSearchToolParams,
        *,
        backend: Optional[SearchBackend] = None,
    ):
        self.params = params
        self.backend = backend

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution."""
        if not self.params.query or not self.params.query.strip():
            return "The 'query' parameter cannot be empty."
        return None

    # -- Description -------------------------------------------------------

    def get_description(self) -> str:
        return f'Searching the web for: "{self.params.query}"'

    # -- Main execution ----------------------------------------------------

    async def execute(self) -> ToolResult:
        """Run the web search and return formatted results."""
        validation_error = self.validate()
        if validation_error:
            return ToolResult(
                llm_content=validation_error,
                return_display="Validation failed.",
                error=ToolError(
                    message=validation_error, type=ToolErrorType.INVALID_PARAMS,
                ),
            )

        if self.backend is None:
            msg = (
                "No search backend configured. Configure one via "
                "web_search.configure(backend=...) before calling execute."
            )
            return ToolResult(
                llm_content=msg,
                return_display="No search backend.",
                error=ToolError(message=msg, type=ToolErrorType.WEB_SEARCH_FAILED),
            )

        try:
            response = await self.backend.search(self.params.query)
        except asyncio.CancelledError:
            return ToolResult(
                llm_content="Web search was cancelled.",
                return_display="Search cancelled.",
            )
        except Exception as e:
            msg = (
                f'Error during web search for query '
                f'"{self.params.query}": {e}'
            )
            logger.warning(msg, exc_info=True)
            return ToolResult(
                llm_content=f"Error: {msg}",
                return_display="Error performing web search.",
                error=ToolError(message=msg, type=ToolErrorType.WEB_SEARCH_FAILED),
            )

        if not response.text or not response.text.strip():
            return ToolResult(
                llm_content=(
                    f'No search results or information found for query: '
                    f'"{self.params.query}"'
                ),
                return_display="No information found.",
            )

        # Insert inline citation markers
        result_text = insert_citations(
            response.text, response.chunks, response.supports,
        )

        # Append source list
        source_list = format_source_list(response.chunks)
        if source_list:
            result_text += f"\n\nSources:\n{source_list}"

        return ToolResult(
            llm_content=(
                f'Web search results for "{self.params.query}":\n\n{result_text}'
            ),
            return_display=f'Search results for "{self.params.query}" returned.',
            sources=response.chunks if response.chunks else None,
        )


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------

@web_search.register(kind="tool", readonly=True)
async def execute(query: str) -> ToolResult:
    """Search the web for information.

    Queries a configured search backend and returns results with source
    citations. The backend is pluggable via
    ``web_search.configure(backend=...)``.

    Args:
        query: The search query string.

    Returns:
        Search results with inline citation markers and a source list.
    """
    ctx = web_search.context
    invocation = WebSearchToolInvocation(
        params=WebSearchToolParams(query=query),
        backend=ctx.get("backend"),
    )
    return await invocation.execute()
