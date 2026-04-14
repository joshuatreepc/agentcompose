"""
Full-featured web fetching primitive.

Ported from the Google TypeScript WebFetchTool/WebFetchToolInvocation.
Provides URL fetching with:
- URL parsing, validation, and normalization
- Per-host rate limiting with sliding window
- GitHub blob-to-raw URL conversion
- Private/local host blocking
- HTML-to-text conversion
- Content truncation with smart budget allocation (water-filling)
- Retry with exponential backoff
- Binary content handling (images, PDFs, video)
- XML-safe source wrapping for LLM consumption
- Dual LLM/human formatted results
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import logging
import re
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from typing import Optional, Protocol, runtime_checkable
from urllib.parse import urlparse, urlunparse

from agentcompose.core import PrimitiveRegistry

logger = logging.getLogger(__name__)

web_fetch = PrimitiveRegistry("web_fetch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL_FETCH_TIMEOUT_S = 10.0
MAX_CONTENT_LENGTH = 250_000
MAX_EXPERIMENTAL_FETCH_SIZE = 10 * 1024 * 1024  # 10MB
USER_AGENT = (
    "Mozilla/5.0 (compatible; AgentCompose/1.0; "
    "+https://github.com/agentcompose)"
)
TRUNCATION_WARNING = "\n\n... [Content truncated due to size limit] ..."

# Rate limiting
RATE_LIMIT_WINDOW_S = 60.0
MAX_REQUESTS_PER_WINDOW = 10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolErrorType(str, Enum):
    """Classification of web fetch errors."""

    INVALID_PARAMS = "invalid_params"
    BLOCKED_HOST = "blocked_host"
    RATE_LIMITED = "rate_limited"
    FETCH_FAILED = "fetch_failed"
    FALLBACK_FAILED = "fallback_failed"
    PROCESSING_ERROR = "processing_error"
    ALL_SKIPPED = "all_skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WebFetchToolParams:
    """Parameters for a web fetch invocation."""

    prompt: Optional[str] = None
    url: Optional[str] = None


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


@dataclass
class FetchResponse:
    """Simplified HTTP response."""

    status: int
    headers: dict[str, str]
    body: bytes
    content_type: str = ""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class WebFetchConfig(Protocol):
    """Protocol for web fetch configuration providers."""

    def get_max_content_length(self) -> int: ...
    def get_fetch_timeout(self) -> float: ...
    def get_max_retries(self) -> int: ...
    def is_direct_fetch(self) -> bool: ...


# ---------------------------------------------------------------------------
# LRU Rate Limiter
# ---------------------------------------------------------------------------


class _LRURateLimiter:
    """Per-host rate limiter with LRU eviction and sliding time window."""

    def __init__(self, max_hosts: int = 1000):
        self._history: OrderedDict[str, list[float]] = OrderedDict()
        self._max_hosts = max_hosts

    def check(self, url: str) -> tuple[bool, float]:
        """Check if a request to the URL's host is allowed.

        Returns:
            (allowed, wait_time_s). If allowed is False, wait_time_s indicates
            how long to wait before retrying.
        """
        try:
            hostname = urlparse(url).hostname or ""
        except Exception:
            return (True, 0.0)

        now = time.monotonic()
        window_start = now - RATE_LIMIT_WINDOW_S

        history = self._history.get(hostname, [])
        # Prune old timestamps
        history = [t for t in history if t > window_start]

        if len(history) >= MAX_REQUESTS_PER_WINDOW:
            oldest = history[0]
            wait = oldest + RATE_LIMIT_WINDOW_S - now
            self._history[hostname] = history
            self._history.move_to_end(hostname)
            return (False, max(0.0, wait))

        history.append(now)
        self._history[hostname] = history
        self._history.move_to_end(hostname)

        # Evict oldest hosts if over capacity
        while len(self._history) > self._max_hosts:
            self._history.popitem(last=False)

        return (True, 0.0)


_rate_limiter = _LRURateLimiter()


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


def normalize_url(url_str: str) -> str:
    """Normalize a URL: lowercase hostname, strip trailing slash, remove default ports."""
    try:
        parsed = urlparse(url_str)
        hostname = (parsed.hostname or "").lower()
        scheme = parsed.scheme.lower()

        # Remove default ports
        port = parsed.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        netloc = hostname
        if port:
            netloc = f"{hostname}:{port}"
        if parsed.username:
            user_info = parsed.username
            if parsed.password:
                user_info += f":{parsed.password}"
            netloc = f"{user_info}@{netloc}"

        # Strip trailing slash (except root)
        path = parsed.path
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]

        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        return url_str


def parse_prompt(text: str) -> tuple[list[str], list[str]]:
    """Parse text to extract valid URLs and identify malformed ones.

    Returns:
        (valid_urls, errors) tuple.
    """
    tokens = text.split()
    valid_urls: list[str] = []
    errors: list[str] = []

    for token in tokens:
        if not token:
            continue
        if "://" not in token:
            continue

        try:
            parsed = urlparse(token)
            if parsed.scheme in ("http", "https"):
                # Reconstruct to validate
                valid_urls.append(
                    urlunparse((
                        parsed.scheme, parsed.netloc, parsed.path,
                        parsed.params, parsed.query, parsed.fragment,
                    ))
                )
            elif parsed.scheme:
                errors.append(
                    f'Unsupported protocol in URL: "{token}". '
                    f"Only http and https are supported."
                )
        except Exception:
            errors.append(f'Malformed URL detected: "{token}".')

    return valid_urls, errors


def convert_github_url_to_raw(url_str: str) -> str:
    """Convert a GitHub blob URL to a raw.githubusercontent.com URL."""
    try:
        parsed = urlparse(url_str)
        if parsed.hostname == "github.com" and "/blob/" in parsed.path:
            # /owner/repo/blob/branch/path -> /owner/repo/branch/path
            new_path = re.sub(r"^/([^/]+/[^/]+)/blob/", r"/\1/", parsed.path)
            return urlunparse((
                parsed.scheme, "raw.githubusercontent.com", new_path,
                parsed.params, parsed.query, parsed.fragment,
            ))
    except Exception:
        pass
    return url_str


# ---------------------------------------------------------------------------
# Host blocking / private IP detection
# ---------------------------------------------------------------------------


def is_private_ip(url_str: str) -> bool:
    """Check if a URL points to a private or loopback IP address."""
    try:
        hostname = urlparse(url_str).hostname or ""
        # Direct IP check
        try:
            addr = ipaddress.ip_address(hostname)
            return addr.is_private or addr.is_loopback or addr.is_reserved
        except ValueError:
            pass
        # DNS resolution check
        try:
            infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for info in infos:
                addr = ipaddress.ip_address(info[4][0])
                if addr.is_private or addr.is_loopback or addr.is_reserved:
                    return True
        except (socket.gaierror, OSError):
            pass
    except Exception:
        return True
    return False


def is_blocked_host(url_str: str) -> bool:
    """Check if a URL points to localhost, a loopback, or private IP."""
    try:
        hostname = (urlparse(url_str).hostname or "").lower()
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True
        return is_private_ip(url_str)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# HTML to text conversion
# ---------------------------------------------------------------------------


class _HTMLToTextParser(HTMLParser):
    """Simple HTML to text converter."""

    # Tags whose content should be skipped entirely
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}
    # Block-level tags that produce line breaks
    _BLOCK_TAGS = {
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "pre", "section", "article",
        "header", "footer", "nav", "main", "aside", "figure",
        "figcaption", "details", "summary", "dt", "dd",
    }

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in self._BLOCK_TAGS and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        # Collapse runs of whitespace / blank lines
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text, stripping tags and scripts."""
    parser = _HTMLToTextParser()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def truncate_string(text: str, max_length: int, warning: str = TRUNCATION_WARNING) -> str:
    """Truncate text to max_length, appending a warning if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + warning


def sanitize_xml(text: str) -> str:
    """Escape text for safe embedding in XML tags."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ---------------------------------------------------------------------------
# HTTP fetching with timeout and retry
# ---------------------------------------------------------------------------


async def fetch_with_timeout(
    url: str,
    timeout_s: float = URL_FETCH_TIMEOUT_S,
    headers: Optional[dict[str, str]] = None,
    max_size: int = MAX_EXPERIMENTAL_FETCH_SIZE,
) -> FetchResponse:
    """Fetch a URL with timeout and size limit.

    Uses aiohttp if available, falls back to urllib.

    Args:
        url: The URL to fetch.
        timeout_s: Request timeout in seconds.
        headers: Optional HTTP headers.
        max_size: Maximum response body size in bytes.

    Returns:
        A FetchResponse with status, headers, body, and content_type.

    Raises:
        Exception on network errors, timeouts, or size limit exceeded.
    """
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=request_headers) as resp:
                # Check content-length header first
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > max_size:
                    raise ValueError(f"Content exceeds size limit of {max_size} bytes")

                # Stream body with size check
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(8192):
                    total += len(chunk)
                    if total > max_size:
                        raise ValueError(f"Content exceeds size limit of {max_size} bytes")
                    chunks.append(chunk)

                body = b"".join(chunks)
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return FetchResponse(
                    status=resp.status,
                    headers=resp_headers,
                    body=body,
                    content_type=resp_headers.get("content-type", ""),
                )

    except ImportError:
        # Fallback to urllib (synchronous, run in executor)
        import urllib.request
        import urllib.error

        req = urllib.request.Request(url, headers=request_headers)

        def _sync_fetch() -> FetchResponse:
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                    content_length = resp_headers.get("content-length")
                    if content_length and int(content_length) > max_size:
                        raise ValueError(f"Content exceeds size limit of {max_size} bytes")

                    body = resp.read(max_size + 1)
                    if len(body) > max_size:
                        raise ValueError(f"Content exceeds size limit of {max_size} bytes")

                    return FetchResponse(
                        status=resp.status,
                        headers=resp_headers,
                        body=body,
                        content_type=resp_headers.get("content-type", ""),
                    )
            except urllib.error.HTTPError as e:
                body = e.read(max_size) if e.fp else b""
                return FetchResponse(
                    status=e.code,
                    headers={k.lower(): v for k, v in e.headers.items()},
                    body=body,
                    content_type=e.headers.get("content-type", ""),
                )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_fetch)


async def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    on_retry: Optional[callable] = None,  # type: ignore[type-arg]
) -> FetchResponse:
    """Retry an async function with exponential backoff.

    Args:
        fn: Async callable that returns a FetchResponse.
        max_retries: Maximum number of retry attempts.
        base_delay_s: Initial delay between retries.
        max_delay_s: Maximum delay between retries.
        on_retry: Optional callback(attempt, error, delay_s) on each retry.

    Returns:
        The successful FetchResponse.

    Raises:
        The last exception if all retries fail.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            delay = min(base_delay_s * (2 ** attempt), max_delay_s)
            if on_retry:
                on_retry(attempt + 1, e, delay)
            logger.debug(
                "Retry %d/%d after %.1fs: %s", attempt + 1, max_retries, delay, e
            )
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WebFetchToolInvocation — the core execution engine
# ---------------------------------------------------------------------------


class WebFetchToolInvocation:
    """Manages a single web fetch operation with full lifecycle support.

    Mirrors the TypeScript WebFetchToolInvocation class. Handles:
    - URL parsing and validation from prompt text
    - URL normalization and GitHub blob-to-raw conversion
    - Private/local host blocking
    - Per-host rate limiting with sliding window
    - Retry with exponential backoff
    - HTML-to-text conversion for HTML responses
    - Binary content handling (base64 for images/PDFs/video)
    - Content truncation with smart water-filling budget for multi-URL fetches
    - XML-safe source wrapping for LLM consumption
    - Dual LLM/human result formatting
    """

    def __init__(
        self,
        params: WebFetchToolParams,
        *,
        max_content_length: int = MAX_CONTENT_LENGTH,
        max_fetch_size: int = MAX_EXPERIMENTAL_FETCH_SIZE,
        fetch_timeout_s: float = URL_FETCH_TIMEOUT_S,
        max_retries: int = 3,
    ):
        self.params = params
        self.max_content_length = max_content_length
        self.max_fetch_size = max_fetch_size
        self.fetch_timeout_s = fetch_timeout_s
        self.max_retries = max_retries

    # -- Display helpers ---------------------------------------------------

    def get_description(self) -> str:
        """Human-readable description of this invocation."""
        if self.params.url:
            return f"Fetching content from: {self.params.url}"
        prompt = self.params.prompt or ""
        display = prompt[:97] + "..." if len(prompt) > 100 else prompt
        return f'Processing URLs and instructions from prompt: "{display}"'

    # -- Validation --------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Validate parameters before execution.

        Returns an error message string, or None if valid.
        """
        if self.params.url:
            try:
                parsed = urlparse(self.params.url)
                if parsed.scheme not in ("http", "https"):
                    return f'Invalid URL: "{self.params.url}"'
                if not parsed.netloc:
                    return f'Invalid URL: "{self.params.url}"'
            except Exception:
                return f'Invalid URL: "{self.params.url}"'
            return None

        if not self.params.prompt or not self.params.prompt.strip():
            return (
                "The 'prompt' parameter cannot be empty and must contain "
                "URL(s) and instructions."
            )

        valid_urls, errors = parse_prompt(self.params.prompt)

        if errors:
            return "Error(s) in prompt URLs:\n- " + "\n- ".join(errors)

        if not valid_urls:
            return (
                "The 'prompt' must contain at least one valid URL "
                "(starting with http:// or https://)."
            )

        return None

    # -- URL filtering -----------------------------------------------------

    def _filter_and_validate_urls(
        self, urls: list[str]
    ) -> tuple[list[str], list[str]]:
        """Filter URLs, blocking private hosts and rate-limited ones.

        Returns:
            (to_fetch, skipped) tuple.
        """
        unique = list(dict.fromkeys(normalize_url(u) for u in urls))
        to_fetch: list[str] = []
        skipped: list[str] = []

        for url in unique:
            if is_blocked_host(url):
                logger.warning("Blocked access to host: %s", url)
                skipped.append(f"[Blocked Host] {url}")
                continue

            allowed, wait = _rate_limiter.check(url)
            if not allowed:
                logger.warning("Rate limit exceeded for host: %s", url)
                skipped.append(f"[Rate limit exceeded] {url}")
                continue

            to_fetch.append(url)

        return to_fetch, skipped

    # -- Single URL fetch --------------------------------------------------

    async def _fetch_single_url(self, url: str) -> str:
        """Fetch a single URL and return its text content.

        Handles GitHub URL conversion, host blocking, retry, HTML conversion,
        and truncation.

        Raises:
            Exception on fetch failure.
        """
        url = convert_github_url_to_raw(url)
        if is_blocked_host(url):
            raise ValueError(f"Access to blocked or private host {url} is not allowed.")

        response = await retry_with_backoff(
            lambda: fetch_with_timeout(
                url,
                timeout_s=self.fetch_timeout_s,
                max_size=self.max_fetch_size,
            ),
            max_retries=self.max_retries,
        )

        if response.status >= 400:
            raise ValueError(
                f"Request failed with status {response.status}"
            )

        raw_content = response.body.decode("utf-8", errors="replace")
        content_type = response.content_type.lower()

        # Convert HTML to text
        if "text/html" in content_type or not content_type:
            return html_to_text(raw_content)

        return raw_content

    # -- Direct / experimental fetch mode ----------------------------------

    async def _execute_direct(self) -> ToolResult:
        """Fetch a single URL directly (experimental mode)."""
        if not self.params.url:
            return ToolResult(
                llm_content="Error: No URL provided.",
                return_display="Error: No URL provided.",
                error=ToolError(
                    message="No URL provided.", type=ToolErrorType.INVALID_PARAMS
                ),
            )

        try:
            parsed = urlparse(self.params.url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL")
        except Exception:
            msg = f'Invalid URL: "{self.params.url}"'
            return ToolResult(
                llm_content=f"Error: {msg}",
                return_display=f"Error: {msg}",
                error=ToolError(message=msg, type=ToolErrorType.INVALID_PARAMS),
            )

        url = convert_github_url_to_raw(self.params.url)

        if is_blocked_host(url):
            msg = f"Access to blocked or private host {url} is not allowed."
            logger.warning("Blocked direct fetch to host: %s", url)
            return ToolResult(
                llm_content=f"Error: {msg}",
                return_display=f"Error: {msg}",
                error=ToolError(message=msg, type=ToolErrorType.BLOCKED_HOST),
            )

        try:
            response = await retry_with_backoff(
                lambda: fetch_with_timeout(
                    url,
                    timeout_s=self.fetch_timeout_s,
                    headers={
                        "Accept": (
                            "text/markdown, text/plain;q=0.9, "
                            "application/json;q=0.9, text/html;q=0.8, "
                            "application/pdf;q=0.7, */*;q=0.5"
                        ),
                    },
                    max_size=self.max_fetch_size,
                ),
                max_retries=self.max_retries,
            )

            content_type = response.content_type.lower()
            status = response.status

            # Handle error status codes
            if status >= 400:
                raw_text = response.body.decode("utf-8", errors="replace")
                raw_text = truncate_string(
                    raw_text, 10_000, "\n\n... [Error response truncated] ..."
                )
                import json
                headers_str = json.dumps(response.headers, indent=2)
                error_content = (
                    f"Request failed with status {status}\n"
                    f"Headers: {headers_str}\n"
                    f"Response: {raw_text}"
                )
                logger.error("Direct fetch failed with status %d for %s", status, url)
                return ToolResult(
                    llm_content=error_content,
                    return_display=f"Failed to fetch {url} (Status: {status})",
                )

            # Text-like content types
            if any(
                ct in content_type
                for ct in ("text/markdown", "text/plain", "application/json")
            ):
                text = response.body.decode("utf-8", errors="replace")
                text = truncate_string(text, self.max_content_length)
                return ToolResult(
                    llm_content=text,
                    return_display=f"Fetched {content_type} content from {url}",
                )

            # HTML
            if "text/html" in content_type:
                html = response.body.decode("utf-8", errors="replace")
                text = html_to_text(html)
                text = truncate_string(text, self.max_content_length)
                return ToolResult(
                    llm_content=text,
                    return_display=f"Fetched and converted HTML content from {url}",
                )

            # Binary content (images, video, PDF) — return as base64
            if (
                content_type.startswith("image/")
                or content_type.startswith("video/")
                or content_type == "application/pdf"
            ):
                b64_data = base64.b64encode(response.body).decode("ascii")
                mime = content_type.split(";")[0]
                return ToolResult(
                    llm_content=f"[Binary content: {mime}, {len(response.body)} bytes, base64-encoded]\n{b64_data}",
                    return_display=f"Fetched {content_type} from {url}",
                )

            # Fallback: try as text
            text = response.body.decode("utf-8", errors="replace")
            text = truncate_string(text, self.max_content_length)
            return ToolResult(
                llm_content=text,
                return_display=f"Fetched {content_type or 'unknown'} content from {url}",
            )

        except Exception as e:
            msg = f"Error during fetch for {url}: {e}"
            logger.error("Direct fetch error: %s", msg)
            return ToolResult(
                llm_content=f"Error: {msg}",
                return_display=f"Error: {msg}",
                error=ToolError(message=msg, type=ToolErrorType.FETCH_FAILED),
            )

    # -- Multi-URL fallback fetch ------------------------------------------

    async def _execute_fallback(self, urls: list[str]) -> ToolResult:
        """Fetch multiple URLs, aggregate results with budget allocation."""
        unique_urls = list(dict.fromkeys(urls))
        successes: list[tuple[str, str]] = []  # (url, content)
        errors: list[tuple[str, str]] = []     # (url, message)

        for url in unique_urls:
            try:
                content = await self._fetch_single_url(url)
                successes.append((url, content))
            except Exception as e:
                errors.append((url, str(e)))

        # Total failure
        if not successes:
            error_details = ", ".join(f"{u}: {m}" for u, m in errors)
            msg = f"All fetch attempts failed: {error_details}"
            logger.error(msg)
            return ToolResult(
                llm_content=f"Error: {msg}",
                return_display=f"Error: {msg}",
                error=ToolError(message=msg, type=ToolErrorType.FALLBACK_FAILED),
            )

        # Water-filling budget allocation across successes
        sorted_successes = sorted(successes, key=lambda s: len(s[1]))
        remaining_budget = self.max_content_length
        remaining_count = len(sorted_successes)
        allocated: dict[str, str] = {}

        for url, content in sorted_successes:
            fair_share = remaining_budget // remaining_count
            truncated = truncate_string(content, fair_share)
            allocated[url] = truncated
            remaining_budget -= len(truncated)
            remaining_count -= 1

        # Build aggregated XML content preserving original URL order
        source_parts: list[str] = []
        for url in unique_urls:
            if url in allocated:
                source_parts.append(
                    f'<source url="{sanitize_xml(url)}">\n'
                    f"{sanitize_xml(allocated[url])}\n"
                    f"</source>"
                )
            else:
                err_msg = next((m for u, m in errors if u == url), "Unknown error")
                source_parts.append(
                    f'<source url="{sanitize_xml(url)}">\n'
                    f"Error: {sanitize_xml(err_msg)}\n"
                    f"</source>"
                )

        aggregated = "\n".join(source_parts)

        return ToolResult(
            llm_content=aggregated,
            return_display=f"Content fetched from {len(successes)} of {len(unique_urls)} URL(s).",
        )

    # -- Main execution ----------------------------------------------------

    async def execute(self) -> ToolResult:
        """Execute the web fetch and return a formatted ToolResult."""
        # Validate first
        validation_error = self.validate()
        if validation_error:
            return ToolResult(
                llm_content=f"Error: {validation_error}",
                return_display=f"Error: {validation_error}",
                error=ToolError(
                    message=validation_error, type=ToolErrorType.INVALID_PARAMS
                ),
            )

        # Direct mode: single URL fetch
        if self.params.url:
            return await self._execute_direct()

        # Prompt mode: extract and fetch multiple URLs
        prompt = self.params.prompt or ""
        valid_urls, _ = parse_prompt(prompt)
        to_fetch, skipped = self._filter_and_validate_urls(valid_urls)

        # All skipped
        if not to_fetch and skipped:
            msg = f"All requested URLs were skipped: {', '.join(skipped)}"
            logger.error(msg)
            return ToolResult(
                llm_content=f"Error: {msg}",
                return_display=f"Error: {msg}",
                error=ToolError(message=msg, type=ToolErrorType.ALL_SKIPPED),
            )

        result = await self._execute_fallback(to_fetch)

        # Prepend warnings for skipped URLs
        if skipped and not result.error:
            warning = (
                f"[Warning] The following URLs were skipped:\n"
                f"{chr(10).join(skipped)}\n\n"
            )
            result.llm_content = warning + result.llm_content

        return result


# ---------------------------------------------------------------------------
# Registry-registered convenience function
# ---------------------------------------------------------------------------


@web_fetch.register(kind="tool", readonly=True)
async def execute(
    url: Optional[str] = None,
    prompt: Optional[str] = None,
) -> ToolResult:
    """Fetch content from one or more URLs.

    Supports two modes:
    - **Direct**: Pass ``url`` to fetch a single URL directly.
    - **Prompt**: Pass ``prompt`` containing URL(s) and instructions.

    Args:
        url: Direct URL to fetch (single URL mode).
        prompt: Text containing URL(s) and processing instructions.

    Returns:
        The fetched content as text, or an error message.
    """
    ctx = web_fetch.context
    invocation = WebFetchToolInvocation(
        params=WebFetchToolParams(prompt=prompt, url=url),
        max_content_length=ctx.get("max_content_length", MAX_CONTENT_LENGTH),
        fetch_timeout_s=ctx.get("fetch_timeout_s", URL_FETCH_TIMEOUT_S),
        max_retries=ctx.get("max_retries", 3),
    )
    return await invocation.execute()

