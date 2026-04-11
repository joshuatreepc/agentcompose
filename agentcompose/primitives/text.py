"""Text-domain primitives."""

from agentcompose.operators import PrimitiveRegistry

text = PrimitiveRegistry("text")


@text.register(kind="tool")
def to_lower(content: str) -> str:
    """Lowercase a string.

    Args:
        content: The input text.

    Returns:
        The input with all characters lowercased.
    """
    return content.lower()


@text.register(kind="tool")
def strip_whitespace(content: str) -> str:
    """Remove leading and trailing whitespace from a string.

    Args:
        content: The input text.

    Returns:
        The input with leading and trailing whitespace removed.
    """
    return content.strip()


@text.register(kind="tool")
def word_count(content: str) -> int:
    """Count the number of whitespace-separated words in a string.

    Args:
        content: The input text to count words in.

    Returns:
        The number of words found.
    """
    return len(content.split())


@text.register(kind="tool")
def is_empty(content: str) -> bool:
    """Check whether a string is empty (or whitespace-only).

    Args:
        content: The input text.

    Returns:
        True if the stripped input is empty, False otherwise.
    """
    return not content.strip()


@text.register(kind="tool")
def constant_zero(content: str) -> int:
    """Return ``0`` regardless of input.

    Useful as a branch target when a pipeline needs a placeholder value.

    Args:
        content: Ignored.

    Returns:
        Always ``0``.
    """
    return 0


@text.register(kind="tool")
def is_longer_than(content: str, limit: int = 5) -> bool:
    """Check whether a string's length exceeds ``limit``.

    Args:
        content: The input text.
        limit: Minimum length (exclusive) for the check to be true.

    Returns:
        True if ``len(content) > limit``, False otherwise.
    """
    return len(content) > limit
