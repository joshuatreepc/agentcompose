"""Text-domain primitives."""

from agentcompose.operators import PrimitiveRegistry

text = PrimitiveRegistry("text")


@text.register(kind="tool")
def word_count(content: str) -> int:
    """Count the number of whitespace-separated words in a string.

    Args:
        content: The input text to count words in.

    Returns:
        The number of words found.
    """
    return len(content.split())
