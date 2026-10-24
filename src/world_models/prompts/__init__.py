import re


def remove_presentation_fence(source: str, name: str) -> str:
    """Return the body of a prompt stored inside a single Markdown ``text`` fence.

    The fence keeps Markdown formatters from rewriting prompt text. Reject a source without
    exactly that fence instead of sending Markdown presentation to inference.

    Args:
        source: Complete Markdown source of the prompt resource.
        name: Prompt name used to identify a malformed resource.

    Returns:
        Prompt text between the opening and closing fence lines.
    """
    match = re.fullmatch(r"(?P<fence>`{3,})text\n(?P<body>.*)\n(?P=fence)\n", source, re.DOTALL)
    if match is None:
        raise ValueError(f"Invalid Markdown presentation fence in prompt {name!r}.")
    return match["body"]
