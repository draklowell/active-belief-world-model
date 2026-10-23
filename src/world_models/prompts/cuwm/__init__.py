import re
from importlib.resources import files

IMAGE_ATTACHMENT_MARKER = "[Current UI screenshot attached as image content]"


def load_paper_prompt(name: str) -> str:
    """Load a complete CUWM paper prompt, removing only its Markdown presentation fence.

    Reject unsupported names or a malformed resource instead of substituting a fallback.

    Args:
        name: One of action_option_generation, action_selection, textual_state_transition,
            or transition_judge, without a filename extension.

    Returns:
        Original prompt text with published examples and placeholders intact.
    """
    if name not in (
        "action_option_generation",
        "action_selection",
        "textual_state_transition",
        "transition_judge",
    ):
        raise ValueError(f"Unsupported CUWM paper prompt: {name!r}.")
    source = files(__package__).joinpath("paper", f"{name}.md").read_text(encoding="utf-8")
    match = re.fullmatch(r"(?P<fence>`{3,})text\n(?P<body>.*)\n(?P=fence)\n", source, re.DOTALL)
    if match is None:
        raise ValueError(f"Invalid Markdown presentation fence in CUWM paper prompt {name!r}.")
    return match["body"]


def render_textual_state_transition(action: str, *, app_name: str, gui_description: str) -> str:
    """Substitute the four Appendix A.5.2 slots without interpreting inserted values.

    Reject blank context or unsupported source placeholders before inference.

    Args:
        action: Complete structured candidate GUI action represented as text.
        app_name: Caller-provided application corresponding to the screenshot.
        gui_description: Function-level description of the action and its arguments.

    Returns:
        Complete paper prompt with an attachment marker in place of the image slot.
    """
    values = {
        "action": action,
        "app_name": app_name,
        "gui_description": gui_description,
        "image": IMAGE_ATTACHMENT_MARKER,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"CUWM requires a nonempty {name}.")
    template = load_paper_prompt("textual_state_transition")
    pattern = r"\{([ \t]*[A-Za-z_][^{}\n]*)\}"
    unsupported = set(re.findall(pattern, template)) - values.keys()
    if unsupported:
        raise ValueError(
            "Unsupported textual transition prompt placeholders: " + ", ".join(sorted(unsupported))
        )
    return re.sub(pattern, lambda match: values[match[1]], template)
