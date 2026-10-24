import re
from importlib.resources import files

from world_models.prompts import remove_presentation_fence

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
    return remove_presentation_fence(source, name)


def load_desktop_transition_prompt() -> str:
    """Load the desktop adaptation of the Appendix A.5.2 textual state transition prompt.

    Returns:
        Adapted prompt text with the action, GUI description, and image placeholders intact.
    """
    source = files(__package__).joinpath("textual_state_transition.md").read_text(encoding="utf-8")
    return remove_presentation_fence(source, "textual_state_transition")


def render_textual_state_transition(action: str, *, gui_description: str) -> str:
    """Substitute the desktop transition prompt slots without interpreting inserted values.

    Reject blank context or unsupported source placeholders before inference.

    Args:
        action: Complete structured candidate GUI action represented as text.
        gui_description: Natural-language description of the candidate action.

    Returns:
        Complete adapted prompt with an attachment marker in place of the image slot.
    """
    values = {
        "action": action,
        "gui_description": gui_description,
        "image": IMAGE_ATTACHMENT_MARKER,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"CUWM requires a nonempty {name}.")
    template = load_desktop_transition_prompt()
    pattern = r"\{([ \t]*[A-Za-z_][^{}\n]*)\}"
    unsupported = set(re.findall(pattern, template)) - values.keys()
    if unsupported:
        raise ValueError(
            "Unsupported textual transition prompt placeholders: " + ", ".join(sorted(unsupported))
        )
    return re.sub(pattern, lambda match: values[match[1]], template)
