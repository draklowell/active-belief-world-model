import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path

import pytest

from world_models.prompts import cuwm
from world_models.prompts.cuwm import (
    IMAGE_ATTACHMENT_MARKER,
    load_desktop_transition_prompt,
    load_paper_prompt,
    render_textual_state_transition,
)

PAPER_PROMPTS = {
    "action_option_generation": "7672842a0ec3e84d62f8ea07add31736351b0760815cc8d573ced40dfa1be61d",
    "action_selection": "00632ba0ccc400aefe4f4b9f57d53512450bc564439fa0404e7d7a720f47979b",
    "textual_state_transition": "37397ae3cadd8fb982d9ef5a6d82e0c6bf7b1131ccee43cb254dddee1092ca08",
    "transition_judge": "decc9b8763b4c93e5ca00f4959dac7b92959ecac473b577f1c08100f7d7ea8d8",
}


@pytest.mark.parametrize("name", PAPER_PROMPTS)
def test_complete_paper_prompt_is_a_package_resource(name: str) -> None:
    """Preserve each complete, verified author listing without PDF-only artifacts."""
    resource = files("world_models").joinpath("prompts", "cuwm", "paper", f"{name}.md")
    assert resource.is_file()
    prompt = load_paper_prompt(name)
    assert prompt.strip()
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == PAPER_PROMPTS[name]
    assert not re.search(r"^\s*(?:2[7-9]|3[0-5]|A\.5(?:\.\d)?)\s*$", prompt, re.MULTILINE)
    assert "\f" not in prompt
    assert "\\begin{lstlisting}" not in prompt


def test_action_generation_retains_all_structured_examples() -> None:
    """Preserve both clicks, typing, dragging, and the terminal example as published."""
    prompt = load_paper_prompt("action_option_generation")
    examples = [
        json.loads(example)
        for example in re.findall(r"```json\n(.*?)\n`{3,4}(?:\n|$)", prompt, re.DOTALL)
    ]
    assert [example["tool_call"]["function"] for example in examples] == [
        "click",
        "click",
        "type",
        "drag",
        "",
    ]
    assert all(set(example) == {"thoughts", "tool_call"} for example in examples)
    assert examples[0]["tool_call"]["args"] == {
        "control_label": 15,
        "coordinate": None,
        "button": "left",
    }
    assert examples[1]["tool_call"]["args"]["coordinate"] == [150, 30]
    assert examples[2]["tool_call"]["args"]["keys"] == "Hello World"
    assert examples[2]["tool_call"]["args"]["clear_current_text"] is True
    assert examples[3]["tool_call"]["args"]["start_coordinate"] == [100, 100]
    assert examples[3]["tool_call"]["args"]["end_coordinate"] == [200, 200]
    assert examples[4]["tool_call"] == {"function": "", "args": {}, "status": "FINISH"}
    assert prompt.count("**Annotated current screenshot**") == 2
    assert {"instruction", "a11y", "actions", "num_options"} == set(
        re.findall(r"\{([a-z_][a-z_0-9]*)\}", prompt)
    )


def test_selection_retains_five_options_and_original_field_names() -> None:
    """Preserve all five image references and the original selection schema."""
    prompt = load_paper_prompt("action_selection")
    assert all(f"[Predicted Image {index}]" in prompt for index in range(1, 6))
    assert "[Instruction]" in prompt
    assert "[Current State Screenshot]" in prompt
    assert "predicted_state_image" in prompt
    assert "`predicted_state`" in prompt
    assert '"action_idx": <index of selected action>' in prompt
    assert "'action_idx' MUST be an integer between 1 and 5 (inclusive)." in prompt


def test_judge_retains_all_scoring_dimensions_and_rationales() -> None:
    """Preserve the eight scores, three score values, notes, and published brace quirk."""
    prompt = load_paper_prompt("transition_judge")
    schema = prompt.split("Return JSON with exactly this structure:\n", 1)[1].split(
        "\nNow perform the evaluation.", 1
    )[0]
    assert schema.strip().startswith("{{")
    assert schema.count("<0|0.5|1>") == 8
    assert schema.count("<one short sentence rationale>") == 8
    example = json.loads(schema.replace("{{", "{").replace("}}", "}").replace("<0|0.5|1>", "0.5"))
    dimensions = {
        "app_name",
        "user_action",
        "title_bar",
        "ribbon",
        "main_editing_area",
        "sidebar_pane",
        "navigation_area",
        "status_bar",
    }
    assert set(example) == {"scores", "notes"}
    assert set(example["scores"]) == dimensions
    assert set(example["notes"]) == dimensions
    assert (
        "Dropdown / Popout: presence, anchor, relative position, size, and visible content"
        in prompt
    )
    assert "- 0   = completely incorrect / contradicts GT / missing when GT contains it" in prompt
    assert "- 0.5 = partially correct:" in prompt
    assert "- 1   = fully correct:" in prompt
    assert "{PRED}" in prompt and "{GT}" in prompt


def test_desktop_transition_prompt_drops_only_the_application_context() -> None:
    """Keep the paper's transition instructions while inferring the application visually."""
    paper_prompt = load_paper_prompt("textual_state_transition")
    desktop_prompt = load_desktop_transition_prompt()
    assert {"app_name", "image", "action", "gui_description"} == set(
        re.findall(r"\{([a-z_]+)\}", paper_prompt)
    )
    assert {"image", "action", "gui_description"} == set(
        re.findall(r"\{([a-z_]+)\}", desktop_prompt)
    )
    assert "Office" not in desktop_prompt
    shared_sections = [
        "Your task is to predict and describe the likely appearance of the Next UI Screenshot",
        "Current UI Screenshot:\n{image}\n\nAction:\n{action}",
        "GUI Action Description:\n{gui_description}",
        "Only describe changed parts; for unchanged elements, state explicitly",
        "Do not include reasoning, internal thoughts, or references to the images",
        "Output a single paragraph of vivid, precise visual description",
    ]
    assert all(section in paper_prompt for section in shared_sections)
    assert all(section in desktop_prompt for section in shared_sections)


def test_render_transition_preserves_the_complete_template() -> None:
    """Substitute the three desktop slots without dropping any adapted instructions."""
    action = '{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1, 2]}}'
    description = "Click the File menu."
    prompt = render_textual_state_transition(action, gui_description=description)
    expected = (
        load_desktop_transition_prompt()
        .replace("{image}", IMAGE_ATTACHMENT_MARKER)
        .replace("{action}", action)
        .replace("{gui_description}", description)
    )
    assert prompt == expected
    assert "Current UI Screenshot:\n" + IMAGE_ATTACHMENT_MARKER in prompt
    assert "Do not answer in bullet points." in prompt
    assert 'unchanged elements, state explicitly (e.g., "Toolbar unchanged")' in prompt
    assert 'enclosed in double quotes (e.g., "File", "Edit", "View")' in prompt


def test_render_preserves_json_and_placeholder_like_user_text() -> None:
    """Treat inserted action and description values as opaque text in a single pass."""
    action = '{"name": "computer_use", "arguments": {"action": "type", "text": "{image}"}}'
    description = "Type the literal text {gui_description} without substitution."
    prompt = render_textual_state_transition(action, gui_description=description)
    assert "Action:\n" + action in prompt
    assert "GUI Action Description:\n" + description in prompt


@pytest.mark.parametrize(
    "field, value",
    [
        ("action", ""),
        ("action", "   "),
        ("gui_description", ""),
        ("gui_description", "\n"),
    ],
)
def test_render_rejects_missing_textual_context(field: str, value: str) -> None:
    """Reject empty context explicitly instead of fabricating missing paper inputs."""
    values = {"action": "click()", "gui_description": "Click."}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        render_textual_state_transition(**values)


def test_render_rejects_unknown_source_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unknown template slots before sending an unresolved instruction to inference."""
    template = load_desktop_transition_prompt()
    monkeypatch.setattr(
        cuwm, "load_desktop_transition_prompt", lambda: template + "\n{unsupported}"
    )
    with pytest.raises(ValueError, match="unsupported"):
        render_textual_state_transition("click()", gui_description="Click the control.")


def test_prompt_loading_is_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Load all canonical sources from package resources outside the repository directory."""
    monkeypatch.chdir(tmp_path)
    for name, digest in PAPER_PROMPTS.items():
        assert hashlib.sha256(load_paper_prompt(name).encode()).hexdigest() == digest
    assert load_desktop_transition_prompt().startswith("Your task is to predict")


@pytest.mark.parametrize("name", ["../cuwm", "unknown", "textual_state_transition.md"])
def test_prompt_loading_rejects_unsupported_resource_names(name: str) -> None:
    """Restrict public resource loading to the four documented paper prompts."""
    with pytest.raises(ValueError, match="prompt"):
        load_paper_prompt(name)
