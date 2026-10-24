import hashlib
import json
from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock

import pytest
from jinja2 import UndefinedError

from prompts import render
from world_models import CUWM
from world_models.cuwm import IMAGE_ATTACHMENT_MARKER

PAPER_PROMPTS = {
    "action_option_generation": "7672842a0ec3e84d62f8ea07add31736351b0760815cc8d573ced40dfa1be61d",
    "action_selection": "00632ba0ccc400aefe4f4b9f57d53512450bc564439fa0404e7d7a720f47979b",
    "textual_state_transition": "37397ae3cadd8fb982d9ef5a6d82e0c6bf7b1131ccee43cb254dddee1092ca08",
    "transition_judge": "decc9b8763b4c93e5ca00f4959dac7b92959ecac473b577f1c08100f7d7ea8d8",
}


PAPER_CONTEXTS = {
    "action_option_generation": {
        "instruction": "{instruction}",
        "a11y": "{a11y}",
        "actions": "{actions}",
        "num_options": "{num_options}",
    },
    "action_selection": {
        "instruction": "[Instruction]",
        "image": "[Current State Screenshot]",
        "predicted_images": [f"[Predicted Image {index}]" for index in range(1, 6)],
    },
    "textual_state_transition": {
        "app_name": "{app_name}",
        "image": "{image}",
        "action": "{action}",
        "gui_description": "{gui_description}",
    },
    "transition_judge": {"PRED": "{PRED}", "GT": "{GT}"},
}


def render_paper_reference(name: str) -> str:
    """Render original input markers for comparisons with canonical source hashes."""
    return render(f"cuwm/{name}.j2", **PAPER_CONTEXTS[name])


@pytest.mark.parametrize("name", PAPER_PROMPTS)
def test_complete_paper_prompt_is_a_package_resource(name: str) -> None:
    """Preserve each complete, verified author listing without PDF-only artifacts."""
    resource = files("prompts").joinpath("templates", "cuwm", f"{name}.j2")
    assert resource.is_file()
    prompt = render_paper_reference(name)
    assert prompt.strip()
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == PAPER_PROMPTS[name]
    assert not any(
        line.strip() in {str(page) for page in range(27, 36)} for line in prompt.splitlines()
    )
    assert "\f" not in prompt
    assert "\\begin{lstlisting}" not in prompt


def test_action_generation_retains_all_structured_examples() -> None:
    """Preserve both clicks, typing, dragging, and the terminal example as published."""
    prompt = render_paper_reference("action_option_generation")
    examples = [
        json.loads(example)
        for example in (block.split("\n```", 1)[0] for block in prompt.split("```json\n")[1:])
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
    assert all(
        "{" + name + "}" in prompt for name in ("instruction", "a11y", "actions", "num_options")
    )


def test_selection_retains_five_options_and_original_field_names() -> None:
    """Preserve all five image references and the original selection schema."""
    prompt = render_paper_reference("action_selection")
    assert all(f"[Predicted Image {index}]" in prompt for index in range(1, 6))
    assert "[Instruction]" in prompt
    assert "[Current State Screenshot]" in prompt
    assert "predicted_state_image" in prompt
    assert "`predicted_state`" in prompt
    assert '"action_idx": <index of selected action>' in prompt
    assert "'action_idx' MUST be an integer between 1 and 5 (inclusive)." in prompt


def test_judge_retains_all_scoring_dimensions_and_rationales() -> None:
    """Preserve the eight scores, three score values, notes, and published brace quirk."""
    prompt = render_paper_reference("transition_judge")
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


def test_render_transition_preserves_the_complete_template() -> None:
    """Substitute the four paper slots without dropping any original instructions."""
    action = '{"function": "click", "args": {"control_label": 15, "button": "left"}}'
    description = "Click accessibility control_label 15 using the left mouse button."
    prompt = render(
        "cuwm/textual_state_transition.j2",
        action=action,
        app_name="Microsoft Excel",
        gui_description=description,
        image=IMAGE_ATTACHMENT_MARKER,
    )
    expected = (
        render_paper_reference("textual_state_transition")
        .replace("{app_name}", "Microsoft Excel")
        .replace("{image}", IMAGE_ATTACHMENT_MARKER)
        .replace("{action}", action)
        .replace("{gui_description}", description)
    )
    assert prompt == expected
    assert prompt.count("Microsoft Excel") == 2
    assert "Current UI Screenshot:\n" + IMAGE_ATTACHMENT_MARKER in prompt
    assert "Do not answer in bullet points." in prompt
    assert 'unchanged elements, state explicitly (e.g., "Ribbon unchanged")' in prompt
    assert 'enclosed in double quotes (e.g., "Home", "File", "New Slide")' in prompt


def test_render_preserves_json_and_placeholder_like_user_text() -> None:
    """Treat inserted action and description values as opaque text in a single pass."""
    action = '{"function": "type", "args": {"keys": "Keep {image} and {app_name}"}}'
    description = "Type the literal text {gui_description} without substitution."
    prompt = render(
        "cuwm/textual_state_transition.j2",
        action=action,
        app_name="Microsoft Word",
        gui_description=description,
        image=IMAGE_ATTACHMENT_MARKER,
    )
    assert "Action:\n" + action in prompt
    assert "GUI Action Description:\n" + description in prompt


@pytest.mark.parametrize(
    "field, value",
    [
        ("action", ""),
        ("action", "   "),
        ("app_name", ""),
        ("app_name", "   "),
        ("gui_description", ""),
        ("gui_description", "\n"),
    ],
)
def test_render_rejects_missing_textual_context(field: str, value: str) -> None:
    """Reject empty context explicitly instead of fabricating missing paper inputs."""
    values = {"action": "click()", "app_name": "Microsoft Excel", "gui_description": "Click."}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        CUWM(text_model=Mock(), image_model=Mock()).generate_text_desc(b"screenshot", **values)


def test_prompt_loading_is_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Load all canonical sources from package resources outside the repository directory."""
    monkeypatch.chdir(tmp_path)
    for name, digest in PAPER_PROMPTS.items():
        assert hashlib.sha256(render_paper_reference(name).encode()).hexdigest() == digest


DESKTOP_CONTEXT = {"image": "{image}", "action": "{action}", "gui_description": "{gui_description}"}


def test_desktop_transition_keeps_the_paper_instructions_without_an_application() -> None:
    """Infer the application visually while keeping the paper's transition instructions."""
    paper_prompt = render_paper_reference("textual_state_transition")
    desktop_prompt = render("cuwm-qwen3vl/textual_state_transition.j2", **DESKTOP_CONTEXT)
    assert "{app_name}" in paper_prompt
    assert "app_name" not in desktop_prompt
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


@pytest.mark.parametrize("missing", DESKTOP_CONTEXT)
def test_desktop_transition_requires_every_slot(missing: str) -> None:
    """Reject a missing transition slot through Jinja2 StrictUndefined."""
    context = {name: value for name, value in DESKTOP_CONTEXT.items() if name != missing}
    with pytest.raises(UndefinedError, match=missing):
        render("cuwm-qwen3vl/textual_state_transition.j2", **context)
