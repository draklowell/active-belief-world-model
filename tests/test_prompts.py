import json
from pathlib import Path

import pytest
from jinja2 import TemplateNotFound, UndefinedError

from prompts import render

FIXTURES = Path(__file__).parent / "fixtures" / "prompts"
CUWM_SNAPSHOTS = json.loads((FIXTURES / "cuwm.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("snapshot", CUWM_SNAPSHOTS)
def test_transition_matches_pre_migration_output(snapshot: dict) -> None:
    """Preserve every character of existing rendered CUWM prompts."""
    result = render(
        "cuwm/textual_state_transition.j2",
        action=snapshot["action"],
        app_name=snapshot["app_name"],
        gui_description=snapshot["gui_description"],
        image="[Current UI screenshot attached as image content]",
    )
    assert result == snapshot["expected"]


@pytest.mark.parametrize("missing", ["action", "app_name", "gui_description", "image"])
def test_missing_template_variables_fail_explicitly(missing: str) -> None:
    """Reject missing required variables through Jinja2 StrictUndefined."""
    context = {
        "action": "click()",
        "app_name": "Excel",
        "gui_description": "Click.",
        "image": "IMG",
    }
    context.pop(missing)
    with pytest.raises(UndefinedError, match=missing):
        render("cuwm/textual_state_transition.j2", **context)


@pytest.mark.parametrize("name", ["missing.j2", "../pyproject.toml"])
def test_unknown_templates_fail_without_fallback(name: str) -> None:
    """Reject missing or out-of-package templates without a fallback."""
    with pytest.raises(TemplateNotFound):
        render(name)


@pytest.mark.parametrize("description", ["", "Transition.", "<b>{{ literal }}</b>\n\n"])
def test_image_preservation_instruction_is_unchanged(description: str) -> None:
    """Preserve the image instruction and caller whitespace exactly."""
    assert render("cuwm/image_preservation.j2", text_description=description) == (
        description + "\n\n"
        "Preserve the layout, text, icons, colors, and all other GUI elements "
        "unrelated to this transition."
    )


def test_packaged_templates_load_outside_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Discover packaged templates independently of the working directory."""
    monkeypatch.chdir(tmp_path)
    snapshot = CUWM_SNAPSHOTS[1]
    assert (
        render(
            "cuwm/textual_state_transition.j2",
            action=snapshot["action"],
            app_name=snapshot["app_name"],
            gui_description=snapshot["gui_description"],
            image="[Current UI screenshot attached as image content]",
        )
        == snapshot["expected"]
    )


def test_history_preserves_multiline_and_jinja_like_actions() -> None:
    """Render history labels without interpreting or trimming action text."""
    assert render("qwen3vl/history.j2", actions=[], start_step=1) == "None"
    assert (
        render("qwen3vl/history.j2", actions=["First\n", "{{ second }}"], start_step=5)
        == "Step 5: First\n\nStep 6: {{ second }}"
    )


def test_selection_requires_all_five_predicted_images() -> None:
    """Reject an incomplete image list through the template's strict contract."""
    with pytest.raises(UndefinedError):
        render(
            "cuwm/action_selection.j2", instruction="Select", image="IMG", predicted_images=["A"]
        )


def test_tool_schema_preserves_quoted_multiline_and_literal_template_text() -> None:
    """Encode schema strings as JSON without evaluating or escaping away their content."""
    description = 'Click "File" & keep <tag> and {{ literal }}.\nNext line.\n'
    actions = "Use 'quotes', {% raw %}, and \\backslashes.\n"
    tools = json.loads(
        render("qwen3vl/tools.json.j2", description_prompt=description, action_descriptions=actions)
    )
    assert tools["function"]["description"] == description
    assert tools["function"]["parameters"]["properties"]["action"]["description"] == actions
    assert tools["function"]["parameters"]["required"] == ["action"]


@pytest.mark.parametrize("missing", ["description_prompt", "action_descriptions"])
def test_tool_schema_rejects_missing_descriptions(missing: str) -> None:
    """Report the missing schema variable before attempting to parse rendered JSON."""
    context = {"description_prompt": "Desktop.", "action_descriptions": "Click."}
    context.pop(missing)
    with pytest.raises(UndefinedError, match=missing):
        render("qwen3vl/tools.json.j2", **context)
