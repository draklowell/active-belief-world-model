import base64
import json
import re
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import UndefinedError
from jsonschema import Draft202012Validator
from PIL import Image

from agents import qwen3vl_cuwm
from agents.base import Agent
from agents.qwen3vl_common import build_computer_use_tool
from agents.qwen3vl_cuwm import Qwen3VLCUWMAgent, build_option_schema
from prompts import render
from world_models import CUWM

CANDIDATES = [
    {
        "thoughts": "The Chrome icon starts the browser.",
        "action": "Click the Chrome icon.",
        "tool_call": {"name": "left_click", "arguments": {"coordinate": [500, 500]}},
    },
    {
        "thoughts": "The task may already be complete.",
        "action": "Report that the task is completed.",
        "tool_call": {"name": "terminate", "arguments": {"status": "success"}},
    },
]


def encode_png(size: tuple[int, int], color: str) -> bytes:
    """Encode a solid PNG fixture at explicitly chosen dimensions."""
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def completion(content: str) -> MagicMock:
    """Build a chat completion substitute returning the given content."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


def count_images(message: dict) -> int:
    """Count image parts in one chat message."""
    return sum(part["type"] == "image_url" for part in message["content"])


class TextModel:
    """Textual transition substitute recording the CUWM prompts."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.screenshots: list[bytes] = []

    def generate(self, screenshot: bytes, prompt: str, *, system_prompt: str) -> str:
        """Record the request and return a numbered transition."""
        self.prompts.append(prompt)
        self.screenshots.append(screenshot)
        return f"Transition {len(self.prompts)}"


class ImageModel:
    """Visual transition substitute returning a green screenshot of the input size."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, screenshot: bytes, prompt: str) -> bytes:
        """Record the request and return a predicted screenshot."""
        self.prompts.append(prompt)
        with Image.open(BytesIO(screenshot)) as image:
            return encode_png(image.size, "green")


@pytest.fixture
def world_model_backends() -> tuple[TextModel, ImageModel]:
    """Provide recording world-model backends."""
    return TextModel(), ImageModel()


@pytest.fixture
def agent_config() -> dict:
    """Request two candidates and one step of history."""
    return {"num_options": 2, "history_n": 1}


@pytest.fixture
def agent(
    monkeypatch: pytest.MonkeyPatch,
    world_model_backends: tuple[TextModel, ImageModel],
    agent_config: dict,
) -> Qwen3VLCUWMAgent:
    """Build the agent with a recording CUWM instead of remote clients."""
    text_model, image_model = world_model_backends
    loaded_paths: list[str] = []
    built_settings: list[dict] = []

    def load_settings(path: str) -> dict:
        """Record the configuration path and return paper prompt settings."""
        loaded_paths.append(path)
        return {"text_model": {}, "image_model": {}, "prompt_set": "cuwm"}

    def build_world_model(settings: dict) -> CUWM:
        """Record the settings and compose recording backends with the requested prompts."""
        built_settings.append(settings)
        return CUWM(text_model, image_model, prompt_set=settings["prompt_set"])

    monkeypatch.setenv("OPENAI_BASE_URL", "https://text.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "text-key")
    monkeypatch.setattr(qwen3vl_cuwm, "load_cuwm_config", load_settings)
    monkeypatch.setattr(qwen3vl_cuwm, "build_cuwm", build_world_model)
    built_agent = Qwen3VLCUWMAgent(agent_config)
    assert loaded_paths == ["config/cuwm.toml"]
    assert built_settings == [{"text_model": {}, "image_model": {}, "prompt_set": "cuwm-qwen3vl"}]
    return built_agent


@pytest.fixture
def computer_use_tool() -> dict:
    """Render the relative-coordinate `computer_use` tool definition."""
    return build_computer_use_tool("relative", 1000, 1000)


@pytest.fixture
def observation() -> dict:
    """Provide a square screenshot observation."""
    return {"screenshot": encode_png((1000, 1000), "white")}


def test_agent_implements_the_benchmark_protocol(agent: Qwen3VLCUWMAgent) -> None:
    """Expose the screenshot and PyAutoGUI interface expected by the benchmark."""
    assert isinstance(agent, Agent)
    assert agent.action_space == "pyautogui"
    assert agent.observation_type == "screenshot"
    assert agent.num_options == 2
    assert agent.history_n == 1
    assert agent.actions == []
    assert agent.screenshots == []


PROMPT_CONTEXTS = {
    "action_option_generation": {
        "instruction": "<g>",
        "actions": {"type": "function"},
        "num_options": 2,
    },
    "action_selection": {"instruction": "<g>", "num_options": 2},
}


@pytest.mark.parametrize("has_history", [True, False])
@pytest.mark.parametrize("name", PROMPT_CONTEXTS)
def test_system_prompts_require_every_variable(name: str, has_history: bool) -> None:
    """Render only with explicit context and reject any missing variable."""
    context = PROMPT_CONTEXTS[name] | {"has_history": has_history}
    prompt = render(f"cuwm-qwen3vl/{name}.j2", **context)
    assert "Office" not in prompt
    assert "control_label" not in prompt
    assert "<g>" in prompt
    assert ("previous step" in prompt.lower()) is has_history
    for missing in context:
        expected_error = (
            (TypeError, "StrictUndefined") if missing == "actions" else (UndefinedError, missing)
        )
        with pytest.raises(expected_error[0], match=expected_error[1]):
            render(
                f"cuwm-qwen3vl/{name}.j2",
                **{key: value for key, value in context.items() if key != missing},
            )


@pytest.mark.parametrize("has_history", [True, False])
def test_generation_prompt_examples_are_valid_tool_calls(has_history: bool) -> None:
    """Show only tool calls named after actions of the Qwen3-VL action space."""
    prompt = render(
        "cuwm-qwen3vl/action_option_generation.j2",
        has_history=has_history,
        **PROMPT_CONTEXTS["action_option_generation"],
    )
    examples = [
        json.loads(example)
        for example in re.findall(
            r"```json\n(.*?)\n```", prompt.split("Task instruction:")[0], re.DOTALL
        )
    ]
    assert [example["tool_call"]["name"] for example in examples] == [
        "left_click",
        "type",
        "key",
        "left_click_drag",
        "terminate",
    ]
    assert all(set(example) == {"thoughts", "action", "tool_call"} for example in examples)
    assert all("action" not in example["tool_call"]["arguments"] for example in examples)


def test_inserted_text_is_not_interpreted_as_a_template() -> None:
    """Keep Jinja-like and JSON instruction text literal in prompts and message fragments."""
    instruction = 'Type {{ literal }} and {"key": [1]} {% raw %}'
    prompt = render(
        "cuwm-qwen3vl/action_selection.j2",
        has_history=True,
        instruction=instruction,
        num_options=3,
    )
    assert f"The task instruction is:\n{instruction}\n" in prompt
    tool_call = {"name": "type", "arguments": {"text": "{{ x }} {% raw %} Don't <b>&"}}
    assert render("cuwm-qwen3vl/history_step.j2", step=4, action=tool_call) == (
        "Step 4 executed action:\n" + json.dumps(tool_call)
    )
    assert render("cuwm-qwen3vl/action_option.j2", index=2, action=tool_call) == (
        "Action Option 2:\n - Action:\n```json\n"
        + json.dumps(tool_call)
        + "\n```\n - Predicted State Image: see below."
    )


def test_predict_simulates_every_candidate_and_executes_the_selected_one(
    agent: Qwen3VLCUWMAgent,
    world_model_backends: tuple[TextModel, ImageModel],
    observation: dict,
) -> None:
    """Generate candidates, predict both transitions, and convert only the selection."""
    text_model, image_model = world_model_backends
    responses = [
        completion(json.dumps(CANDIDATES)),
        completion('{"action_idx": 1, "thought": "Chrome must be opened first."}'),
    ]
    with patch.object(
        agent._client.chat.completions, "create", side_effect=responses
    ) as mock_create:
        response, actions = agent.predict("Open example.com", observation)

    assert actions == [f"pyautogui.click({int(500 * 1000 / 999)}, {int(500 * 1000 / 999)})"]
    assert len(text_model.prompts) == 2
    assert json.dumps(CANDIDATES[0]["tool_call"]) in text_model.prompts[0]
    assert "Click the Chrome icon." in text_model.prompts[0]
    assert json.dumps(CANDIDATES[1]["tool_call"]) in text_model.prompts[1]
    assert "World Model assistant for desktop applications" in text_model.prompts[0]
    assert image_model.prompts[0].startswith("Transition 1\n\n")
    with Image.open(BytesIO(text_model.screenshots[0])) as screenshot:
        assert screenshot.size == (992, 992)

    generation, selection = (call.kwargs["messages"] for call in mock_create.call_args_list)
    generation_prompt = generation[0]["content"][0]["text"]
    assert "Task instruction:\nOpen example.com" in generation_prompt
    assert "generate 2 ALTERNATIVE candidates" in generation_prompt
    assert '"name": "computer_use"' in generation_prompt
    assert '"name": "left_click"' in generation_prompt
    assert "Don't click boxes" in generation_prompt
    assert "resolution is 1000x1000" in generation_prompt
    assert count_images(generation[1]) == 1
    selection_prompt = selection[0]["content"][0]["text"]
    assert "Open example.com" in selection_prompt
    assert "between 1 and 2 (inclusive)" in selection_prompt
    assert count_images(selection[1]) == 3
    option_texts = [part["text"] for part in selection[1]["content"] if part["type"] == "text"]
    assert any("Action Option 2:" in text and "terminate" in text for text in option_texts)
    assert not any("Transition" in text for text in option_texts)

    trace = json.loads(response)
    assert [candidate["transition"] for candidate in trace["candidates"]] == [
        "Transition 1",
        "Transition 2",
    ]
    assert agent.actions == [CANDIDATES[0]["tool_call"]]
    assert len(agent.screenshots) == 1


def test_history_includes_executed_actions_within_the_window(
    agent: Qwen3VLCUWMAgent, observation: dict
) -> None:
    """Show only the most recent executed step and its screenshot to both policy requests."""
    agent.actions = [
        {"name": "type", "arguments": {"text": "first"}},
        {"name": "type", "arguments": {"text": "second"}},
    ]
    agent.screenshots = [
        base64.b64encode(encode_png((8, 8), "red")).decode(),
        base64.b64encode(encode_png((8, 8), "blue")).decode(),
    ]
    responses = [
        completion(json.dumps(CANDIDATES)),
        completion('{"action_idx": 2, "thought": "Done."}'),
    ]
    with patch.object(
        agent._client.chat.completions, "create", side_effect=responses
    ) as mock_create:
        _, actions = agent.predict("Open example.com", observation)

    assert actions == ["DONE"]
    generation, selection = (call.kwargs["messages"] for call in mock_create.call_args_list)
    for messages, image_count in ((generation, 2), (selection, 4)):
        texts = [part["text"] for part in messages[1]["content"] if part["type"] == "text"]
        assert 'Step 2 executed action:\n{"name": "type", "arguments": {"text": "second"}}' in texts
        assert not any("first" in text for text in texts)
        assert count_images(messages[1]) == image_count
    assert len(agent.actions) == 3

    agent.reset()
    assert agent.actions == []
    assert agent.screenshots == []


@pytest.mark.parametrize(
    "selection_response",
    [
        "not json",
        '{"action_idx": 3}',
        '{"action_idx": 0}',
        '{"action_idx": true}',
        '{"action_idx": false}',
        '{"action_idx": "1"}',
        '{"action_idx": 1.5}',
        '{"thought": "missing"}',
    ],
)
def test_invalid_selection_executes_nothing(
    agent: Qwen3VLCUWMAgent, observation: dict, selection_response: str
) -> None:
    """Return no commands and keep history unchanged when the selection is unusable."""
    responses = [completion(json.dumps(CANDIDATES)), completion(selection_response)]
    with patch.object(agent._client.chat.completions, "create", side_effect=responses):
        _, actions = agent.predict("Open example.com", observation)
    assert actions == []
    assert agent.actions == []
    assert agent.screenshots == []


@pytest.mark.parametrize("name", PROMPT_CONTEXTS)
def test_prompts_without_history_only_drop_the_history_input(name: str) -> None:
    """Describe no previous steps when K = 0 while keeping the rest of the prompt."""
    with_history = render(f"cuwm-qwen3vl/{name}.j2", has_history=True, **PROMPT_CONTEXTS[name])
    without_history = render(f"cuwm-qwen3vl/{name}.j2", has_history=False, **PROMPT_CONTEXTS[name])
    assert "previous" not in without_history.lower()
    assert without_history.endswith(with_history.splitlines()[-1])
    assert without_history.startswith(with_history.splitlines()[0])


@pytest.mark.parametrize("agent_config", [{"num_options": 2, "history_n": 0}])
def test_zero_history_omits_previous_steps(agent: Qwen3VLCUWMAgent, observation: dict) -> None:
    """Use the history-free prompts and send no previous steps when K = 0."""
    agent.actions = [{"name": "type", "arguments": {"text": "first"}}]
    agent.screenshots = [base64.b64encode(encode_png((8, 8), "red")).decode()]
    responses = [
        completion(json.dumps(CANDIDATES)),
        completion('{"action_idx": 1, "thought": "Open Chrome."}'),
    ]
    with patch.object(
        agent._client.chat.completions, "create", side_effect=responses
    ) as mock_create:
        _, actions = agent.predict("Open example.com", observation)

    assert len(actions) == 1
    generation, selection = (call.kwargs["messages"] for call in mock_create.call_args_list)
    for messages, image_count in ((generation, 1), (selection, 3)):
        assert "previous" not in messages[0]["content"][0]["text"].lower()
        texts = [part["text"] for part in messages[1]["content"] if part["type"] == "text"]
        assert not any("Previous steps" in text or "first" in text for text in texts)
        assert count_images(messages[1]) == image_count


def test_first_step_with_history_starts_at_the_current_screenshot(
    agent: Qwen3VLCUWMAgent, observation: dict
) -> None:
    """Send no history parts before any action is executed, as the prompt describes."""
    responses = [
        completion(json.dumps(CANDIDATES)),
        completion('{"action_idx": 1, "thought": "Open Chrome."}'),
    ]
    with patch.object(
        agent._client.chat.completions, "create", side_effect=responses
    ) as mock_create:
        agent.predict("Open example.com", observation)
    generation = mock_create.call_args_list[0].kwargs["messages"]
    assert "This input is empty at the first step" in generation[0]["content"][0]["text"]
    assert generation[1]["content"][0] == {
        "type": "text",
        "text": "The current screenshot is: See the image below.",
    }
    assert count_images(generation[1]) == 1


def test_unparseable_candidates_skip_simulation(
    agent: Qwen3VLCUWMAgent,
    world_model_backends: tuple[TextModel, ImageModel],
    observation: dict,
) -> None:
    """Avoid world-model and selection requests when no candidate can be parsed."""
    text_model, _ = world_model_backends
    with patch.object(
        agent._client.chat.completions, "create", return_value=completion("I cannot help.")
    ) as mock_create:
        response, actions = agent.predict("Open example.com", observation)
    assert (response, actions) == ("I cannot help.", [])
    assert mock_create.call_count == 1
    assert text_model.prompts == []


def test_parse_candidates_skips_malformed_options_and_truncates(
    agent: Qwen3VLCUWMAgent, computer_use_tool: dict
) -> None:
    """Keep only valid candidates, up to the requested number."""
    options = [
        {
            "action": "Click.",
            "tool_call": {"name": "computer_use", "arguments": {"action": "wait"}},
        },
        {"action": "", "tool_call": CANDIDATES[0]["tool_call"]},
        {"action": "Fly.", "tool_call": {"name": "fly", "arguments": {}}},
        {"action": "Click.", "tool_call": {"name": "left_click", "arguments": {}}},
        {"action": "Wait.", "tool_call": {"name": "wait", "arguments": {"action": "wait"}}},
        "not an object",
        *CANDIDATES,
        option_with("wait", {}),
    ]
    response = "```json\n" + json.dumps(options) + "\n```"
    assert agent.parse_candidates(response, computer_use_tool, (1000, 1000)) == CANDIDATES


def test_parse_candidates_keeps_the_first_of_identical_tool_calls(
    agent: Qwen3VLCUWMAgent, computer_use_tool: dict
) -> None:
    """Drop repeated tool calls before truncation, regardless of their descriptions."""
    duplicate = {
        "thoughts": "Another reason.",
        "action": "Open the browser.",
        "tool_call": {"name": "left_click", "arguments": {"coordinate": [500.0, 500]}},
    }
    options = [CANDIDATES[0], duplicate, CANDIDATES[0], *CANDIDATES[1:]]
    assert agent.parse_candidates(json.dumps(options), computer_use_tool, (1000, 1000)) == (
        CANDIDATES
    )


def test_duplicate_candidates_are_simulated_once(
    agent: Qwen3VLCUWMAgent,
    world_model_backends: tuple[TextModel, ImageModel],
    observation: dict,
) -> None:
    """Run CUWM once per distinct tool call and number the remaining options consecutively."""
    text_model, image_model = world_model_backends
    responses = [
        completion(json.dumps([CANDIDATES[0], CANDIDATES[0], CANDIDATES[1]])),
        completion('{"action_idx": 2, "thought": "Done."}'),
    ]
    with patch.object(
        agent._client.chat.completions, "create", side_effect=responses
    ) as mock_create:
        _, actions = agent.predict("Open example.com", observation)
    assert actions == ["DONE"]
    assert len(text_model.prompts) == len(image_model.prompts) == 2
    selection = mock_create.call_args_list[1].kwargs["messages"]
    assert "between 1 and 2 (inclusive)" in selection[0]["content"][0]["text"]


def option_with(name: object, arguments: object) -> dict:
    """Wrap a tool call in a described option."""
    return {"action": "Act.", "tool_call": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("key", {"keys": ["ctrl", "s"]}),
        ("type", {"text": "hello\n"}),
        ("mouse_move", {"coordinate": [0, 0]}),
        ("left_click", {"coordinate": [999, 999.5]}),
        ("left_click_drag", {"coordinate": [10, 20]}),
        ("right_click", {"coordinate": [1, 2]}),
        ("middle_click", {"coordinate": [1, 2]}),
        ("double_click", {"coordinate": [1, 2]}),
        ("scroll", {"pixels": -5}),
        ("wait", {}),
        ("wait", {"time": 2.5}),
        ("terminate", {"status": "failure"}),
    ],
)
def test_valid_arguments_for_every_action(
    name: str, arguments: dict, computer_use_tool: dict
) -> None:
    """Accept each action with the arguments its PyAutoGUI conversion uses."""
    schema = build_option_schema(computer_use_tool, (1000, 1000))
    assert Draft202012Validator(schema).is_valid(option_with(name, arguments))


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("computer_use", {"action": "wait"}),
        ("fly", {}),
        (["left_click"], {"coordinate": [1, 2]}),
        ("left_click", {}),
        ("left_click", {"coordinate": [1, 2], "action": "left_click"}),
        ("left_click", {"coordinate": [1, 2], "button": "left"}),
        ("left_click", {"coordinate": [1]}),
        ("left_click", {"coordinate": [1000, 2]}),
        ("left_click", {"coordinate": [-1, 2]}),
        ("left_click", {"coordinate": ["1", 2]}),
        ("left_click", {"coordinate": [True, 2]}),
        ("left_click", {"coordinate": "1, 2"}),
        ("left_click", [1, 2]),
        ("left_click_drag", {"coordinate": [1, 2], "duration": 1}),
        ("key", {}),
        ("key", {"keys": []}),
        ("key", {"keys": ["ctrl", " "]}),
        ("key", {"keys": "ctrl+s"}),
        ("type", {}),
        ("type", {"text": ""}),
        ("scroll", {}),
        ("scroll", {"pixels": "5"}),
        ("scroll", {"pixels": True}),
        ("wait", {"time": -1}),
        ("terminate", {}),
        ("terminate", {"status": "done"}),
    ],
)
def test_invalid_arguments_are_rejected(
    name: object, arguments: object, computer_use_tool: dict
) -> None:
    """Reject tool calls that the tool definition or the PyAutoGUI conversion cannot use."""
    schema = build_option_schema(computer_use_tool, (1000, 1000))
    assert not Draft202012Validator(schema).is_valid(option_with(name, arguments))


@pytest.mark.parametrize(
    "option",
    [
        {"tool_call": {"name": "wait", "arguments": {}}},
        {"action": " ", "tool_call": {"name": "wait", "arguments": {}}},
        {"action": "Wait.", "tool_call": {"arguments": {}}},
        {"action": "Wait.", "tool_call": {"name": "wait"}},
        {"action": "Wait.", "thoughts": 1, "tool_call": CANDIDATES[0]["tool_call"]},
        "not an object",
    ],
)
def test_malformed_options_are_rejected(option: object, computer_use_tool: dict) -> None:
    """Require a described tool call with a name, arguments, and string thoughts."""
    schema = build_option_schema(computer_use_tool, (1000, 1000))
    assert not Draft202012Validator(schema).is_valid(option)


def test_whole_number_selection_selects_that_option(agent: Qwen3VLCUWMAgent) -> None:
    """Accept JSON whole numbers such as 2.0 as integer indices."""
    assert agent.parse_selection('{"action_idx": 2.0}', 2) == 1


def test_absolute_coordinates_are_bounded_by_the_processed_screenshot(
    agent: Qwen3VLCUWMAgent, observation: dict
) -> None:
    """Bound absolute coordinates by the processed screenshot shown to the policy."""
    agent.coordinate_type = "absolute"
    candidates = [
        {
            "action": f"Click at {x}.",
            "tool_call": {"name": "left_click", "arguments": {"coordinate": [x, 10]}},
        }
        for x in (991, 992)
    ]
    responses = [
        completion(json.dumps(candidates)),
        completion('{"action_idx": 1, "thought": "Only valid option."}'),
    ]
    with patch.object(agent._client.chat.completions, "create", side_effect=responses):
        response, actions = agent.predict("Click near the edge", observation)
    assert [candidate["action"] for candidate in json.loads(response)["candidates"]] == [
        "Click at 991."
    ]
    assert actions == [f"pyautogui.click({int(991 * 1000 / 992)}, {int(10 * 1000 / 992)})"]
