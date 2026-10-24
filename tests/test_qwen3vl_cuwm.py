import base64
import json
import re
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agents import qwen3vl_cuwm
from agents.base import Agent
from agents.qwen3vl_cuwm import Qwen3VLCUWMAgent, load_agent_prompt, render_prompt
from world_models import CUWM

CANDIDATES = [
    {
        "thoughts": "The Chrome icon starts the browser.",
        "action": "Click the Chrome icon.",
        "tool_call": {
            "name": "computer_use",
            "arguments": {"action": "left_click", "coordinate": [500, 500]},
        },
    },
    {
        "thoughts": "The task may already be complete.",
        "action": "Report that the task is completed.",
        "tool_call": {
            "name": "computer_use",
            "arguments": {"action": "terminate", "status": "success"},
        },
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
    monkeypatch.setenv("OPENAI_BASE_URL", "https://text.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "text-key")
    monkeypatch.setattr(qwen3vl_cuwm, "load_cuwm_config", lambda path: loaded_paths.append(path))
    monkeypatch.setattr(
        qwen3vl_cuwm,
        "build_cuwm",
        lambda settings: CUWM(text_model=text_model, image_model=image_model),
    )
    built_agent = Qwen3VLCUWMAgent(agent_config)
    assert loaded_paths == ["config/cuwm.toml"]
    return built_agent


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


@pytest.mark.parametrize(
    "name",
    [
        "action_option_generation",
        "action_selection",
        "action_option_generation_without_history",
        "action_selection_without_history",
    ],
)
def test_prompts_only_use_supported_placeholders(name: str) -> None:
    """Keep JSON examples literal and expose only the placeholders filled by the agent."""
    template = load_agent_prompt(name)
    assert "Office" not in template
    assert "control_label" not in template
    placeholders = set(re.findall(r"\{([a-z_]+)\}", template))
    expected = {"instruction", "num_options"}
    if name.startswith("action_option_generation"):
        expected.add("actions")
    assert placeholders == expected
    rendered = render_prompt(template, {key: f"<{key}>" for key in expected})
    assert not re.findall(r"\{([a-z_]+)\}", rendered)


def test_generation_prompt_examples_are_valid_tool_calls() -> None:
    """Show only `computer_use` tool calls from the Qwen3-VL action space."""
    template = load_agent_prompt("action_option_generation")
    examples = [
        json.loads(example) for example in re.findall(r"```json\n(.*?)\n```", template, re.DOTALL)
    ]
    assert [example["tool_call"]["arguments"]["action"] for example in examples] == [
        "left_click",
        "type",
        "key",
        "left_click_drag",
        "terminate",
    ]
    assert all(set(example) == {"thoughts", "action", "tool_call"} for example in examples)
    assert all(example["tool_call"]["name"] == "computer_use" for example in examples)


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
    assert image_model.prompts[0].startswith("Transition 1\n\n")
    with Image.open(BytesIO(text_model.screenshots[0])) as screenshot:
        assert screenshot.size == (992, 992)

    generation, selection = (call.kwargs["messages"] for call in mock_create.call_args_list)
    generation_prompt = generation[0]["content"][0]["text"]
    assert "Task instruction:\nOpen example.com" in generation_prompt
    assert "generate 2 diverse" in generation_prompt
    assert '"name": "computer_use"' in generation_prompt
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
    assert agent.actions == [
        "Action: Click the Chrome icon.\n<tool_call>\n"
        + json.dumps(CANDIDATES[0]["tool_call"])
        + "\n</tool_call>"
    ]
    assert len(agent.screenshots) == 1


def test_history_includes_executed_actions_within_the_window(
    agent: Qwen3VLCUWMAgent, observation: dict
) -> None:
    """Show only the most recent executed step and its screenshot to both policy requests."""
    agent.actions = ["Action: first", "Action: second"]
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
        assert "Step 2 executed action:\nAction: second" in texts
        assert not any("Action: first" in text for text in texts)
        assert count_images(messages[1]) == image_count
    assert len(agent.actions) == 3

    agent.reset()
    assert agent.actions == []
    assert agent.screenshots == []


@pytest.mark.parametrize(
    "selection_response",
    ["not json", '{"action_idx": 3}', '{"action_idx": 0}', '{"thought": "missing"}'],
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


@pytest.mark.parametrize("name", ["action_option_generation", "action_selection"])
def test_prompts_without_history_only_drop_the_history_input(name: str) -> None:
    """Describe no previous steps when K = 0 while keeping the rest of the prompt."""
    with_history = load_agent_prompt(name)
    without_history = load_agent_prompt(f"{name}_without_history")
    assert "previous step" in with_history.lower()
    assert "previous" not in without_history.lower()
    assert without_history.endswith(with_history.splitlines()[-1])
    assert without_history.startswith(with_history.splitlines()[0])


@pytest.mark.parametrize("agent_config", [{"num_options": 2, "history_n": 0}])
def test_zero_history_omits_previous_steps(agent: Qwen3VLCUWMAgent, observation: dict) -> None:
    """Use the history-free prompts and send no previous steps when K = 0."""
    agent.actions = ["Action: first"]
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
        assert not any("Previous steps" in text or "Action: first" in text for text in texts)
        assert count_images(messages[1]) == image_count


def test_first_step_with_history_reports_no_previous_steps(
    agent: Qwen3VLCUWMAgent, observation: dict
) -> None:
    """Keep the history input with an explicit empty marker before any action is executed."""
    responses = [
        completion(json.dumps(CANDIDATES)),
        completion('{"action_idx": 1, "thought": "Open Chrome."}'),
    ]
    with patch.object(
        agent._client.chat.completions, "create", side_effect=responses
    ) as mock_create:
        agent.predict("Open example.com", observation)
    generation = mock_create.call_args_list[0].kwargs["messages"]
    texts = [part["text"] for part in generation[1]["content"] if part["type"] == "text"]
    assert texts[:2] == ["Previous steps:", "None"]


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
    agent: Qwen3VLCUWMAgent,
) -> None:
    """Keep only valid `computer_use` candidates, up to the requested number."""
    options = [
        {"action": "Click.", "tool_call": {"name": "other", "arguments": {"action": "wait"}}},
        {"action": "", "tool_call": CANDIDATES[0]["tool_call"]},
        {"action": "Fly.", "tool_call": {"name": "computer_use", "arguments": {"action": "fly"}}},
        {"action": "Click.", "tool_call": {"name": "computer_use", "arguments": {}}},
        "not an object",
        *CANDIDATES,
        CANDIDATES[0],
    ]
    response = "```json\n" + json.dumps(options) + "\n```"
    assert agent.parse_candidates(response) == CANDIDATES
