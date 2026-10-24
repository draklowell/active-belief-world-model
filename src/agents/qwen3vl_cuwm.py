import base64
import json
import logging
import re
from pathlib import Path

import openai
from decouple import config

from agents.base import Agent
from agents.qwen3vl_common import (
    COMPUTER_USE_ACTIONS,
    build_computer_use_tool,
    build_image_content,
    convert_tool_call_to_pyautogui,
    prepare_screenshot,
    process_image,
    request_chat_completion,
)
from world_models.config import build_cuwm, load_cuwm_config
from world_models.prompts import remove_presentation_fence

logger = logging.getLogger(__name__)

PROMPT_DIRECTORY = Path(__file__).parent / "qwen3vl_cuwm"


def load_agent_prompt(name: str) -> str:
    """Load an agent prompt template stored beside this module.

    Args:
        name: Either action_option_generation or action_selection, optionally followed by
            _without_history, without an extension.

    Returns:
        Prompt template with its named placeholders intact.
    """
    source = (PROMPT_DIRECTORY / f"{name}.md").read_text(encoding="utf-8")
    return remove_presentation_fence(source, name)


def render_prompt(template: str, values: dict[str, str]) -> str:
    """Substitute named placeholders in one pass, leaving JSON braces and inserted text intact.

    Args:
        template: Prompt template containing placeholders such as {instruction}.
        values: Mapping from placeholder names to inserted text.

    Returns:
        Prompt text with every placeholder replaced.
    """
    return re.sub(r"\{([a-z_]+)\}", lambda match: values[match[1]], template)


def format_action(candidate: dict) -> str:
    """Render a candidate in the Qwen3-VL response format used for history and selection.

    Args:
        candidate: Candidate with an `action` description and a `computer_use` `tool_call`.

    Returns:
        Action description followed by the tool call inside <tool_call> tags.
    """
    return (
        f"Action: {candidate['action']}\n"
        f"<tool_call>\n{json.dumps(candidate['tool_call'])}\n</tool_call>"
    )


class Qwen3VLCUWMAgent(Agent):
    """Qwen3-VL agent that simulates candidate actions with CUWM before executing one.

    Each step generates candidate `computer_use` tool calls, predicts each candidate's next
    screenshot with CUWM, and executes the candidate whose predicted outcome the policy selects.

    Attributes:
        model: Served Qwen3-VL identifier used for candidate generation and selection.
        max_tokens: Maximum number of tokens generated per policy request.
        top_p: Nucleus sampling probability mass for policy requests.
        temperature: Sampling temperature for policy requests.
        history_n: Number of previous screenshot and action pairs shown to the policy; zero
            selects prompts without a history input.
        coordinate_type: Either absolute processed-image pixels or relative 1000x1000 coordinates.
        num_options: Number of candidate actions requested per step.
        cuwm_config: Path to the CUWM TOML configuration.
        actions: Executed actions in the Qwen3-VL response format, one per completed step.
        screenshots: Base64 processed screenshots on which each executed action was taken.

    Args:
        agent_config: Agent configuration whose keys override the keyword defaults.
        model: Default served Qwen3-VL identifier.
        max_tokens: Default maximum number of generated tokens.
        top_p: Default nucleus sampling probability mass.
        temperature: Default sampling temperature.
        history_n: Default number of previous steps shown to the policy.
        coordinate_type: Default coordinate convention.
        num_options: Default number of candidate actions.
        cuwm_config: Default path to the CUWM TOML configuration.
    """

    action_space: str = "pyautogui"
    observation_type: str = "screenshot"

    def __init__(
        self,
        agent_config: dict | None = None,
        *,
        model: str = "qwen3-vl",
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        history_n: int = 4,
        coordinate_type: str = "relative",
        num_options: int = 5,
        cuwm_config: str = "config/cuwm.toml",
    ) -> None:
        super().__init__(agent_config or {})
        cfg = self.config
        self.model = cfg.get("model", model)
        self.max_tokens = cfg.get("max_tokens", max_tokens)
        self.top_p = cfg.get("top_p", top_p)
        self.temperature = cfg.get("temperature", temperature)
        self.history_n = cfg.get("history_n", history_n)
        self.coordinate_type = cfg.get("coordinate_type", coordinate_type)
        self.num_options = cfg.get("num_options", num_options)
        self.cuwm_config = cfg.get("cuwm_config", cuwm_config)
        self.actions: list[str] = []
        self.screenshots: list[str] = []

        self._client = openai.OpenAI(
            base_url=config("OPENAI_BASE_URL"), api_key=config("OPENAI_API_KEY")
        )
        self._world_model = build_cuwm(load_cuwm_config(self.cuwm_config))
        prompt_variant = "" if self.history_n > 0 else "_without_history"
        self._generation_template = load_agent_prompt(f"action_option_generation{prompt_variant}")
        self._selection_template = load_agent_prompt(f"action_selection{prompt_variant}")

    def reset(self) -> None:
        """Reset task history after the environment has been reset."""
        self.actions = []
        self.screenshots = []

    def _build_history_content(self) -> list[dict]:
        if self.history_n == 0:
            return []
        first_index = max(0, len(self.actions) - self.history_n)
        content: list[dict] = []
        for index in range(first_index, len(self.actions)):
            content.append({"type": "text", "text": f"Step {index + 1} screenshot:"})
            content.append(build_image_content(self.screenshots[index]))
            content.append(
                {
                    "type": "text",
                    "text": f"Step {index + 1} executed action:\n{self.actions[index]}",
                }
            )
        return [
            {"type": "text", "text": "Previous steps:"},
            *(content or [{"type": "text", "text": "None"}]),
        ]

    def _build_generation_messages(
        self, instruction: str, processed_image: str, computer_use_tool: dict
    ) -> list[dict]:
        system_prompt = render_prompt(
            self._generation_template,
            {
                "instruction": instruction,
                "actions": json.dumps(computer_use_tool),
                "num_options": str(self.num_options),
            },
        )
        content = [
            *self._build_history_content(),
            {"type": "text", "text": "Current screenshot:"},
            build_image_content(processed_image),
        ]
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def _build_selection_messages(
        self,
        instruction: str,
        processed_image: str,
        candidates: list[dict],
        predicted_images: list[str],
    ) -> list[dict]:
        system_prompt = render_prompt(
            self._selection_template,
            {"instruction": instruction, "num_options": str(len(candidates))},
        )
        content = [
            *self._build_history_content(),
            {"type": "text", "text": "The current screenshot is: See the image below."},
            build_image_content(processed_image),
            {"type": "text", "text": "Here are the candidate action options:"},
        ]
        for index, (candidate, predicted_image) in enumerate(
            zip(candidates, predicted_images), start=1
        ):
            option_text = (
                f"Action Option {index}:\n"
                f" - Action:\n{format_action(candidate)}\n"
                " - Predicted State Image: see below."
            )
            content.append({"type": "text", "text": option_text})
            content.append(build_image_content(predicted_image))
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def parse_candidates(self, response: str) -> list[dict]:
        """Extract valid candidate actions from the JSON array returned by the policy.

        Elements without a string `action` and a `computer_use` tool call whose `action`
        argument names a supported action are skipped. At most `num_options` candidates are kept.

        Args:
            response: Raw completion text of the action option generation request.

        Returns:
            Candidates with `thoughts`, `action`, and `tool_call` keys, in generation order.
        """
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match is None:
            logger.error("Candidate response contains no JSON array")
            return []
        try:
            options = json.loads(match[0])
        except json.JSONDecodeError as error:
            logger.error("Failed to parse candidate actions: %s", error)
            return []

        candidates = []
        for option in options if isinstance(options, list) else []:
            tool_call = option.get("tool_call") if isinstance(option, dict) else None
            is_valid = (
                isinstance(tool_call, dict)
                and tool_call.get("name") == "computer_use"
                and isinstance(tool_call.get("arguments"), dict)
                and tool_call["arguments"].get("action") in COMPUTER_USE_ACTIONS
                and isinstance(option.get("action"), str)
                and bool(option["action"].strip())
            )
            if not is_valid:
                logger.warning("Skipping malformed candidate action: %s", option)
                continue
            candidates.append(
                {
                    "thoughts": str(option.get("thoughts", "")),
                    "action": option["action"].strip(),
                    "tool_call": {"name": "computer_use", "arguments": tool_call["arguments"]},
                }
            )
        return candidates[: self.num_options]

    def parse_selection(self, response: str, num_candidates: int) -> int | None:
        """Extract the zero-based index of the selected candidate.

        Args:
            response: Raw completion text of the action selection request.
            num_candidates: Number of candidates shown to the policy.

        Returns:
            Zero-based candidate index, or None when the response has no valid `action_idx`.
        """
        match = re.search(r"\{.*\}", response, re.DOTALL)
        try:
            selection = json.loads(match[0]) if match else None
        except json.JSONDecodeError as error:
            logger.error("Failed to parse action selection: %s", error)
            return None
        action_index = selection.get("action_idx") if isinstance(selection, dict) else None
        if not isinstance(action_index, int) or not 1 <= action_index <= num_candidates:
            logger.error("Invalid action selection: %s", response)
            return None
        return action_index - 1

    def predict(self, instruction: str, observation: dict) -> tuple[str, list[str]]:
        """Generate candidates, simulate them with CUWM, and return the selected action.

        Generate up to `num_options` candidate tool calls from the current screenshot, task
        instruction, and recent history. For each candidate, CUWM predicts a textual transition
        and renders the predicted next screenshot. The policy then selects one candidate from
        the predicted screenshots, which is converted into PyAutoGUI commands. A step whose
        candidates or selection cannot be parsed returns no commands and leaves history intact.

        Args:
            instruction: Task instruction to execute.
            observation: Environment observation containing screenshot bytes.

        Returns:
            JSON trace of candidates, predicted transitions, and selection, and ordered commands.
        """
        processed_image, original_size, processed_size = prepare_screenshot(
            observation["screenshot"]
        )
        computer_use_tool = build_computer_use_tool(self.coordinate_type, *processed_size)
        generation_response = request_chat_completion(
            self._client,
            self._build_generation_messages(instruction, processed_image, computer_use_tool),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        candidates = self.parse_candidates(generation_response)
        if not candidates:
            return generation_response, []

        screenshot = base64.b64decode(processed_image)
        predictions = []
        for index, candidate in enumerate(candidates, start=1):
            logger.info(
                "Simulating candidate %d/%d: %s", index, len(candidates), candidate["action"]
            )
            predictions.append(
                self._world_model.predict(
                    screenshot,
                    json.dumps(candidate["tool_call"]),
                    gui_description=candidate["action"],
                )
            )

        selection_response = request_chat_completion(
            self._client,
            self._build_selection_messages(
                instruction,
                processed_image,
                candidates,
                [process_image(prediction.predicted_screenshot) for prediction in predictions],
            ),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        selected_index = self.parse_selection(selection_response, len(candidates))
        trace = json.dumps(
            {
                "candidates": [
                    candidate | {"transition": prediction.text_description}
                    for candidate, prediction in zip(candidates, predictions)
                ],
                "selection": selection_response,
            },
            indent=2,
        )
        if selected_index is None:
            return trace, []

        selected = candidates[selected_index]
        self.screenshots.append(processed_image)
        self.actions.append(format_action(selected))
        return trace, convert_tool_call_to_pyautogui(
            selected["tool_call"]["arguments"], self.coordinate_type, original_size, processed_size
        )
