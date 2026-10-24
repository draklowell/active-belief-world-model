import base64
import json
import logging
import re

import jsonschema
import openai
from decouple import config

from agents.base import Agent
from agents.qwen3vl_common import (
    build_computer_use_tool,
    build_image_content,
    convert_tool_call_to_pyautogui,
    prepare_screenshot,
    process_image,
    request_chat_completion,
)
from prompts import render
from world_models.config import build_cuwm, load_cuwm_config

logger = logging.getLogger(__name__)

PROMPT_SET = "cuwm-qwen3vl"
RELATIVE_COORDINATE_SPACE = (1000, 1000)
REQUIRED_ARGUMENTS = {
    "key": ["keys"],
    "type": ["text"],
    "mouse_move": ["coordinate"],
    "left_click": ["coordinate"],
    "left_click_drag": ["coordinate"],
    "right_click": ["coordinate"],
    "middle_click": ["coordinate"],
    "double_click": ["coordinate"],
    "scroll": ["pixels"],
    "terminate": ["status"],
}


def format_action(candidate: dict) -> str:
    """Render a candidate in the Qwen3-VL response format used for history and selection.

    Args:
        candidate: Candidate with an `action` description and a `computer_use` `tool_call`.

    Returns:
        Action description followed by the tool call inside <tool_call> tags.
    """
    return render(
        f"{PROMPT_SET}/action.j2",
        action=candidate["action"],
        tool_call=json.dumps(candidate["tool_call"]),
    )


def build_option_schema(computer_use_tool: dict, coordinate_space: tuple[int, int]) -> dict:
    """Build the JSON schema of one action option proposed by the policy.

    Argument names, types, enumerations, and the required `action` come from the tool definition
    shown to the policy. The schema adds the arguments each action's PyAutoGUI conversion uses,
    nonempty keys and text, a nonnegative wait time, and coordinates inside the coordinate space.

    Args:
        computer_use_tool: Tool definition shown to the policy.
        coordinate_space: Width and height of the coordinate frame stated to the policy.

    Returns:
        JSON schema (draft 2020-12) of an option with `thoughts`, `action`, and `tool_call`.
    """
    argument_constraints = {
        "keys": {"minItems": 1, "items": {"type": "string", "pattern": r"\S"}},
        "text": {"minLength": 1},
        "coordinate": {
            "prefixItems": [
                {"type": "number", "minimum": 0, "exclusiveMaximum": bound}
                for bound in coordinate_space
            ],
            "minItems": 2,
            "items": False,
        },
        "time": {"minimum": 0},
    }
    action_requirements = [
        {"if": {"properties": {"action": {"const": action}}}, "then": {"required": required}}
        for action, required in REQUIRED_ARGUMENTS.items()
    ]
    arguments_schema = computer_use_tool["function"]["parameters"] | {"additionalProperties": False}
    return {
        "type": "object",
        "required": ["action", "tool_call"],
        "properties": {
            "thoughts": {"type": "string"},
            "action": {"type": "string", "pattern": r"\S"},
            "tool_call": {
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"const": "computer_use"},
                    "arguments": {
                        "allOf": [
                            arguments_schema,
                            {"properties": argument_constraints},
                            *action_requirements,
                        ]
                    },
                },
            },
        },
    }


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
            renders the prompts without a history input.
        coordinate_type: Either absolute processed-image pixels or relative 1000x1000 coordinates.
        num_options: Number of candidate actions requested per step.
        cuwm_config: Path to the CUWM TOML configuration; its prompt_set is replaced by the
            `cuwm-qwen3vl` templates, which do not require an application name.
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
        self._world_model = build_cuwm(
            load_cuwm_config(self.cuwm_config) | {"prompt_set": PROMPT_SET}
        )

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
            step_text = render(
                f"{PROMPT_SET}/history_step.j2", step=index + 1, action=self.actions[index]
            )
            content.append(build_image_content(self.screenshots[index]))
            content.append({"type": "text", "text": step_text})
        return content

    def _build_current_content(self, processed_image: str) -> list[dict]:
        return [
            {"type": "text", "text": render(f"{PROMPT_SET}/current_screenshot.j2")},
            build_image_content(processed_image),
        ]

    def _build_generation_messages(
        self, instruction: str, processed_image: str, computer_use_tool: dict
    ) -> list[dict]:
        system_prompt = render(
            f"{PROMPT_SET}/action_option_generation.j2",
            has_history=self.history_n > 0,
            instruction=instruction,
            actions=json.dumps(computer_use_tool),
            num_options=self.num_options,
        )
        content = [*self._build_history_content(), *self._build_current_content(processed_image)]
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
        system_prompt = render(
            f"{PROMPT_SET}/action_selection.j2",
            has_history=self.history_n > 0,
            instruction=instruction,
            num_options=len(candidates),
        )
        content = [*self._build_history_content(), *self._build_current_content(processed_image)]
        for index, (candidate, predicted_image) in enumerate(
            zip(candidates, predicted_images), start=1
        ):
            option_text = render(
                f"{PROMPT_SET}/action_option.j2", index=index, action=format_action(candidate)
            )
            content.append({"type": "text", "text": option_text})
            content.append(build_image_content(predicted_image))
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def parse_candidates(
        self, response: str, computer_use_tool: dict, coordinate_space: tuple[int, int]
    ) -> list[dict]:
        """Extract valid candidate actions from the JSON array returned by the policy.

        Elements that do not match build_option_schema are logged and skipped. At most
        `num_options` candidates are kept.

        Args:
            response: Raw completion text of the action option generation request.
            computer_use_tool: Tool definition shown to the policy.
            coordinate_space: Width and height of the coordinate frame stated to the policy.

        Returns:
            Candidates with `thoughts`, `action`, and `tool_call` keys, in generation order.
        """
        match = re.search(r"\[.*\]", response, re.DOTALL)
        try:
            options = json.loads(match[0] if match else "")
        except json.JSONDecodeError as error:
            logger.error("Failed to parse candidate actions: %s", error)
            return []

        option_schema = build_option_schema(computer_use_tool, coordinate_space)
        candidates = []
        for option in options:
            try:
                jsonschema.validate(option, option_schema)
            except jsonschema.ValidationError as error:
                logger.warning("Skipping invalid candidate action %s: %s", option, error.message)
                continue
            candidates.append(
                {
                    "thoughts": option.get("thoughts", ""),
                    "action": option["action"].strip(),
                    "tool_call": {
                        "name": "computer_use",
                        "arguments": option["tool_call"]["arguments"],
                    },
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
        selection_schema = {
            "type": "object",
            "required": ["action_idx"],
            "properties": {
                "action_idx": {"type": "integer", "minimum": 1, "maximum": num_candidates}
            },
        }
        try:
            selection = json.loads(match[0] if match else "")
            jsonschema.validate(selection, selection_schema)
        except json.JSONDecodeError as error:
            logger.error("Failed to parse action selection: %s", error)
            return None
        except jsonschema.ValidationError as error:
            logger.error("Invalid action selection: %s", error.message)
            return None
        return int(selection["action_idx"]) - 1

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
        coordinate_space = (
            processed_size if self.coordinate_type == "absolute" else RELATIVE_COORDINATE_SPACE
        )
        candidates = self.parse_candidates(generation_response, computer_use_tool, coordinate_space)
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
