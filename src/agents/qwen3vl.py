import json
import logging

import openai
from bs4 import BeautifulSoup
from decouple import config

from agents.base import Agent
from agents.qwen3vl_common import (
    build_computer_use_tool,
    build_image_content,
    convert_tool_call_to_pyautogui,
    prepare_screenshot,
    request_chat_completion,
)
from prompts import render

logger = logging.getLogger(__name__)


class Qwen3VLAgent(Agent):
    """Qwen3-VL screenshot agent for desktop environments using OpenAI-compatible endpoints."""

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
    ) -> None:
        super().__init__(agent_config or {})
        cfg = self.config
        self.model = cfg.get("model", model)
        self.max_tokens = cfg.get("max_tokens", max_tokens)
        self.top_p = cfg.get("top_p", top_p)
        self.temperature = cfg.get("temperature", temperature)
        self.history_n = cfg.get("history_n", history_n)
        self.coordinate_type = cfg.get("coordinate_type", cfg.get("coord", coordinate_type))
        self.actions: list[str] = []
        self.responses: list[str] = []
        self.screenshots: list[str] = []

        base_url = config("OPENAI_BASE_URL")
        api_key = config("OPENAI_API_KEY")
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def reset(self) -> None:
        """Reset task history after the environment has been reset."""
        self.actions = []
        self.responses = []
        self.screenshots = []

    def _build_system_prompt(self, computer_use_tool: dict) -> str:
        return render("qwen3vl/system.j2", tool_definition=json.dumps(computer_use_tool))

    def _build_messages(
        self,
        instruction: str,
        processed_image: str,
        system_prompt: str,
        previous_actions_str: str,
    ) -> list[dict]:
        instruction_prompt = render(
            "qwen3vl/user.j2", instruction=instruction, previous_actions_str=previous_actions_str
        )
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        history_len = min(self.history_n, len(self.responses))
        if history_len > 0:
            history_responses = self.responses[-history_len:]
            history_screenshots = self.screenshots[-history_len - 1 : -1]
        else:
            history_responses = []
            history_screenshots = []

        for index, screenshot_b64 in enumerate(history_screenshots):
            content: list[dict] = [build_image_content(screenshot_b64)]
            if index == 0:
                content.append({"type": "text", "text": instruction_prompt})
            messages.append({"role": "user", "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(history_responses[index])}],
                }
            )

        curr_content: list[dict] = [build_image_content(processed_image)]
        if history_len == 0:
            curr_content.append({"type": "text", "text": instruction_prompt})
        messages.append({"role": "user", "content": curr_content})

        return messages

    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> tuple[str, list[str]]:
        """Parse model response into action description and executable commands.

        Parses the full response text as HTML/XML using BeautifulSoup, extracting
        the action description from the text before and after tool calls and tool
        calls from <tool_call> tags.

        Args:
            response: Raw completion text from model.
            original_width: Native screen width in pixels.
            original_height: Native screen height in pixels.
            processed_width: Resized image width fed to model.
            processed_height: Resized image height fed to model.

        Returns:
            Tuple containing human-readable action description and ordered action strings.
        """
        if not response or not response.strip():
            return "", []

        soup = BeautifulSoup(response, "html.parser")

        pyautogui_code: list[str] = []
        for tool_call_tag in soup.find_all("tool_call"):
            raw = tool_call_tag.decode_contents().strip()
            try:
                tool_call = json.loads(raw)
                if tool_call.get("name") == "computer_use":
                    actions = convert_tool_call_to_pyautogui(
                        tool_call.get("arguments", {}),
                        self.coordinate_type,
                        (original_width, original_height),
                        (processed_width, processed_height),
                    )
                    pyautogui_code.extend(actions)
            except (json.JSONDecodeError, KeyError) as error:
                logger.error("Failed to parse tool call: %s", error)

        action = soup.find_all(string=True, recursive=False)

        clean_text = "\n".join(str(substring).strip() for substring in action)

        return clean_text, pyautogui_code

    def predict(self, instruction: str, observation: dict) -> tuple[str, list[str]]:
        """Generate response and actions for the current screenshot observation.

        Args:
            instruction: Task instruction to execute.
            observation: Environment observation containing screenshot bytes.

        Returns:
            Tuple of model response text and ordered action commands.
        """
        processed_image, (width, height), (processed_width, processed_height) = prepare_screenshot(
            observation["screenshot"]
        )

        self.screenshots.append(processed_image)

        current_step = len(self.actions)
        history_start_index = max(0, current_step - self.history_n)
        previous_actions_str = render(
            "qwen3vl/history.j2",
            actions=self.actions[history_start_index:current_step],
            start_step=history_start_index + 1,
        )

        computer_use_tool = build_computer_use_tool(
            self.coordinate_type, processed_width, processed_height
        )
        system_prompt = self._build_system_prompt(computer_use_tool)
        messages = self._build_messages(
            instruction, processed_image, system_prompt, previous_actions_str
        )

        response = request_chat_completion(
            self._client,
            messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self.responses.append(response)

        instruction_summary, pyautogui_code = self.parse_response(
            response,
            width,
            height,
            processed_width,
            processed_height,
        )
        self.actions.append(instruction_summary)

        return response, pyautogui_code
