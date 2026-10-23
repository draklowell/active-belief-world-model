import base64
import json
import logging
import math
from io import BytesIO
from ssl import SSLError
from typing import cast

import backoff
import openai
from bs4 import BeautifulSoup
from decouple import config
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from agents.base import Agent

logger = logging.getLogger(__name__)


def round_by_factor(number: float, factor: int) -> int:
    """Round a number to the nearest multiple of a factor.

    Args:
        number: Value to round.
        factor: Multiple divisor.

    Returns:
        Nearest integer multiple of factor.
    """
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Round a number up to the nearest multiple of a factor.

    Args:
        number: Value to round up.
        factor: Multiple divisor.

    Returns:
        Smallest integer multiple of factor greater than or equal to number.
    """
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Round a number down to the nearest multiple of a factor.

    Args:
        number: Value to round down.
        factor: Multiple divisor.

    Returns:
        Largest integer multiple of factor less than or equal to number.
    """
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 16 * 16 * 4 * 12800,
    max_long_side: int = 8192,
) -> tuple[int, int]:
    """Calculate target image dimensions respecting factor multiples and token budgets.

    Args:
        height: Original image height in pixels.
        width: Original image width in pixels.
        factor: Multiple requirement for resized dimensions.
        min_pixels: Lower bound on total pixel count.
        max_pixels: Upper bound on total pixel count.
        max_long_side: Maximum allowable dimension for the longer side.

    Returns:
        Tuple of resized height and width.
    """
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(float(height), factor)
    w_bar = round_by_factor(float(width), factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def process_image(image_bytes: bytes) -> str:
    """Resize image to model constraints and encode as base64 PNG.

    Args:
        image_bytes: Raw screenshot bytes.

    Returns:
        Base64-encoded PNG string.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    resized_image = image.resize((resized_width, resized_height))
    buffer = BytesIO()
    resized_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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

    def _build_description_prompt(self, processed_width: int, processed_height: int) -> str:
        resolution = (
            f"* The screen's resolution is {processed_width}x{processed_height}."
            if self.coordinate_type == "absolute"
            else "* The screen's resolution is 1000x1000."
        )
        lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            (
                "* This is an interface to a desktop GUI. You do not have access to a terminal "
                "or applications menu. You must click on desktop icons to start applications."
            ),
            (
                "* Some applications may take time to start or process actions, so you may need "
                "to wait and take successive screenshots to see the results of your actions. "
                "E.g. if you click on Firefox and a window doesn't open, try wait and taking "
                "another screenshot."
            ),
            resolution,
            (
                "* Whenever you intend to move the cursor to click on an element like an icon, "
                "you should consult a screenshot to determine the coordinates of the element "
                "before moving the cursor."
            ),
            (
                "* If you tried clicking on a program or link but it failed to load even after "
                "waiting, try adjusting your cursor position so that the tip of the cursor "
                "visually falls on the element that you want to click."
            ),
            (
                "* Make sure to click any buttons, links, icons, etc with the cursor tip in "
                "the center of the element. Don't click boxes on their edges unless asked."
            ),
        ]
        return "\n".join(lines)

    def _build_system_prompt(self, description_prompt: str) -> str:
        action_descriptions = (
            "* `key`: Performs key down presses on the arguments passed in order, "
            "then performs key releases in reverse order.\n"
            "* `type`: Type a string of text on the keyboard.\n"
            "* `mouse_move`: Move cursor to a specified (x, y) pixel coordinate.\n"
            "* `left_click`: Click the left mouse button at (x, y).\n"
            "* `left_click_drag`: Click and drag cursor to (x, y).\n"
            "* `right_click`: Click the right mouse button at (x, y).\n"
            "* `middle_click`: Click the middle mouse button at (x, y).\n"
            "* `double_click`: Double-click left mouse button at (x, y).\n"
            "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
            "* `wait`: Wait specified seconds for the change to happen.\n"
            "* `terminate`: Terminate the current task and report completion status.\n"
        )
        tools_def = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_descriptions,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "scroll",
                                "wait",
                                "terminate",
                            ],
                            "type": "string",
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {
                            "description": "Required only by `action=type`.",
                            "type": "string",
                        },
                        "coordinate": {
                            "description": "The x,y coordinates for mouse actions.",
                            "type": "array",
                        },
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "The seconds to wait.", "type": "number"},
                        "status": {
                            "description": "The status of the task.",
                            "type": "string",
                            "enum": ["success", "failure"],
                        },
                    },
                    "required": ["action"],
                    "type": "object",
                },
                "args_format": "Format the arguments as a JSON object.",
            },
        }
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{json.dumps(tools_def)}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block containing only the JSON: "
            '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

    def _build_messages(
        self,
        instruction: str,
        processed_image: str,
        system_prompt: str,
        previous_actions_str: str,
    ) -> list[dict]:
        instruction_prompt = (
            "Please generate the next move according to the UI screenshot, "
            f"instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n{previous_actions_str}"
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
            content: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            ]
            if index == 0:
                content.append({"type": "text", "text": instruction_prompt})
            messages.append({"role": "user", "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(history_responses[index])}],
                }
            )

        curr_content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{processed_image}"},
            }
        ]
        if history_len == 0:
            curr_content.append({"type": "text", "text": instruction_prompt})
        messages.append({"role": "user", "content": curr_content})

        return messages

    @backoff.on_exception(
        backoff.constant,
        (
            SSLError,
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ),
        interval=30,
        max_tries=5,
    )
    def call_llm(self, messages: list[dict]) -> str:
        """Call the OpenAI-compatible endpoint with chat completion messages.

        Args:
            messages: List of message dictionaries following the OpenAI chat schema.

        Returns:
            The completion text content from the model.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = response.choices[0].message.content
        return content or ""

    def _adjust_coordinates(
        self,
        x: float,
        y: float,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> tuple[int, int]:
        if self.coordinate_type == "absolute":
            x_scale = original_width / processed_width
            y_scale = original_height / processed_height
            return int(x * x_scale), int(y * y_scale)
        x_scale = original_width / 999
        y_scale = original_height / 999
        return int(x * x_scale), int(y * y_scale)

    def _parse_tool_call_action(
        self,
        args: dict,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> list[str]:
        action = args.get("action")
        coordinate = args.get("coordinate")

        if action in ("left_click", "right_click", "middle_click", "double_click"):
            command_map = {
                "left_click": "click",
                "right_click": "rightClick",
                "middle_click": "middleClick",
                "double_click": "doubleClick",
            }
            command = command_map[action]
            if coordinate:
                adj_x, adj_y = self._adjust_coordinates(
                    coordinate[0],
                    coordinate[1],
                    original_width,
                    original_height,
                    processed_width,
                    processed_height,
                )
                return [f"pyautogui.{command}({adj_x}, {adj_y})"]
            return [f"pyautogui.{command}()"]

        if action == "type":
            text = args.get("text", "")
            lines = text.split("\n")
            actions = []
            for index, line in enumerate(lines):
                if line:
                    actions.append(f"pyautogui.typewrite({repr(line)}, interval=0.03)")
                if index < len(lines) - 1:
                    actions.append("pyautogui.press('enter')")
            return actions

        if action == "key":
            keys = []
            for key in args.get("keys", []):
                key = str(key).strip()
                if key:
                    keys.append(key)

            keys_str = ", ".join(repr(k) for k in keys)
            if len(keys) > 1:
                return [f"pyautogui.hotkey({keys_str})"]
            if keys:
                return [f"pyautogui.press({keys_str})"]
            return []

        if action == "scroll":
            pixels = args.get("pixels", 0)
            return [f"pyautogui.scroll({pixels})"]

        if action == "wait":
            return ["WAIT"]

        if action == "terminate":
            return ["DONE"]

        if action == "mouse_move":
            if coordinate:
                adj_x, adj_y = self._adjust_coordinates(
                    coordinate[0],
                    coordinate[1],
                    original_width,
                    original_height,
                    processed_width,
                    processed_height,
                )
                return [f"pyautogui.moveTo({adj_x}, {adj_y})"]
            return ["pyautogui.moveTo(0, 0)"]

        if action == "left_click_drag":
            duration = args.get("duration", 0.5)
            if coordinate:
                adj_x, adj_y = self._adjust_coordinates(
                    coordinate[0],
                    coordinate[1],
                    original_width,
                    original_height,
                    processed_width,
                    processed_height,
                )
                return [f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})"]
            return [f"pyautogui.dragTo(0, 0, duration={duration})"]

        return []

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
                    actions = self._parse_tool_call_action(
                        tool_call.get("arguments", {}),
                        original_width,
                        original_height,
                        processed_width,
                        processed_height,
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
        screenshot_bytes = observation["screenshot"]
        image = Image.open(BytesIO(screenshot_bytes))
        width, height = image.size

        processed_image = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image)))
        processed_width, processed_height = processed_img.size

        self.screenshots.append(processed_image)

        current_step = len(self.actions)
        history_start_index = max(0, current_step - self.history_n)
        previous_actions = [
            f"Step {i + 1}: {self.actions[i]}" for i in range(history_start_index, current_step)
        ]
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        description_prompt = self._build_description_prompt(processed_width, processed_height)
        system_prompt = self._build_system_prompt(description_prompt)
        messages = self._build_messages(
            instruction, processed_image, system_prompt, previous_actions_str
        )

        response = self.call_llm(messages)
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
