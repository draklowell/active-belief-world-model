import base64
import json
import math
from io import BytesIO
from ssl import SSLError
from typing import cast

import backoff
import openai
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from prompts import render


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


def prepare_screenshot(screenshot: bytes) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Encode a screenshot for Qwen3-VL and measure it before and after resizing.

    Args:
        screenshot: Raw screenshot bytes from the environment.

    Returns:
        Base64-encoded processed PNG, original width and height, and processed width and height.
    """
    with Image.open(BytesIO(screenshot)) as image:
        original_size = image.size
    processed_image = process_image(screenshot)
    with Image.open(BytesIO(base64.b64decode(processed_image))) as image:
        processed_size = image.size
    return processed_image, original_size, processed_size


def build_image_content(image_b64: str) -> dict:
    """Wrap a base64 PNG as an OpenAI chat image content part.

    Args:
        image_b64: Base64-encoded PNG image.

    Returns:
        Chat content part referencing the image as a data URL.
    """
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}


def build_computer_use_tool(
    coordinate_type: str, processed_width: int, processed_height: int
) -> dict:
    """Build the Qwen3-VL `computer_use` function signature that defines the action space.

    Args:
        coordinate_type: Either absolute processed-image pixels or relative 1000x1000 coordinates.
        processed_width: Width of the screenshot shown to the model.
        processed_height: Height of the screenshot shown to the model.

    Returns:
        Tool definition with the environment description, action names, and their arguments.
    """
    description_prompt = render(
        "qwen3vl/description.j2",
        coordinate_type=coordinate_type,
        processed_width=processed_width,
        processed_height=processed_height,
    )
    return json.loads(
        render(
            "qwen3vl/tools.json.j2",
            description_prompt=description_prompt,
            action_descriptions=render("qwen3vl/action_descriptions.j2"),
        )
    )


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
def request_chat_completion(
    client: openai.OpenAI,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Call an OpenAI-compatible endpoint with chat completion messages.

    Args:
        client: OpenAI-compatible client connected to the served Qwen3-VL model.
        messages: List of message dictionaries following the OpenAI chat schema.
        model: Served model identifier.
        max_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability mass.

    Returns:
        The completion text content from the model.
    """
    response = client.chat.completions.create(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    content = response.choices[0].message.content
    return content or ""


def adjust_coordinates(
    x: float,
    y: float,
    coordinate_type: str,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> tuple[int, int]:
    """Map model coordinates to native screen pixels.

    Args:
        x: Horizontal coordinate predicted by the model.
        y: Vertical coordinate predicted by the model.
        coordinate_type: Either absolute processed-image pixels or relative 1000x1000 coordinates.
        original_size: Native screen width and height in pixels.
        processed_size: Width and height of the resized image fed to the model.

    Returns:
        Native screen x and y pixel coordinates.
    """
    original_width, original_height = original_size
    processed_width, processed_height = processed_size
    if coordinate_type == "absolute":
        x_scale = original_width / processed_width
        y_scale = original_height / processed_height
        return int(x * x_scale), int(y * y_scale)
    x_scale = original_width / 999
    y_scale = original_height / 999
    return int(x * x_scale), int(y * y_scale)


def convert_tool_call_to_pyautogui(
    arguments: dict,
    coordinate_type: str,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> list[str]:
    """Convert `computer_use` tool call arguments into PyAutoGUI commands.

    Args:
        arguments: Tool call arguments, including the required `action` field.
        coordinate_type: Either absolute processed-image pixels or relative 1000x1000 coordinates.
        original_size: Native screen width and height in pixels.
        processed_size: Width and height of the resized image fed to the model.

    Returns:
        Ordered PyAutoGUI commands, or WAIT and DONE special actions.
    """
    action = arguments.get("action")
    coordinate = arguments.get("coordinate")

    if action in ("left_click", "right_click", "middle_click", "double_click"):
        command_map = {
            "left_click": "click",
            "right_click": "rightClick",
            "middle_click": "middleClick",
            "double_click": "doubleClick",
        }
        command = command_map[action]
        if coordinate:
            adj_x, adj_y = adjust_coordinates(
                coordinate[0], coordinate[1], coordinate_type, original_size, processed_size
            )
            return [f"pyautogui.{command}({adj_x}, {adj_y})"]
        return [f"pyautogui.{command}()"]

    if action == "type":
        text = arguments.get("text", "")
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
        for key in arguments.get("keys", []):
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
        pixels = arguments.get("pixels", 0)
        return [f"pyautogui.scroll({pixels})"]

    if action == "wait":
        return ["WAIT"]

    if action == "terminate":
        return ["DONE"]

    if action == "mouse_move":
        if coordinate:
            adj_x, adj_y = adjust_coordinates(
                coordinate[0], coordinate[1], coordinate_type, original_size, processed_size
            )
            return [f"pyautogui.moveTo({adj_x}, {adj_y})"]
        return ["pyautogui.moveTo(0, 0)"]

    if action == "left_click_drag":
        duration = arguments.get("duration", 0.5)
        if coordinate:
            adj_x, adj_y = adjust_coordinates(
                coordinate[0], coordinate[1], coordinate_type, original_size, processed_size
            )
            return [f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})"]
        return [f"pyautogui.dragTo(0, 0, duration={duration})"]

    return []
