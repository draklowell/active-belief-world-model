import base64
import logging
from io import BytesIO
from ssl import SSLError

import backoff
from decouple import config
from openai import APIConnectionError, BadRequestError, InternalServerError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from PIL import Image

logger = logging.getLogger(__name__)


def encode_png_base64(screenshot: bytes) -> str:
    """Re-encode an image as PNG and return it as base64 text for a data URL.

    Args:
        screenshot: Encoded image in any format Pillow reads.

    Returns:
        Base64 text of the PNG encoding.
    """
    with Image.open(BytesIO(screenshot)) as image:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAITextModel:
    """OpenAI-compatible VLM client for caller-provided multimodal prompts.

    Args:
        model: Served VLM identifier, including aliases or fine-tuned checkpoints.
        max_tokens: Maximum number of generated text tokens.
        temperature: Text sampling temperature.
        base_url: Endpoint URL, or OPENAI_BASE_URL from decouple when omitted.
        api_key_env: Environment variable containing the API key.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url if base_url is not None else config("OPENAI_BASE_URL"),
            api_key=config(api_key_env),
            timeout=timeout,
        )

    def generate(self, screenshot: bytes, prompt: str, *, system_prompt: str) -> str:
        """Request text using the supplied prompts and full-resolution screenshot.

        Args:
            screenshot: Encoded image of the current GUI state.
            prompt: User instruction supplied by the caller.
            system_prompt: Domain instructions supplied by the caller.

        Returns:
            Nonempty generated text.
        """
        return self.generate_from_screenshots([screenshot], prompt, system_prompt=system_prompt)

    @backoff.on_exception(
        backoff.constant,
        (SSLError, RateLimitError, BadRequestError, InternalServerError, APIConnectionError),
        interval=30,
        max_tries=5,
    )
    def generate_from_screenshots(
        self, screenshots: list[bytes], prompt: str, *, system_prompt: str
    ) -> str:
        """Request text using the supplied prompts and full-resolution screenshots.

        The screenshots are attached in the given order before the prompt, so the prompt can refer
        to them as the first, second, and later images.

        Args:
            screenshots: Encoded images, such as the current and the next GUI state.
            prompt: User instruction supplied by the caller.
            system_prompt: Domain instructions supplied by the caller.

        Returns:
            Nonempty generated text.
        """
        content: list[ChatCompletionContentPartParam] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_png_base64(screenshot)}"},
            }
            for screenshot in screenshots
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        logger.info("Requesting text with %d screenshots from %s", len(screenshots), self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if not response.choices:
            raise ValueError("The textual inference endpoint returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("The text response was truncated; increase max_tokens.")
        text = choice.message.content
        if text is None or not text.strip():
            raise ValueError("The textual inference endpoint returned no usable text.")
        return text.strip()
