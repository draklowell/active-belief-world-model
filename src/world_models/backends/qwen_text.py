import base64
import logging
from io import BytesIO

from decouple import config
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

logger = logging.getLogger(__name__)


class QwenTextModel:
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
        model: str = "qwen3-vl",
        max_tokens: int = 512,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url if base_url is not None else config("OPENAI_BASE_URL"),
            api_key=config(api_key_env),
            timeout=timeout,
            max_retries=0,
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
        with Image.open(BytesIO(screenshot)) as image:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
        logger.info("Requesting text from %s", self.model)
        response = self._client.chat.completions.create(
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
