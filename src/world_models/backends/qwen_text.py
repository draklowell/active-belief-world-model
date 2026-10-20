import base64
import logging
from io import BytesIO

from decouple import config
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

logger = logging.getLogger(__name__)

TRANSITION_PROMPT = (
    "You predict immediate GUI state transitions for a computer-using world model. "
    "Given the current screenshot and one hypothetical candidate action, describe the "
    "visible changes that would most likely follow that action. Identify the affected "
    "UI elements and their locations, including selection, focus, content, menus, or "
    "dialogs when relevant. Use the screenshot as the baseline and mention relevant "
    "unchanged elements concisely. If no visible change is expected, say so. "
    "Do not execute the action or claim it has happened. Do not choose another action, "
    "write a task plan, or invent hidden state or details that cannot reasonably be inferred. "
    "Return only a concise natural-language transition description for an image editor."
)


class QwenTextModel:
    """OpenAI-compatible VLM adapter for predicting GUI action consequences.

    Args:
        model: Served VLM identifier, including aliases or fine-tuned checkpoints.
        max_tokens: Maximum number of generated transition tokens.
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

    def generate(self, screenshot: bytes, action: str) -> str:
        """Predict immediate UI changes using the full-resolution screenshot.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Candidate action to interpret without parsing or executing it.

        Returns:
            Nonempty natural-language description of the expected transition.
        """
        with Image.open(BytesIO(screenshot)) as image:
            width, height = image.size
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": TRANSITION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Current screenshot: {width} x {height} pixels.\n"
                            f"Hypothetical candidate action:\n{action}\n\n"
                            "Predict only the immediate resulting UI transition."
                        ),
                    },
                ],
            },
        ]
        logger.info("Predicting textual transition with %s", self.model)
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
            raise ValueError("The transition description was truncated; increase max_tokens.")
        description = choice.message.content
        if description is None or not description.strip():
            raise ValueError("The textual inference endpoint returned no usable description.")
        return description.strip()
