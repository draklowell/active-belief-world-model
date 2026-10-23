import base64
import json
import logging
from io import BytesIO
from ssl import SSLError

import backoff
from decouple import config
from openai import APIConnectionError, BadRequestError, InternalServerError, OpenAI, RateLimitError
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class StableDiffusionImageModel:
    """stable-diffusion.cpp image-editing client using its OpenAI-compatible endpoint.

    Args:
        model: Model identifier exposed by the image-editing server.
        max_size: Optional longest-side cap; omit to retain the source resolution.
        padding_multiple: Positive alignment divisor, defaulting to no padding.
        steps: Optional sampling-step override; omit to use the server's default.
        cfg: Optional guidance override applied to both text and image CFG.
        seed: Optional sampling seed; omit to use the server's default.
        base_url: Image endpoint URL, or IMAGE_BASE_URL from decouple when omitted.
        api_key_env: Environment variable containing the image endpoint API key.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        max_size: int | None = None,
        padding_multiple: int = 1,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        base_url: str | None = None,
        api_key_env: str = "IMAGE_API_KEY",
        timeout: float = 300.0,
    ) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError("Image max_size must be positive when configured.")
        if padding_multiple < 1:
            raise ValueError("Image padding_multiple must be positive.")
        self.model = model
        self.max_size = max_size
        self.padding_multiple = padding_multiple
        self.steps = steps
        self.cfg = cfg
        self.seed = seed
        self.client = OpenAI(
            base_url=base_url if base_url is not None else config("IMAGE_BASE_URL"),
            api_key=config(api_key_env),
            timeout=timeout,
        )

    @backoff.on_exception(
        backoff.constant,
        (SSLError, RateLimitError, BadRequestError, InternalServerError, APIConnectionError),
        interval=30,
        max_tries=5,
    )
    def generate(self, screenshot: bytes, prompt: str) -> bytes:
        """Render a transition and return PNG bytes, removing only alignment padding.

        Apply configured resizing and padding, then submit the prepared image and
        prompt to the remote API. Validate the returned size before removing only
        the added padding.

        Args:
            screenshot: Encoded image of the current GUI state.
            prompt: Image-editing instruction supplied by the caller.

        Returns:
            PNG-encoded prediction at the source size or explicitly configured downscaled size.
        """
        with Image.open(BytesIO(screenshot)) as source:
            image = source.convert("RGB")
        if self.max_size is not None and max(image.size) > self.max_size:
            original_size = image.size
            image.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
            logger.warning("Downscaling image input from %s to %s", original_size, image.size)
        width, height = image.size
        alignment = self.padding_multiple
        padded_width = (width + alignment - 1) // alignment * alignment
        padded_height = (height + alignment - 1) // alignment * alignment
        image = ImageOps.expand(
            image, border=(0, 0, padded_width - width, padded_height - height), fill="white"
        )
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        sample_params: dict[str, object] = {}
        if self.steps is not None:
            sample_params["sample_steps"] = self.steps
        if self.cfg is not None:
            sample_params["guidance"] = {"txt_cfg": self.cfg, "img_cfg": self.cfg}
        extra_params: dict[str, object] = {}
        if sample_params:
            extra_params["sample_params"] = sample_params
        if self.seed is not None:
            extra_params["seed"] = self.seed
        if extra_params:
            prompt += f"<sd_cpp_extra_args>{json.dumps(extra_params)}</sd_cpp_extra_args>"
        logger.info("Requesting image edit from %s", self.model)
        response = self.client.images.edit(
            model=self.model,
            image=("screenshot.png", image_buffer.getvalue(), "image/png"),
            prompt=prompt,
            size=f"{padded_width}x{padded_height}",
            n=1,
            response_format="b64_json",
            output_format="png",
        )
        if not response.data or not response.data[0].b64_json:
            raise ValueError(
                "The image inference endpoint returned no usable image; "
                "expected inline data[0].b64_json."
            )
        image_bytes = base64.b64decode(response.data[0].b64_json, validate=True)
        buffer = BytesIO()
        with Image.open(BytesIO(image_bytes)) as prediction:
            if prediction.size != (padded_width, padded_height):
                raise ValueError(
                    f"The image endpoint returned size {prediction.size}; "
                    f"expected {(padded_width, padded_height)}."
                )
            prediction.crop((0, 0, width, height)).save(buffer, format="PNG")
        return buffer.getvalue()
