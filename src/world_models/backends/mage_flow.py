import base64
import logging
from io import BytesIO
from ssl import SSLError

import backoff
from decouple import config
from openai import APIConnectionError, BadRequestError, InternalServerError, OpenAI, RateLimitError
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class MageFlowImageModel:
    """Remote Mage client using an OpenAI-compatible image-editing endpoint.

    Args:
        model: Model identifier exposed by the image-editing server.
        steps: Number of denoising steps, normally four for Turbo or thirty for Base.
        cfg: Classifier-free guidance scale, normally one for Turbo or five for Base.
        max_size: Optional longest-side cap; omit to retain the source resolution.
        seed: Reproducible generation seed.
        base_url: Image endpoint URL, or IMAGE_BASE_URL from decouple when omitted.
        api_key_env: Environment variable containing the image endpoint API key.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "microsoft/Mage-Flow-Edit-Turbo",
        steps: int = 4,
        cfg: float = 1.0,
        max_size: int | None = None,
        seed: int = 2026,
        base_url: str | None = None,
        api_key_env: str = "IMAGE_API_KEY",
        timeout: float = 300.0,
    ) -> None:
        if steps < 1:
            raise ValueError("Mage steps must be positive.")
        if max_size is not None and max_size < 16:
            raise ValueError("Mage max_size must be at least 16 pixels when configured.")
        if seed < 0:
            raise ValueError("Mage seed must be nonnegative for reproducible inference.")
        self.model = model
        self.steps = steps
        self.cfg = cfg
        self.max_size = max_size
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

        Cap the image size when configured, pad it to Mage's alignment, and submit
        the prepared image and prompt to the remote API. Validate the returned size
        before removing only the added padding.

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
            logger.warning("Downscaling Mage input from %s to %s", original_size, image.size)
        width, height = image.size
        padded_width = (width + 15) // 16 * 16
        padded_height = (height + 15) // 16 * 16
        image = ImageOps.expand(
            image, border=(0, 0, padded_width - width, padded_height - height), fill="white"
        )
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        logger.info("Requesting image edit from %s", self.model)
        response = self.client.images.edit(
            model=self.model,
            image=("screenshot.png", image_buffer.getvalue(), "image/png"),
            prompt=prompt,
            size=f"{padded_width}x{padded_height}",
            n=1,
            response_format="b64_json",
            output_format="png",
            extra_body={
                "num_inference_steps": self.steps,
                "guidance_scale": self.cfg,
                "seed": self.seed,
            },
        )
        if not response.data or not response.data[0].b64_json:
            raise ValueError("The image inference endpoint returned no usable image.")
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
