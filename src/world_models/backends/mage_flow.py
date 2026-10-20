import logging
from importlib import import_module
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class MageFlowImageModel:
    """Lazy adapter for the official MageFlowPipeline image-editing API.

    Args:
        model_id: Official editing checkpoint ID or local checkpoint directory.
        device: Device passed to the official pipeline loader.
        steps: Number of denoising steps, normally four for Turbo or thirty for Base.
        cfg: Classifier-free guidance scale, normally one for Turbo or five for Base.
        max_size: Optional longest-side cap; omit to retain the source resolution.
        seed: Reproducible generation seed.
        vl_cond_long_edge: Longest-side cap for Mage's internal VLM conditioning image.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Mage-Flow-Edit-Turbo",
        device: str = "cuda",
        steps: int = 4,
        cfg: float = 1.0,
        max_size: int | None = None,
        seed: int = 2026,
        vl_cond_long_edge: int = 384,
    ) -> None:
        if steps < 1:
            raise ValueError("Mage steps must be positive.")
        if max_size is not None and max_size < 16:
            raise ValueError("Mage max_size must be at least 16 pixels when configured.")
        if seed < 0:
            raise ValueError("Mage seed must be nonnegative for reproducible inference.")
        self.model_id = model_id
        self.device = device
        self.steps = steps
        self.cfg = cfg
        self.max_size = max_size
        self.seed = seed
        self.vl_cond_long_edge = vl_cond_long_edge
        self._pipeline = None

    def _load_pipeline(self) -> Any:
        try:
            mage_flow = import_module("mage_flow")
        except ModuleNotFoundError as error:
            if error.name != "mage_flow":
                raise
            raise RuntimeError(
                "The official mage-flow package is required for visual inference. "
                "Install it in a compatible CUDA environment as described in "
                "src/world_models/README.md, or use --text-only."
            ) from error
        logger.info("Loading Mage checkpoint %s on %s", self.model_id, self.device)
        return mage_flow.MageFlowPipeline.from_pretrained(self.model_id, device=self.device)

    def generate(self, screenshot: bytes, text_description: str) -> bytes:
        """Render a transition and return PNG bytes, removing only alignment padding.

        Args:
            screenshot: Encoded image of the current GUI state.
            text_description: Predicted immediate UI changes to render.

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
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        instruction = (
            f"{text_description}\n\n"
            "Preserve the layout, text, icons, colors, and all other GUI elements "
            "unrelated to this transition."
        )
        logger.info("Rendering visual transition with %s", self.model_id)
        images = self._pipeline.edit(
            [instruction],
            [image],
            heights=[padded_height],
            widths=[padded_width],
            steps=self.steps,
            cfg=self.cfg,
            seeds=[self.seed],
            vl_cond_long_edge=self.vl_cond_long_edge,
        )
        if not images or not isinstance(images[0], Image.Image):
            raise ValueError("Mage returned no usable image.")
        prediction = images[0]
        if prediction.size != (padded_width, padded_height):
            raise ValueError(
                f"Mage returned image size {prediction.size}; "
                f"expected {(padded_width, padded_height)}."
            )
        buffer = BytesIO()
        prediction.crop((0, 0, width, height)).save(buffer, format="PNG")
        return buffer.getvalue()
