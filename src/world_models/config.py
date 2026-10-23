import tomllib
from pathlib import Path
from typing import Any

from world_models.cuwm import CUWM


def load_cuwm_config(path: str | Path) -> dict[str, Any]:
    """Load the dedicated CUWM backend configuration from TOML.

    Args:
        path: Path to the CUWM configuration file.

    Returns:
        Mapping with text_model and image_model sections containing backend names and options.
    """
    with Path(path).open("rb") as config_file:
        settings = tomllib.load(config_file)
    for section in ("text_model", "image_model"):
        if not isinstance(settings.get(section), dict):
            raise ValueError(f"CUWM configuration requires a {section} table.")
    return settings


def build_cuwm(settings: dict[str, Any]) -> CUWM:
    """Compose the configured textual and visual API clients.

    Args:
        settings: Mapping of backend names and constructor options from load_cuwm_config.

    Returns:
        CUWM with separate OpenAI-compatible text and stable-diffusion.cpp image clients.
    """
    from world_models.backends.openai_text import OpenAITextModel
    from world_models.backends.stable_diffusion_image import StableDiffusionImageModel

    text_settings = settings["text_model"].copy()
    image_settings = settings["image_model"].copy()
    text_backend = text_settings.pop("backend", None)
    image_backend = image_settings.pop("backend", None)
    if text_backend != "openai_compatible_text":
        raise ValueError(f"Unsupported CUWM textual backend: {text_backend!r}.")
    if image_backend != "stable_diffusion_image":
        raise ValueError(f"Unsupported CUWM visual backend: {image_backend!r}.")
    return CUWM(
        text_model=OpenAITextModel(**text_settings),
        image_model=StableDiffusionImageModel(**image_settings),
    )
