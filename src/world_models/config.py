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
    """Compose the configured textual and visual adapters without loading Mage weights.

    Args:
        settings: Mapping of backend names and constructor options from load_cuwm_config.

    Returns:
        CUWM with an OpenAI-compatible textual adapter and a lazy Mage visual adapter.
    """
    from world_models.backends.mage_flow import MageFlowImageModel
    from world_models.backends.qwen_text import QwenTextModel

    text_settings = settings["text_model"].copy()
    image_settings = settings["image_model"].copy()
    text_backend = text_settings.pop("backend", None)
    image_backend = image_settings.pop("backend", None)
    if text_backend != "openai_compatible_vlm":
        raise ValueError(f"Unsupported CUWM textual backend: {text_backend!r}.")
    if image_backend != "mage_flow":
        raise ValueError(f"Unsupported CUWM visual backend: {image_backend!r}.")
    return CUWM(
        text_model=QwenTextModel(**text_settings),
        image_model=MageFlowImageModel(**image_settings),
    )
