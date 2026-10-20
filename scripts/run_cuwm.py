import argparse
import logging
import sys
from pathlib import Path

from world_models.config import build_cuwm, load_cuwm_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Predict one hypothetical GUI transition and save the textual and visual outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Predict a one-step GUI state transition.")
    parser.add_argument("--screenshot", type=Path, required=True, help="Existing screenshot image.")
    parser.add_argument("--action", required=True, help="Hypothetical candidate action text.")
    parser.add_argument("--config", type=Path, default=Path("config/cuwm.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true", help="Run only the textual stage.")
    args = parser.parse_args()

    settings = load_cuwm_config(args.config)
    screenshot = args.screenshot.read_bytes()
    cuwm = build_cuwm(settings)
    text_description = cuwm.generate_text_desc(screenshot, args.action)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "transition.txt"
    transition_path.write_text(text_description + "\n", encoding="utf-8")
    logger.info("Saved textual transition to %s", transition_path)

    if not args.text_only:
        predicted_screenshot = cuwm.generate_image(screenshot, text_description)
        image_path = args.output_dir / "predicted_next_state.png"
        image_path.write_bytes(predicted_screenshot)
        logger.info("Saved predicted screenshot to %s", image_path)


if __name__ == "__main__":
    main()
