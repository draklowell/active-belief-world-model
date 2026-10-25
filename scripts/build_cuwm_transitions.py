#!/usr/bin/env python
"""Build CUWM transitions from the AgentNet ubuntu subset and sample them for annotation.

Reads the pinned ubuntu JSONL and the screenshot archive index from data/raw/agentnet/, which are
written by download_agentnet.py and download_agentnet_images.py. Writes to the configured output
directory: transitions.parquet with every step and its exclusion reason, sample.parquet with the
split and domain stratified sample to annotate, osworld_overlap.csv with the excluded tasks, and
build_info.json with the settings and statistics of the build. No screenshot is downloaded.
"""

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

import polars as pl

from agentnet import DATA_DIR, IMAGE_INDEX_FILE, SUBSET_FILES, pinned_commit, scan_jsonl
from cuwm_dataset import (
    allocate_quota,
    assign_splits,
    build_transitions,
    find_osworld_overlap,
    load_osworld_instructions,
    sample_transitions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def load_settings(path: Path) -> dict:
    """Load the data preparation settings and check that splits and sample sizes agree.

    Args:
        path: Path to the TOML configuration.

    Returns:
        Settings with seed, osworld_tasks, overlap_threshold, output_dir, and the
        split_fractions, sample_sizes and domain_weights tables.
    """
    with path.open("rb") as config_file:
        settings = tomllib.load(config_file)
    if list(settings["split_fractions"]) != list(settings["sample_sizes"]):
        raise ValueError(
            "The [split_fractions] and [sample_sizes] tables must name the same splits in order."
        )
    if abs(sum(settings["split_fractions"].values()) - 1) > 1e-9:
        raise ValueError("The [split_fractions] values must sum to one.")
    return settings


def count_by(frame: pl.DataFrame, *columns: str) -> list[dict]:
    """Count rows per combination of column values.

    Args:
        frame: Frame to count.
        columns: Columns to group by.

    Returns:
        One record per combination with its values and row count, sorted by the columns.
    """
    return frame.group_by(columns).agg(pl.len().alias("rows")).sort(columns).to_dicts()


def main() -> None:
    """Build the transition table, the OSWorld overlap report, the sample and the build summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/cuwm-data.toml"))
    args = parser.parse_args()
    settings = load_settings(args.config)
    seed = settings["seed"]
    output_dir = Path(settings["output_dir"])

    trajectories = scan_jsonl(DATA_DIR / SUBSET_FILES["ubuntu"])
    tasks = trajectories.select(
        "task_id", "domain", "instruction", "natural_language_task", "actual_task"
    ).collect()
    osworld_tasks = load_osworld_instructions(Path(settings["osworld_tasks"]))
    overlap = find_osworld_overlap(tasks, osworld_tasks, settings["overlap_threshold"])
    logger.info(
        "Found %d AgentNet tasks overlapping %d OSWorld tasks, %d of them exact copies",
        overlap.height,
        overlap["osworld_id"].n_unique(),
        overlap["is_exact"].sum(),
    )

    image_index = pl.read_csv(IMAGE_INDEX_FILE)
    transitions = build_transitions(trajectories, image_index, overlap["task_id"]).join(
        assign_splits(tasks, settings["split_fractions"], seed), on="task_id"
    )
    logger.info(
        "Built %d steps, %d eligible transitions",
        transitions.height,
        transitions["exclusion_reason"].null_count(),
    )

    quotas = pl.DataFrame(
        [
            {"split": split, "domain": domain, "quota": quota}
            for split, size in settings["sample_sizes"].items()
            for domain, quota in allocate_quota(size, settings["domain_weights"]).items()
        ]
    )
    sample = sample_transitions(transitions, quotas, seed)
    sampled_images = (
        sample.unpivot(["image", "next_image"], value_name="name")
        .select("name")
        .unique()
        .join(image_index, on="name")
    )
    coverage = (
        quotas.join(
            sample.group_by("split", "domain").agg(pl.len().alias("sampled")),
            on=["split", "domain"],
            how="left",
        )
        .with_columns(pl.col("sampled").fill_null(0))
        .sort("split", "domain")
    )
    shortfall = coverage.filter(pl.col("sampled") < pl.col("quota"))
    if shortfall.height:
        logger.warning("Quotas not met for %d split and domain pairs", shortfall.height)

    output_dir.mkdir(parents=True, exist_ok=True)
    transitions.write_parquet(output_dir / "transitions.parquet")
    sample.write_parquet(output_dir / "sample.parquet")
    overlap.join(
        osworld_tasks.select(pl.col("task_id").alias("osworld_id"), "domain"), on="osworld_id"
    ).rename({"domain": "osworld_domain"}).write_csv(output_dir / "osworld_overlap.csv")
    eligible = transitions.filter(pl.col("exclusion_reason").is_null())
    build_info = {
        "revision": pinned_commit(),
        "config": str(args.config),
        "settings": settings,
        "steps": transitions.height,
        "eligible_transitions": eligible.height,
        "exclusions": count_by(transitions.drop_nulls("exclusion_reason"), "exclusion_reason"),
        "osworld_overlap": {
            "tasks": overlap.height,
            "exact_tasks": int(overlap["is_exact"].sum()),
            "osworld_tasks": overlap["osworld_id"].n_unique(),
            "steps": transitions.filter(pl.col("exclusion_reason") == "osworld_overlap").height,
        },
        "eligible_by_split": count_by(eligible, "split"),
        "sample": {
            "transitions": sample.height,
            "tasks": sample["task_id"].n_unique(),
            "by_split_and_domain": coverage.to_dicts(),
            "by_action": count_by(sample, "split", "action_name"),
            "by_judge_status": count_by(sample, "split", "judge_status"),
            "unique_images": sampled_images.height,
            "compressed_megabytes": round(
                sampled_images.select(pl.col("compressed_bytes").sum() / 1e6).item(), 1
            ),
        },
    }
    (output_dir / "build_info.json").write_text(json.dumps(build_info, indent=2) + "\n")
    logger.info(
        "Sampled %d transitions from %d tasks with %d screenshots (%.1f MB compressed) into %s",
        sample.height,
        build_info["sample"]["tasks"],
        sampled_images.height,
        build_info["sample"]["compressed_megabytes"],
        output_dir,
    )


if __name__ == "__main__":
    main()
