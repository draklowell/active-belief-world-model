#!/usr/bin/env python
"""Download screenshots of a seeded, domain stratified sample of AgentNet ubuntu tasks.

Only the bytes of the selected images are fetched with HTTP range requests, using the
zip central directory of the pinned dataset revision (see data/raw/agentnet/REVISION.txt).
Every image is CRC32 verified. The sample is a deterministic function of the revision,
--tasks and --seed, and the resulting file list with sha256 digests is written to
data/raw/agentnet/ubuntu_images_manifest.csv. Rerunning skips images already on disk.
"""

import argparse
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import polars as pl

from agentnet import (
    DATA_DIR,
    IMAGE_INDEX_FILE,
    IMAGE_MANIFEST_FILE,
    IMAGES_DIR,
    SUBSET_FILES,
    fetch_entry,
    pinned_commit,
    read_zip_index,
    scan_jsonl,
    segment_url,
)

logger = logging.getLogger(__name__)


def sample_tasks(tasks: pl.DataFrame, n_tasks: int, seed: int) -> pl.DataFrame:
    fraction = n_tasks / tasks.height
    shuffled = tasks.sort("task_id").sample(fraction=1.0, shuffle=True, seed=seed)
    return (
        shuffled.with_columns(pl.int_range(pl.len()).over("domain").alias("rank"))
        .filter(pl.col("rank") < (pl.len().over("domain") * fraction).ceil())
        .drop("rank")
        .sort("task_id")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    commit = pinned_commit()
    with httpx.Client(
        follow_redirects=True, timeout=120, transport=httpx.HTTPTransport(retries=3)
    ) as client:
        if IMAGE_INDEX_FILE.exists():
            index = pl.read_csv(IMAGE_INDEX_FILE)
        else:
            last_disk = 13
            index = read_zip_index(client, segment_url(commit, last_disk, last_disk))
            index.write_csv(IMAGE_INDEX_FILE)
        last_disk = int(index.select(pl.col("disk").max()).item())

        steps = (
            scan_jsonl(DATA_DIR / SUBSET_FILES["ubuntu"])
            .select("task_id", "domain", "traj")
            .explode("traj")
            .select("task_id", "domain", pl.col("traj").struct.field("image").alias("name"))
            .collect()
        )
        tasks = sample_tasks(steps.select("task_id", "domain").unique(), args.tasks, args.seed)
        images = (
            steps.join(tasks.select("task_id"), on="task_id")
            .join(index, on="name")
            .sort("task_id", "name")
        )
        logger.info(
            "Selected %d tasks, %d steps, %d unique images, %.1f MB compressed",
            tasks.height,
            images.height,
            images["name"].n_unique(),
            images.unique("name").select(pl.col("compressed_bytes").sum() / 1e6).item(),
        )

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        def download(row: dict) -> str:
            path = IMAGES_DIR / row["name"]
            if not path.exists():
                url = segment_url(commit, row["disk"], last_disk)
                data = fetch_entry(
                    client, url, row["offset"], row["compressed_bytes"], row["crc32"]
                )
                path.write_bytes(data)
            return hashlib.sha256(path.read_bytes()).hexdigest()

        unique = images.unique("name", maintain_order=True)
        with ThreadPoolExecutor(args.workers) as pool:
            digests = list(pool.map(download, unique.iter_rows(named=True)))

    manifest = images.join(
        unique.select("name").with_columns(pl.Series("sha256", digests)), on="name"
    ).with_columns(
        pl.lit(commit).alias("revision"),
        pl.lit(args.seed).alias("seed"),
        pl.lit(args.tasks).alias("requested_tasks"),
    )
    manifest.write_csv(IMAGE_MANIFEST_FILE)
    logger.info(
        "Wrote %d images to %s and manifest %s", unique.height, IMAGES_DIR, IMAGE_MANIFEST_FILE
    )


if __name__ == "__main__":
    main()
