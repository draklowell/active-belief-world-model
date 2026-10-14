#!/usr/bin/env python
"""Download a text only subset of the AgentNet dataset from Hugging Face.

The full dataset is about 200 GB, almost all of it split zip archives of screenshots.
This script fetches only the JSONL files (annotations and trajectories) into
data/raw/agentnet/ and records the resolved dataset commit in REVISION.txt.

Subsets and approximate sizes: meta 19 MB, ubuntu 282 MB, win_mac 1.4 GB.
"""

import argparse
import logging

from huggingface_hub import HfApi, hf_hub_download

from agentnet import DATA_DIR, REPO_ID, REVISION_FILE, SUBSET_FILES

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets", nargs="+", choices=list(SUBSET_FILES), default=["meta", "ubuntu"]
    )
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit sha")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    commit = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        path = hf_hub_download(
            REPO_ID,
            SUBSET_FILES[subset],
            repo_type="dataset",
            revision=commit,
            local_dir=DATA_DIR,
        )
        logger.info("Downloaded %s", path)
    REVISION_FILE.write_text(f"{REPO_ID}@{commit}\n")
    logger.info("Recorded %s in %s", commit, REVISION_FILE)


if __name__ == "__main__":
    main()
