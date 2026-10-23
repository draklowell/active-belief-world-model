# data/

Data card for the datasets used in this project. Raw data is not committed: `data/raw/*`,
`data/interim/*` and `data/processed/*` are gitignored, and only this file and the
`.gitkeep` placeholders are tracked.

## AgentNet

**AgentNet** (`xlangai/AgentNet` on Hugging Face), a dataset of computer-use trajectories
with annotations and screenshots. The full dataset is about 200 GB, almost all of it split
zip archives of screenshots, so this project fetches only parts of it.

- Hugging Face page: <https://huggingface.co/datasets/xlangai/AgentNet>
- License and citation: to be confirmed from the dataset card before redistributing
  anything derived from it.

## How to get it

```bash
uv run python scripts/download_agentnet.py
uv run python scripts/download_agentnet_images.py
```

`download_agentnet.py` fetches the JSONL annotation subsets into `data/raw/agentnet/` and
records the resolved dataset commit in `data/raw/agentnet/REVISION.txt`, so every subset comes
from one snapshot. Subsets and approximate sizes: `meta` 19 MB, `ubuntu` 282 MB, `win_mac`
1.4 GB.

`download_agentnet_images.py` fetches screenshots for a seeded, domain stratified sample of
ubuntu tasks using HTTP range requests against the pinned revision, verifies every image with
CRC32, and writes `data/raw/agentnet/ubuntu_images_manifest.csv` with sha256 digests.

## Layout

| Path                 | Contents                                                         |
| -------------------- | ---------------------------------------------------------------- |
| `data/raw/agentnet/` | Downloaded JSONL files, `REVISION.txt`, image index and manifest |
| `data/interim/`      | Intermediate forms of the raw data, currently empty              |
| `data/processed/`    | Model-ready data, currently empty                                |

Exploration of the data is in `notebooks/05_agentnet_eda.ipynb` and
`notebooks/06_agentnet_images_eda.ipynb`.
