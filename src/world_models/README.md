# Computer-Using World Model

CUWM predicts one hypothetical GUI transition without executing the candidate action.
It reproduces the two-stage factorization in
[Guan et al., section 3.1 and Figure 2](https://arxiv.org/abs/2602.17365):

1. `TextTransitionModel.generate(screenshot, action)` predicts a transition description.
2. `ImageTransitionModel.generate(screenshot, text_description)` edits the original
   screenshot to realize that prediction.

Both inputs use encoded screenshot `bytes`. Actions and transition descriptions are
plain strings. Visual output is PNG-encoded `bytes`. `CUWM.predict` returns a
`CUWMPrediction` containing `text_description` and `predicted_screenshot`.

`base.py` owns the two protocols, `types.py` owns the prediction dataclass, and
`cuwm.py` composes injected models. Importing `world_models` uses only the standard
library. Model libraries, prompts, preprocessing, and inference settings stay in
`backends/`. There is no dependency on the agent hierarchy or OSWorld.

## Python API

```python
from pathlib import Path

from world_models.config import build_cuwm, load_cuwm_config

cuwm = build_cuwm(load_cuwm_config("config/cuwm.toml"))
screenshot = Path("data/example.png").read_bytes()
prediction = cuwm.predict(screenshot=screenshot, action="Select column H")
Path("predicted_next_state.png").write_bytes(prediction.predicted_screenshot)
```

Use `cuwm.generate_text_desc(screenshot, action)` or
`cuwm.generate_image(screenshot, text_description)` to run stages independently.
For direct dependency injection, construct `CUWM(text_model=..., image_model=...)`
with objects implementing the corresponding `generate` signatures.

## Configuration

`config/cuwm.toml` selects the two implemented adapters. It follows the shared
[configuration conventions](../../AGENTS.md#configuration-conventions): TOML for
runtime configuration and JSON for project and service metadata. Each model table
contains a `backend` identifier and that adapter's constructor options.

| Section       | Backend                 | Default model                    |
| ------------- | ----------------------- | -------------------------------- |
| `text_model`  | `openai_compatible_vlm` | `qwen3-vl`                       |
| `image_model` | `mage_flow`             | `microsoft/Mage-Flow-Edit-Turbo` |

The textual model uses the existing `OPENAI_BASE_URL` and `OPENAI_API_KEY`
environment variables through `python-decouple`, including the project's `.env`.
The default served name matches `config/server-qwen3vl.json`. Set `base_url` in the
text section to override the endpoint, or `api_key_env` to select a different
secret variable. Never put the key itself in TOML. `max_tokens`, `temperature`,
and `timeout` are adapter options. The dedicated prompt predicts consequences,
rather than choosing the next agent action. The adapter preserves screenshot
dimensions when constructing its multimodal request; the server may apply its
own model preprocessing. Empty and truncated responses fail explicitly.

To switch textual checkpoints, change `text_model.model` to the identifier
actually served by your OpenAI-compatible VLM endpoint. Fine-tuned or adapter
weights are loaded by that server. To switch visual checkpoints, change
`image_model.model_id` to an official editing model ID or local Mage checkpoint
directory. For `microsoft/Mage-Flow-Edit-Base`, also set `steps = 30` and `cfg = 5.0`.
Turbo uses `steps = 4` and `cfg = 1.0`. Swapping either checkpoint leaves CUWM unchanged.
Adding another backend requires its adapter and one explicit branch in
`build_cuwm`; there is no discovery framework.

The loader uses Python's standard-library `tomllib` and requires both model tables.
Omit optional values to retain adapter defaults. For example, omitting `max_size`
keeps the source resolution; TOML has no `null` value.

The Mage adapter loads once on its first `generate` call using the official
`MageFlowPipeline.from_pretrained(model_id, device=...)`, then calls
`pipeline.edit([instruction], [image], ...)`. The first visual call can download
the checkpoint and associated assets. Constructing CUWM does not download them.
The API was checked against
[Microsoft Mage revision 76bec2b](https://github.com/microsoft/Mage/tree/76bec2bb3818863f470de7e867c2dc7f1d0bfd83/mage_flow),
including `pipeline.py`, the README, and its dependency files.

## Standalone inference

From the repository root in a configured project environment:

```bash
uv run python scripts/run_cuwm.py \
  --screenshot data/example.png \
  --action "Select column H" \
  --config config/cuwm.toml \
  --output-dir .artifacts/cuwm/inference
```

Supply an existing screenshot and a reachable textual inference endpoint.
The script writes `transition.txt` before visual inference, followed by
`predicted_next_state.png`. Add `--text-only` to produce only the description
without importing or loading the actual Mage pipeline. Use a fresh output
directory for each run so earlier outputs cannot be mistaken for new predictions.

## Optional GPU runtime

The main project targets Python 3.14. Mage's tested requirements target Python
3.11, torch 2.13, transformers 5.5, and diffusers 0.38, with a separate
FlashAttention 2 installation. They are deliberately excluded from the main
project dependencies. The official pipeline hardcodes BF16, so the adapter does
not advertise a dtype setting that the upstream API cannot honor.

Use a separate environment on a compatible Linux NVIDIA GPU machine. A CUDA
toolkit matching the installed torch build, build tools, and enough VRAM are
required. The stock BF16/FlashAttention 2 implementation does not support T4;
see the [upstream GPU requirements](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).
An RTX 3090 is in the supported hardware family, but this module's memory use
and inference quality have not been measured on it. CPU and macOS visual
inference are not validated. No quantization, offloading, or alternate attention
implementation is provided.

For a compatible CUDA 12.6 GPU host, the following is an unexecuted setup example
using Microsoft's pinned requirements. Adjust the torch wheel index to match
the actual CUDA toolkit before installation:

```bash
uv venv --python 3.11 .artifacts/cuwm-gpu
uv pip install --python .artifacts/cuwm-gpu/bin/python \
  torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .artifacts/cuwm-gpu/bin/python \
  -r https://raw.githubusercontent.com/microsoft/Mage/76bec2bb3818863f470de7e867c2dc7f1d0bfd83/mage_flow/requirements.txt
uv pip install --python .artifacts/cuwm-gpu/bin/python --no-deps \
  'mage-flow @ git+https://github.com/microsoft/Mage.git@76bec2bb3818863f470de7e867c2dc7f1d0bfd83#subdirectory=mage_flow'
uv pip install --python .artifacts/cuwm-gpu/bin/python setuptools wheel ninja
uv pip install --python .artifacts/cuwm-gpu/bin/python \
  --no-build-isolation flash-attn==2.8.3
uv pip install --python .artifacts/cuwm-gpu/bin/python \
  openai==3.35.1 python-decouple==3.8
```

Run only the standalone source package in that environment, avoiding the root
project's Python 3.14 dependency installation:

```bash
PYTHONPATH=src uv run --no-project --no-sync \
  --python .artifacts/cuwm-gpu/bin/python python scripts/run_cuwm.py \
  --screenshot data/example.png \
  --action "Select column H" \
  --config config/cuwm.toml \
  --output-dir .artifacts/cuwm/inference
```

These commands have not been run with GPU inference. For local source-tree
development without an active editable installation, prefix the project commands
with `PYTHONPATH=src` as well.

## Image fidelity and research differences

By default, the adapter retains the source size. It pads the right and bottom
edges to multiples of 16 for Mage, requests those explicit dimensions, then
removes only the added padding. It never crops source content or stretches it.
Set `max_size` to explicitly cap the longest side using aspect-preserving
downscaling, which is logged. The resulting PNG retains the downscaled dimensions;
it is not enlarged afterward. Alignment can require up to 15 extra inference
pixels per dimension. Mage's internal VLM conditioning image defaults to a
384-pixel longest side; `vl_cond_long_edge = 0` disables that cap, increasing cost.
The VAE path receives the full aligned image. Start within Mage's documented
512 to 2048 resolution range and consider the memory cost of native GUI images.

This is an architectural reproduction with configurable pretrained Qwen3-VL and
Mage backbones. The paper instead uses trained Qwen2.5-VL and
Qwen-Image-Edit-2509, including SFT and RL. This implementation does not reproduce
those weights, training procedures, or benchmark results. Its prompt accepts raw
action strings without the paper's Office-specific structured action context.

Pretrained models can invent UI details or alter unchanged text, cells, and icons.
Padding does not correct those model errors. Seed 2026 makes the requested
sampling reproducible within a fixed runtime, not necessarily across hardware.
The upstream Mage pipeline also applies its own content gate and watermarking;
it can return a refusal image for filtered input. This adapter retains upstream
behavior and never substitutes an unmodified screenshot on failure.

## Minimal verification

```bash
PYTHONPATH=src uv run --locked --no-sync pytest tests/test_cuwm.py
uv run --locked --no-sync isort --check-only src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync black --check src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync flake8 src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync ty check src/world_models scripts/run_cuwm.py tests/test_cuwm.py
```

The contract test uses lightweight substitutes to verify stage order, original
screenshot propagation, and prediction fields. It does not validate neural-model
quality or actual GPU execution. No full benchmark or full repository test run is
required for this module.
