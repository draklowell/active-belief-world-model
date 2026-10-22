# Computer-Using World Model

CUWM predicts one hypothetical GUI transition without executing the candidate action.
It reproduces the two-stage factorization in
[Guan et al., section 3.1 and Figure 2](https://arxiv.org/abs/2602.17365):

1. CUWM prepares system and action prompts, then calls the textual backend to
   predict the immediate UI changes.
2. CUWM prepares an edit instruction with preservation requirements, then calls the
   visual backend using the original screenshot.

Both stages receive encoded screenshot `bytes`. The textual result is a plain
string, and the visual result is PNG-encoded `bytes`. `CUWM.predict` returns a
`CUWMPrediction` containing `text_description` and `predicted_screenshot`.

`cuwm.py` owns orchestration and domain prompts. `base.py` defines the backend
protocols, and `types.py` defines the prediction dataclass. Importing `world_models`
uses only the standard library. Clients in `backends/` encode requests and call
remote inference APIs. Model loading and GPU execution run on servers.

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
For dependency injection, construct `CUWM(text_model=..., image_model=...)` with
objects implementing these contracts:

- `TextTransitionModel.generate(screenshot, prompt, *, system_prompt) -> str`
- `ImageTransitionModel.generate(screenshot, prompt) -> bytes`

CUWM supplies domain instructions. Backends forward the prepared prompts to
inference endpoints without adding GUI instructions or executing actions.

## Configuration

`config/cuwm.toml` follows the shared
[configuration conventions](../../AGENTS.md#configuration-conventions): TOML for
runtime configuration and JSON for project and service metadata. Each model table
contains a `backend` identifier and that client's constructor options.

| Section       | Backend                 | Default served model             |
| ------------- | ----------------------- | -------------------------------- |
| `text_model`  | `openai_compatible_vlm` | `qwen3-vl`                       |
| `image_model` | `mage_flow`             | `microsoft/Mage-Flow-Edit-Turbo` |

The textual client reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The image client
reads `IMAGE_BASE_URL` and `IMAGE_API_KEY`. Set these through environment variables
or the project's `.env`; both clients use `python-decouple`. Base URLs include the
API prefix, usually `/v1`. Each table accepts `base_url` to override its endpoint
and `api_key_env` to select a different secret variable. Keep credentials out of
TOML. Configure the image endpoint separately from the text endpoint.

Both tables use `model` for the identifier exposed by their server. Checkpoints,
local model directories, device selection, precision, and internal VLM settings
belong to server configuration. To switch checkpoints, serve the desired weights
and update the corresponding `model`. Adding a backend requires its adapter and
one explicit branch in `build_cuwm`.

The textual client accepts `max_tokens`, `temperature`, and `timeout`. It preserves
screenshot dimensions when encoding the request. Empty or truncated replies fail.

The Mage client accepts `steps`, `cfg`, `seed`, `max_size`, and `timeout`. Turbo
uses `steps = 4`, `cfg = 1.0`, and `seed = 2026`. A server serving Base normally
uses `steps = 30` and `cfg = 5.0`. Both API clients are initialized in their
constructors, so both endpoints' settings must be available when CUWM is built.

Both clients retain the OpenAI SDK's default two retries. Their `generate` calls
also follow the agent's `backoff.constant` policy: five attempts with a 30-second
interval and default full jitter for SSL, connection, rate-limit, bad-request,
and server errors. Each application attempt can include SDK retries, allowing up
to 15 HTTP requests for a persistent retryable server failure. Exhausted API
failures propagate. Invalid image data and unusable model output are not retried.

The loader requires `[text_model]` and `[image_model]`. Omit optional values to
retain defaults. For example, omitting `max_size` keeps source resolution; TOML
has no `null` value.

## Image API contract

The Mage client sends `POST /v1/images/edits` using the OpenAI SDK. The multipart
request contains `model`, the PNG `image`, the prepared `prompt`, `size` as
`WIDTHxHEIGHT`, `n = 1`, `response_format = "b64_json"`, and
`output_format = "png"`. Extension fields `num_inference_steps`, `guidance_scale`,
and `seed` map from `steps`, `cfg`, and `seed`. These follow the documented
[vLLM-Omni image-edit API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_edit_api/).

The server must implement this contract for the configured Mage model and return
`data[0].b64_json` containing a PNG with the requested dimensions. A service
wrapping the official Mage pipeline maps the extensions to its `steps`, `cfg`,
and `seeds` arguments and honors the requested width and height. Missing image
data, invalid image bytes, unexpected sizes, and API failures propagate.

Deploy a compatible image service before visual inference. The existing Qwen
server configuration serves the textual model. Mage is absent from the pinned
vLLM-Omni 0.30
[supported model list](https://docs.vllm.ai/projects/vllm-omni/en/v0.30.0/models/supported_models/),
so that server cannot be assumed to serve Mage without an additional integration.
This module implements the remote client contract. A Mage API deployment and real
GPU inference have not been validated.

## Standalone inference

From the repository root with the inference endpoints configured:

```bash
uv run python scripts/run_cuwm.py \
  --screenshot data/example.png \
  --action "Select column H" \
  --config config/cuwm.toml \
  --output-dir .artifacts/cuwm/inference
```

Supply an existing screenshot. The script writes `transition.txt` before visual
inference, followed by `predicted_next_state.png`. Add `--text-only` to produce
only the description. Use a fresh output directory for each run to distinguish
new predictions from earlier outputs. For source-tree development without an
active editable installation, prefix commands with `PYTHONPATH=src`.

The world-model client imports no Mage pipeline, torch, or diffusion runtime.
Mage's Python, CUDA, BF16, and attention requirements apply to its inference
server. Project dependency compatibility is maintained in the OSWorld setup
branch; resolve its native-build requirements before treating `uv sync` or
ordinary `uv run` as validated.

## Image fidelity and research differences

The image client retains source size by default. It pads the right and bottom
edges to multiples of 16, requests those dimensions, validates the response size,
and removes only the added padding. Source content is not cropped or stretched.
Set `max_size` to cap the longest side using aspect-preserving downscaling, which
is logged. The output retains those downscaled dimensions.

This is an architectural reproduction using configurable pretrained backbones.
The paper uses trained Qwen2.5-VL and Qwen-Image-Edit-2509 with SFT and RL. This
implementation does not reproduce its weights, training, or benchmark results.
Prompts accept raw action strings without the paper's Office-specific structured
action context. Models may alter unrelated UI details; alignment does not prevent
those errors. Seed 2026 requests reproducible sampling within a fixed server
runtime, subject to that server's implementation and hardware.

## Minimal verification

```bash
PYTHONPATH=src uv run --locked --no-sync pytest tests/test_cuwm.py
uv run --locked --no-sync isort --check-only src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync black --check src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync flake8 src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync ty check src/world_models scripts/run_cuwm.py tests/test_cuwm.py
```

The tests cover stage order, original screenshot propagation, prepared prompts,
SDK request serialization, API image decoding, padding removal, and missing image
data. HTTP responses are simulated at the transport boundary. They do not validate
OSWorld evaluators, dependency builds, a deployed Mage server, or model quality.
No checkpoint download, benchmark run, or GPU inference is required.
