# Computer-Using World Model

CUWM predicts one hypothetical GUI transition without executing the candidate action.
It reproduces the two-stage factorization in
[Guan et al., section 3.1 and Figure 2](https://arxiv.org/abs/2602.17365):

1. CUWM renders the transition prompt of its prompt set, by default the complete
   Appendix A.5.2 prompt, using the supplied application and structured action
   context, then calls the textual backend to predict the immediate UI changes.
2. CUWM prepares an edit instruction with preservation requirements, then calls the
   visual backend using the original screenshot.

Both stages receive encoded screenshot `bytes`. The textual result is a plain
string, and the visual result is PNG-encoded `bytes`. `CUWM.predict` returns a
`CUWMPrediction` containing `text_description` and `predicted_screenshot`.

`cuwm.py` owns orchestration and invokes the prompt renderer. The complete paper
prompt templates live in [`../prompts/templates/cuwm/`](../prompts/templates/cuwm/).
The desktop prompt set used by the Qwen3-VL CUWM agent lives in
[`../prompts/templates/cuwm-qwen3vl/`](../prompts/templates/cuwm-qwen3vl/), and the
[CUWM changes report](cuwm/README.md) documents how it differs from the paper.
`base.py` defines the backend protocols, and `types.py` defines the prediction dataclass.
Importing `world_models` loads the shared Jinja2 prompt renderer. Clients in `backends/`
encode requests and call remote inference APIs. Model loading and GPU execution
run on servers.

## Python API

```python
from pathlib import Path

from world_models.config import build_cuwm, load_cuwm_config

cuwm = build_cuwm(load_cuwm_config("config/cuwm.toml"))
screenshot = Path("data/example.png").read_bytes()
action = (
    '{"function": "click", "args": '
    '{"control_label": 15, "coordinate": null, "button": "left"}}'
)
gui_description = (
    "Click the UI control identified by accessibility control_label 15 "
    "using the left mouse button."
)
prediction = cuwm.predict(
    screenshot=screenshot,
    action=action,
    app_name="Microsoft Excel",
    gui_description=gui_description,
)
Path("predicted_next_state.png").write_bytes(prediction.predicted_screenshot)
```

Use `cuwm.generate_text_desc(screenshot, action, gui_description=..., app_name=...)` or
`cuwm.generate_image(screenshot, text_description)` to run stages independently.
`generate_text_desc` and `predict` require `gui_description` as a keyword
argument; `app_name` is an optional keyword argument. The paper prompt set
requires an application name and a function-level description of the candidate
action and its arguments, so omitting `app_name` with it raises a Jinja2
`UndefinedError` before inference. The `cuwm-qwen3vl` prompt set identifies the
application from the screenshot and ignores a supplied `app_name`. A GUI
description is not a screenshot caption. Keep the complete structured action
representation, including targeting arguments. Blank textual inputs fail; CUWM
does not fabricate missing context.

For dependency injection, construct
`CUWM(text_model=..., image_model=..., prompt_set="cuwm")` with objects
implementing these contracts:

- `TextTransitionModel.generate(screenshot, prompt, *, system_prompt) -> str`
- `ImageTransitionModel.generate(screenshot, prompt) -> bytes`

The canonical paper template is distinct from the runtime-substituted prompt.
The runtime renderer supplies textual variables and an image-attachment marker;
the actual screenshot is attached using the backend's existing multimodal image
content. The complete prompt appears once in user text, with an empty system
message. This explicit message-role choice is an implementation arrangement,
not a claim that the paper prescribes OpenAI message roles.

Backends forward prepared requests without adding GUI instructions or executing
actions. The image adapter appends stable-diffusion.cpp sampling controls to the
request prompt as server metadata. The image stage's existing preservation
instruction is custom project text, separate from the four canonical paper prompts.
The three agent-evaluation
and judge prompts are stored and loadable without an execution pipeline.

## Prompt templates

Executable prompts use the shared `prompts.render(template_name, **context)` API.
Jinja2 loads `.j2` files from `src/prompts/templates/` through a package loader,
independently of the working directory. Missing variables raise `UndefinedError`;
plain-text values, JSON, and XML are not HTML-escaped or recursively rendered.
The `tojson` filter is `json.dumps` with `ensure_ascii=False`, so it keeps key order
and does not apply the HTML-safe escapes of the Jinja2 built-in filter.

CUWM templates are named `cuwm/textual_state_transition.j2`,
`cuwm/image_preservation.j2`, `cuwm/action_option_generation.j2`,
`cuwm/action_selection.j2`, and `cuwm/transition_judge.j2`. The four paper templates
retain the complete instructions from
[arXiv:2602.17365v1, Appendix A.5, pages 27–35](https://arxiv.org/pdf/2602.17365v1#page=27).
Source-derived hashes and pre-migration output snapshots protect the original
wording and formatting. The former Markdown listings remain available in
[Git history](https://github.com/draklowell/active-belief-world-model/tree/d56f354/src/world_models/prompts/cuwm/paper).
The judge's literal doubled JSON braces are preserved with a Jinja2 raw block.
The action-selection template accepts
`instruction`, `image`, and five `predicted_images` markers; its published empty
action lines remain unchanged. The judge accepts `PRED` and `GT`.

CUWM renders `<prompt_set>/textual_state_transition.j2` and
`<prompt_set>/image_preservation.j2`, where `prompt_set` names a template directory.
The default `cuwm` set holds the paper prompts. The `cuwm-qwen3vl` set holds a
desktop adaptation of the transition prompt without `app_name`, and its image
template includes `cuwm/image_preservation.j2` unchanged. The same directory holds
the prompts of the [Qwen3-VL CUWM agent](../agents/qwen3vl_cuwm/README.md).

The CUWM facade retains nonblank input validation and the image attachment marker.
Multimodal message assembly and JSON serialization remain in Python. The Qwen
tool schema lives in `qwen3vl/tools.json.j2`; JSON encoding protects its string fields.
Qwen3VLAgent uses the same renderer for its description, system, user, action
descriptions, and history templates.

## Configuration

`config/cuwm.toml` follows the shared
[configuration conventions](../../AGENTS.md#configuration-conventions): TOML for
runtime configuration and JSON for project and service metadata. Each model table
contains a `backend` identifier and that client's constructor options.

| Section       | Backend                  | Adapter                     |
| ------------- | ------------------------ | --------------------------- |
| `text_model`  | `openai_compatible_text` | `OpenAITextModel`           |
| `image_model` | `stable_diffusion_image` | `StableDiffusionImageModel` |

The optional root key `prompt_set` selects the prompt template directory, for
example `prompt_set = "cuwm-qwen3vl"` placed before the tables. Omit it to use the
paper prompts in `cuwm`.

OpenAI compatibility describes the API communication contract. It does not
require OpenAI-hosted models. `OpenAITextModel` uses Chat Completions, while
`StableDiffusionImageModel` uses the stable-diffusion.cpp OpenAI-compatible Image
Editing endpoint with its server-specific parameter encoding.

The textual client reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The image client
reads `IMAGE_BASE_URL` and `IMAGE_API_KEY`. Set these through environment variables
or the project's `.env`; both clients use `python-decouple`. Base URLs include the
API prefix, usually `/v1`. Each table accepts `base_url` to override its endpoint
and `api_key_env` to select a different secret variable. Keep credentials out of
TOML. Configure the image endpoint separately from the text endpoint.

Each table requires `model`, the identifier or alias exposed by that inference
server. There is no model identifier default in either adapter. `backend` selects
the server API contract, `model` is the identifier submitted with the request,
and `base_url` selects the inference endpoint. Checkpoints, local model directories,
devices, and precision belong to the server. stable-diffusion.cpp uses its loaded
model; the request's model string does not load or switch weights. Point the client
at the server with the desired model loaded and use its supported identifier.
Adding a different API contract requires its adapter and one explicit branch
in `build_cuwm`.

The textual client accepts `max_tokens` (default 512), `temperature` (default
0.0), and `timeout` (default 120 seconds). It preserves screenshot dimensions
when encoding the request. Its vision-capable Chat Completions endpoint must
accept a system prompt and base64 PNG `image_url` user content, and return
nonempty text in `choices[0].message.content`. Empty or truncated replies fail.

The image client accepts `max_size` (optional positive longest-side cap),
`padding_multiple` (positive alignment divisor, default 1), `steps`, `cfg`, `seed`,
and `timeout` (default 300 seconds). Sampling controls are optional and omitted
when not configured, preserving the server's defaults. `steps` maps to
`sample_params.sample_steps`; `cfg` sets both `sample_params.guidance.txt_cfg`
and `sample_params.guidance.img_cfg`; `seed` is a native root field. Explicit zero
values are preserved. No Mage-specific sampling defaults or alignment are added.

The supplied configuration explicitly retains the previous Qwen and Mage setup:

```toml
[text_model]
backend = "openai_compatible_text"
model = "qwen3-vl"

[image_model]
backend = "stable_diffusion_image"
model = "microsoft/Mage-Flow-Edit-Turbo"
steps = 4
cfg = 1.0
seed = 2026
padding_multiple = 16
```

For another image model served by stable-diffusion.cpp, set its identifier and
endpoint and choose or omit sampling overrides for that model. The image adapter
requests `response_format = "b64_json"` and `output_format = "png"`. Replace the
previous `[image_model.extra_body]` table with root `steps`, `cfg`, and `seed`
options. Sampling controls are no longer submitted as extra multipart fields.
The former `openai_compatible_vlm`, `mage_flow`, and `openai_compatible_image`
backend identifiers are replaced by the identifiers above.

Both API clients are initialized in their constructors, so both endpoints'
settings must be available when CUWM is built, including for the text-only CLI.

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

`StableDiffusionImageModel` sends `POST /images/edits` relative to the configured API base
URL, normally `/v1`. The multipart request contains the configured `model`, a
single RGB PNG `image`, the prepared `prompt`, `size` as `WIDTHxHEIGHT`, and
`n = 1`, plus the base64-response and PNG-format options above. When sampling
overrides are configured, their JSON is appended to the caller's prompt inside
`<sd_cpp_extra_args>...</sd_cpp_extra_args>`, as required by the
[stable-diffusion.cpp compatibility API](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md#sd_cpp_extra_args).
The server extracts those controls and removes the block before image generation.
The adapter makes one image-edit call per attempt; existing retries remain in place.

The server must accept those fields and return `data[0].b64_json` containing a
decodable image with exactly the requested dimensions. The client normalizes
the result to PNG. URL-only responses are unsupported and fail with an error
identifying the required inline field. Invalid base64, undecodable images,
unexpected sizes, and exhausted API failures propagate; no image download or
response fallback is attempted. A server that forces different output dimensions
does not satisfy this contract. Configure resizing or alignment to match its
requirements.

Deploy stable-diffusion.cpp with the desired image-editing model and support for
the documented inline controls before visual inference. The existing Qwen server
configuration serves the textual model. Other OpenAI-compatible image servers
may not understand these controls and require their own adapter. This module
implements the remote client contract; the deployed server revision, Mage support,
and real image inference have not been validated here.

## Standalone inference

From the repository root with the inference endpoints configured:

```bash
uv run python scripts/run_cuwm.py \
  --screenshot data/example.png \
  --action '{"function":"click","args":{"control_label":15,"coordinate":null,"button":"left"}}' \
  --app-name "Microsoft Excel" \
  --gui-description "Click accessibility control_label 15 using the left mouse button." \
  --config config/cuwm.toml \
  --output-dir .artifacts/cuwm/inference
```

Supply an existing screenshot. The script writes `transition.txt` before visual
inference, followed by `predicted_next_state.png`. Add `--text-only` to produce
only the description. `--gui-description` is required in both modes because
both perform textual transition generation. `--app-name` is optional on the
command line but required by the default paper prompt set; omit it only with a
configuration whose `prompt_set` infers the application, such as `cuwm-qwen3vl`. Use a fresh output
directory for each run to distinguish
new predictions from earlier outputs. For source-tree development without an
active editable installation, prefix commands with `PYTHONPATH=src`.

The world-model client imports no Mage pipeline, torch, or diffusion runtime.
Model loading, compute devices, precision, and runtime requirements belong to
the inference server. Project dependency compatibility is maintained in the OSWorld setup
branch; resolve its native-build requirements before treating `uv sync` or
ordinary `uv run` as validated.

## Image fidelity and research differences

The image client retains source size by default. When `padding_multiple` is
greater than 1, it pads the right and bottom edges with white to that alignment,
requests those dimensions, validates the response size, and removes only the
added padding. The supplied Mage configuration uses 16. Source content is not
cropped or stretched. Set `max_size` to cap the longest side using
aspect-preserving downscaling, which is logged. The output retains those
downscaled dimensions.

This is an architectural reproduction using configurable pretrained backbones.
The paper uses trained Qwen2.5-VL and Qwen-Image-Edit-2509 with SFT and RL. This
implementation does not reproduce its weights, training, or benchmark results.
The full Office-oriented paper prompt now receives explicit application and
function-level context, but actions remain caller-supplied strings without an
action parser or generated accessibility annotations. Models may alter unrelated
UI details; alignment does not prevent those errors. The supplied Mage
`seed = 2026` requests reproducible
sampling within a fixed server runtime, subject to that server's implementation
and hardware.

## Minimal verification

```bash
PYTHONPATH=src uv run --locked --no-sync pytest tests/test_cuwm.py tests/test_cuwm_prompts.py
uv run --locked --no-sync isort --check-only src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync black --check src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync flake8 src/world_models scripts/run_cuwm.py tests/test_cuwm.py
uv run --locked --no-sync ty check src/world_models scripts/run_cuwm.py tests/test_cuwm.py
```

The tests cover stage order, original screenshot propagation, prepared prompts,
model selection, SDK request serialization and retries, API image decoding,
configured resizing and padding, malformed responses, and text-only CLI behavior.
They verify the supplied Mage configuration's inline sampling controls, partial
overrides including zero values, and another image model without implicit sampling
controls. HTTP responses are
simulated at the transport boundary. They do not validate OSWorld evaluators,
dependency builds, a deployed image service, or model quality. No checkpoint
download, benchmark run, or GPU inference is required.

Prompt tests additionally verify source-derived fingerprints of all four complete
listings, structured examples, judge dimensions, safe substitution, required
context, and package-resource loading outside the repository directory.
