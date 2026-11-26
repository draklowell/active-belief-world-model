# data/

Data card for the datasets used in this project. Raw data is not committed: `data/raw/*`,
`data/interim/*` and `data/processed/*` are gitignored, and only this file and the
`.gitkeep` placeholders are tracked.

## AgentNet

**AgentNet** (`xlangai/AgentNet` on Hugging Face), a dataset of computer-use trajectories
with annotations and screenshots. The full dataset is about 200 GB, almost all of it split
zip archives of screenshots, so this project fetches only parts of it.

- Hugging Face page: <https://huggingface.co/datasets/xlangai/AgentNet>
- License: MIT, as stated on the Hugging Face dataset card (checked 2026-10-23). Raw data is
  not committed, and the dataset's own terms apply to anything derived from it.
- Citation: arXiv 2508.09123, listed on the dataset card.

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

## CUWM transitions

CUWM is trained on transitions `(s_t, a_t, s_t+1)`: a screenshot, the action taken on it, and
the screenshot that follows. `scripts/build_cuwm_transitions.py` builds them from the pinned
ubuntu JSONL and the screenshot archive index, both under `data/raw/agentnet/`, without
downloading screenshots. It then draws the sample that receives textual transition labels:

```bash
uv run python scripts/build_cuwm_transitions.py --config config/cuwm-data.toml
uv run python scripts/download_agentnet_images.py \
  --transitions data/interim/cuwm/sample.parquet \
  --manifest data/interim/cuwm/images_manifest.csv
```

The second command fetches the current and next screenshots of every sampled transition into
`data/raw/agentnet/images/`. With the supplied configuration that is 6,436 screenshots, about
5.8 GB compressed.

### Configuration

`config/cuwm-data.toml` holds every option. All keys are required.

| Key                 | Supplied value                        | Meaning                                                            |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| `seed`              | `2026`                                | Seed of the split and sample shuffles                              |
| `osworld_tasks`     | `"evaluation_examples/test_all.json"` | OSWorld 1.0 manifest checked for overlap                           |
| `overlap_threshold` | `0.8`                                 | Minimum content-token containment of a near copy                   |
| `output_dir`        | `"data/interim/cuwm"`                 | Directory of the outputs                                           |
| `[split_fractions]` | `train`, `validation`, `test`         | Ordered share of instruction groups per split, sum 1               |
| `[sample_sizes]`    | 2,876, 98 and 339                     | Transitions to sample per split, same names as `[split_fractions]` |
| `[domain_weights]`  | One weight per AgentNet domain        | Relative share of each domain in every split's sample              |

The sample sizes are the training, validation and test sizes of the CUWM paper (Guan et al.,
Table 6). The domain weights are the task counts of the OSWorld 1.0 `test_all.json` manifest for
the ten application domains. `infeasible` and `error_correction` have no OSWorld domain and
receive the weight 26, the number of OSWorld 1.0 tasks in `test_infeasible.json`. A domain
without a weight is not sampled.

### Construction

1. **Pairing.** Each step forms a transition with the step whose `index` follows directly.

2. **Actions.** The AgentNet `code` is parsed with `ast`, never executed, and converted to the
   `computer_use` arguments of the Qwen3-VL agent, including the `action` name. Coordinates are
   fractions of the screen and become `round(fraction * 999)`, the inverse of the agent's
   `adjust_coordinates`. Every converted action must pass the agent's option schema,
   `build_option_schema`, with the AgentNet `action` text as the description.

   | AgentNet code                         | Tool call arguments                            |
   | ------------------------------------- | ---------------------------------------------- |
   | `pyautogui.click(x, y)`               | `left_click` with `coordinate`                 |
   | `pyautogui.doubleClick(x, y)`         | `double_click` with `coordinate`               |
   | `pyautogui.rightClick(x, y)`          | `right_click` with `coordinate`                |
   | `pyautogui.moveTo(x, y)`              | `mouse_move` with `coordinate`                 |
   | `pyautogui.write(message)`            | `type` with `text`                             |
   | `pyautogui.press(key)`                | `key` with `keys = [key]`                      |
   | `pyautogui.hotkey(keys)`              | `key` with `keys`                              |
   | `moveTo` calls, then `scroll` calls   | `scroll`, last `coordinate`, summed `pixels`   |
   | `computer.wait()`                     | `wait`                                         |
   | `computer.terminate(status)`          | `terminate` with `status`                      |
   | `moveTo` then `dragTo`, `tripleClick` | Unsupported, no equivalent in the action space |

   A `left_click_drag` only names its end point and starts at the current cursor, which in
   AgentNet is not at the drag start, so drags are excluded. Scroll amounts are copied
   unchanged because both AgentNet and the agent pass them directly to `pyautogui.scroll`.

3. **Exclusions.** Each step receives the first matching reason in this order, or none when it
   is eligible: `osworld_overlap`, `last_step`, `non_consecutive_index`, `missing_image` (absent
   from the archive index), `unsupported_action`, `invalid_action`, `terminate`,
   `unchanged_screenshot` (equal CRC32 and size of both screenshots), and `duplicate_transition`
   (the screenshot and action of an earlier eligible transition).

4. **OSWorld overlap.** Instructions are normalized (NFKC, lowercase, a leading `TaskN:` prefix
   removed, punctuation dropped). An AgentNet task overlaps when any of `instruction`,
   `natural_language_task` or `actual_task` equals an OSWorld 1.0 instruction, contains one of at
   least 30 characters verbatim, or contains at least `overlap_threshold` of its content tokens
   (words outside a stopword list, at least three). Overlapping tasks are excluded as a whole.

5. **Splits.** Tasks are grouped by normalized instruction, shuffled with the seed inside each
   domain, and cut at the cumulative `[split_fractions]` shares. A task and its duplicates stay in one
   split, and every trajectory belongs to exactly one split.

6. **Sample.** For every split, its `[sample_sizes]` count is divided across domains by the largest remainder
   method, and eligible transitions are drawn uniformly within each split and domain. No
   success filter is applied, because a failed trajectory still records real screen changes;
   `judge_status` (`completed`, `failed` or `unscored`) is kept for analysis.

### Outputs

| File in `data/interim/cuwm/` | Contents                                                             |
| ---------------------------- | -------------------------------------------------------------------- |
| `transitions.parquet`        | Every step with its split and exclusion reason                       |
| `sample.parquet`             | Sampled eligible transitions to annotate                             |
| `osworld_overlap.csv`        | Excluded tasks, best OSWorld match, exactness and containment        |
| `build_info.json`            | Revision, settings, exclusion counts, sample coverage and image size |
| `images_manifest.csv`        | Downloaded screenshots of the sample with sha256 digests             |
| `labels.jsonl`               | Teacher labels of the sample with teacher metadata                   |

Both Parquet files have one row per step with `task_id`, `domain`, `instruction`, `index`,
`image`, `next_image`, `code`, `tool_arguments` (JSON `computer_use` arguments),
`action_name`, `gui_description` (the AgentNet `action` text), `reflection` (AgentNet's own
comparison of the two screenshots, null on about a third of the sample), `next_observation`
(AgentNet's description of the next screenshot, the `observation` of the next step),
`judge_status`, the
CRC32 and size of both screenshots, `exclusion_reason` and `split`. The `tool_arguments` keep the
`action` argument so that either tool call layout of the agent can be rendered from them.

### Annotation prompt

The textual transition labels `Δ_t` are written by a teacher VLM that sees both screenshots,
using [`cuwm-qwen3vl/transition_annotation.j2`](../src/prompts/templates/cuwm-qwen3vl/transition_annotation.j2).
The template includes the world model's inference prompt,
`cuwm-qwen3vl/textual_state_transition.j2`, verbatim and asks the teacher to answer it as an
ideal world model would, so labels follow the same structure, region order and quoting rules
the trained model is prompted with. Additional rules ground every change in the next
screenshot, follow the screenshot when it disagrees with the action or its description, ignore
the clock, exclude the task goal, and keep the answer under 300 words to fit the world model's
512 token limit.

| Slot               | Value                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `image`            | Marker of the current screenshot, attached first                                          |
| `next_image`       | Marker of the next screenshot, attached second                                            |
| `action`           | `{"name": "computer_use", "arguments": ...}` built from `tool_arguments`, as at inference |
| `gui_description`  | The AgentNet `action` text                                                                |
| `next_observation` | The `next_observation` column as an untrusted hint, or `None` to omit it                  |

The task instruction and `reflection` are not given to the teacher. `reflection` exists only for
judged tasks, so its coverage depends on the domain, and both AgentNet texts mention the task
goal. `next_observation` exists for every step and is used only to compare labelling with and
without it.

### Teacher server

The teacher runs on vLLM through the project's inference server launcher, on a Colab G4 GPU
with an A100 80GB as backup:

```bash
uv run python scripts/host_inference_server.py \
  --config config/server-cuwm-teacher.json --backend colab --accelerator gpu/G4
```

Use `--accelerator gpu/A100` for the backup. The launcher needs the `colab` CLI logged in and
`TAILSCALE_AUTH_KEY` and `HF_TOKEN` in `.env`. The server joins Tailscale as `teacher-server`,
so clients use `http://teacher-server:8000/v1`.

| Key in `config/server-cuwm-teacher.json` | Value                                      | Reason                                                   |
| ---------------------------------------- | ------------------------------------------ | -------------------------------------------------------- |
| `model`                                  | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`       | Teacher checkpoint, 32.3 GB of FP8 weights               |
| `revision`                               | `d9748a51ae66354c4dad665aab2c71f26cf2c8cd` | Every label comes from the same weights                  |
| `served_model_name`                      | `cuwm-teacher`                             | Clients keep one name when the pilot swaps the teacher   |
| `max_model_len`                          | 16384                                      | Two 1920x1080 screenshots, the prompt and the answer fit |
| `enforce_eager`                          | `false`                                    | CUDA graphs for throughput on a single large GPU         |
| `limit_mm_per_prompt`                    | `{"image": 2}`                             | Current and next screenshot in one request               |

`revision` and `limit_mm_per_prompt` are optional server options that map to the vLLM flags
`--revision` and `--limit-mm-per-prompt`; other server configurations omit them.

### Labelling

`scripts/label_cuwm_transitions.py` sends every sampled transition to the teacher server and
appends one JSON record per label to the output file:

```bash
uv run python scripts/label_cuwm_transitions.py --config config/cuwm-labels.toml \
  --output data/interim/cuwm/labels.jsonl --gpu G4
```

`--split` and `--limit` restrict a run to the first transitions of one split in task id and
step index order, for example `--split validation --limit 30` for a pilot.
`--include-next-observation` adds the hint. `--gpu` names the GPU of the server, which the
client cannot detect, and is stored with every label.

| Key in `config/cuwm-labels.toml` | Supplied value                       | Meaning                                                    |
| -------------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| `transitions`                    | `"data/interim/cuwm/sample.parquet"` | Transitions to label                                       |
| `images_dir`                     | `"data/raw/agentnet/images"`         | Directory of the downloaded screenshots                    |
| `server_config`                  | `"config/server-cuwm-teacher.json"`  | Server configuration the checkpoint and revision come from |
| `workers`                        | `8`                                  | Requests in flight, batched by vLLM                        |
| `[text_model]`                   | `cuwm-teacher` at `teacher-server`   | `OpenAITextModel` options, `max_tokens` 512                |

`max_tokens` is the world model's own limit, so a label the student could not reproduce within
512 tokens is rejected as truncated. Every screenshot of the run must be on disk before the first
request. The script checks that the server serves the checkpoint named by `server_config`, so a
label is never stored with the metadata of another model.

Each record holds `task_id` and `index`, which identify the transition in `sample.parquet`,
`split`, `label`, `has_next_observation_hint`, `labelled_at` and `teacher` with the
`checkpoint`, `revision`, `served_model`, `gpu`, `max_tokens`, `temperature` and `template` of the
request. A rerun with the same output skips the transitions already in it, so an interrupted run
resumes. A truncated or rejected label is logged and not written, and a rerun requests it again.

## Layout

| Path                 | Contents                                                         |
| -------------------- | ---------------------------------------------------------------- |
| `data/raw/agentnet/` | Downloaded JSONL files, `REVISION.txt`, image index and manifest |
| `data/interim/cuwm/` | CUWM transitions, their sample, overlap report and build summary |
| `data/processed/`    | Model-ready data, currently empty                                |

Exploration of the data is in `notebooks/01_agentnet_eda.ipynb` and
`notebooks/02_agentnet_images_eda.ipynb`.
