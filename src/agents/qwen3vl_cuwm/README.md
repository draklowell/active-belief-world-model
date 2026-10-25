# Qwen3-VL CUWM agent

`Qwen3VLCUWMAgent` in [`qwen3vl_cuwm.py`](../qwen3vl_cuwm.py) is a screenshot agent
that simulates several candidate actions with the Computer-Using World Model
(CUWM) before executing one of them. It implements the world-model-guided
test-time action search of
[Guan et al., section 3.4 and Appendix A.4](https://arxiv.org/abs/2602.17365) on
top of the action space of the Qwen3-VL agent in [`qwen3vl.py`](../qwen3vl.py).

The agent does not inherit from `Qwen3VLAgent` or `CUWM`. It owns its policy
client and history, and it holds a `CUWM` instance by composition. Functions
used by both Qwen3-VL agents live in [`qwen3vl_common.py`](../qwen3vl_common.py):
screenshot resizing and encoding, the `computer_use` tool definition rendered
from the `qwen3vl` templates, chat completion requests with retries, and the
conversion of tool calls into PyAutoGUI commands. `Qwen3VLAgent` keeps its behavior and now uses the same
functions.

## Notation

| Symbol     | Meaning                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| `A`        | `computer_use` action space of `qwen3vl.py`: a description and a JSON call |
| `S`        | State space, the set of possible screenshots of the environment            |
| `s(t)`     | Current screenshot at step `t`                                             |
| `a(t)`     | Action executed at step `t` after observing `s(t)`                         |
| `a_i(t)`   | Candidate action `i` generated at step `t`                                 |
| `s_i(t+1)` | Predicted screenshot after executing `a_i(t)` in `s(t)`                    |
| `Δ_i(t)`   | Predicted textual description of the UI changes caused by `a_i(t)`         |
| `g`        | Task instruction (goal)                                                    |
| `N`, `K`   | Number of candidates (`num_options`) and history length (`history_n`)      |

## Pipeline

Each call to `predict` performs one step:

1. Observe `s(t)`. The screenshot is resized with `smart_resize` to multiples of
   32 pixels and encoded as PNG. The same processed screenshot is used by the
   policy and by CUWM.
2. Generate candidates
   `{a_i(t)} for i = 1..N ~ P({a} | s(t), {a(k), s(k)} for t - K <= k < t, g, A)`
   with the adapted
   [action option generation prompt](../../prompts/templates/cuwm-qwen3vl/action_option_generation.j2)
   as the system prompt. The prompt contains `g`, the `computer_use` tool
   definition with its screen resolution, and `N`. The user message contains the
   `K` most recent screenshots, each followed by its executed action, and then
   `s(t)`. When `K = 0`, the prompt is rendered without its history input and
   the user message contains only `s(t)`. The policy answers with one JSON array whose elements
   contain `thoughts`, `action` (one imperative sentence), and `tool_call`
   (`{"name": <action>, "arguments": {...}}`, where `<action>` is one of the
   `computer_use` actions and `arguments` omits the `action` parameter). The
   response is validated
   before simulation (see [Validation](#validation)).
3. For each candidate, call `CUWM.predict(s(t), action, gui_description=...)`
   without `app_name`, where the action is the tool call JSON and the GUI
   description is the candidate's `action` sentence. CUWM uses the
   `cuwm-qwen3vl` prompt set, which identifies the application from the
   screenshot:
   1. `Δ_i(t) ~ P(Δ | s(t), a_i(t))` with the CUWM textual stage, served by the
      same Qwen3-VL endpoint (`OPENAI_BASE_URL`) as the policy.
   2. `s_i(t+1) ~ P(s | s(t), Δ_i(t))` with the CUWM visual stage, served by
      MageFlowEdit-Turbo on stable-diffusion.cpp (`IMAGE_BASE_URL`).
4. Select `j ~ P(1..N | {a_i(t), s_i(t+1)} for i = 1..N, {a(k), s(k)} for t - K <= k <= t, g)`
   with the adapted [action selection prompt](../../prompts/templates/cuwm-qwen3vl/action_selection.j2)
   as the system prompt. The user message contains the same history, `s(t)`, and each
   option as its tool call JSON and its predicted screenshot. The
   policy answers with `{"action_idx": j, "thought": ...}`, and `j` is validated
   against the number of candidates.
5. Execute `a(t) = a_j(t)`. Only the selected tool call is converted into
   PyAutoGUI code by the shared converter, with `name` restored as the
   `action` argument, using the original screen size, and returned to the
   environment. `terminate` becomes `DONE` and `wait` becomes `WAIT`. The pair
   `(s(t), a(t))` is appended to the history, where `a(t)` is the selected tool
   call JSON without its description, and the benchmark repeats from
   step 1 until `DONE` or the step limit.

`predict` returns a JSON trace with every candidate, its predicted transition
`Δ_i(t)`, and the raw selection response, which `scripts/benchmark.py` records
in the MLflow `predict` span. Exceptions from CUWM, such as a truncated
transition, propagate and fail the task.

## Validation

Both policy responses are validated with JSON Schema (draft 2020-12, through
`jsonschema.validate`) before their content is used. In both cases the first
JSON array or object in the response is decoded, so a surrounding Markdown code
fence is tolerated; a response without one is rejected.

`parse_candidates` validates each element of the candidate array against
`build_option_schema` and keeps at most the first `N` valid elements with
distinct tool calls, in generation order. An option is an object with an optional string `thoughts`, a
string `action` containing a non-whitespace character, and a `tool_call` whose
`name` is one of the `action` enumeration values of the rendered
`qwen3vl/tools.json.j2` tool definition, the same definition shown to the
policy (`key`, `type`, `mouse_move`, `left_click`, `left_click_drag`,
`right_click`, `middle_click`, `double_click`, `scroll`, `wait`, `terminate`).
Its `arguments` must match all of the following:

1. The remaining `parameters` of the tool definition, without `action`, with
   `additionalProperties: false`. This fixes the argument names, their JSON
   types (booleans are not numbers), and the `status` enumeration (`success`,
   `failure`).

2. Value constraints: `keys` is a nonempty array of strings with a
   non-whitespace character, `text` is not empty, `time` is not negative, and
   `coordinate` is exactly `[x, y]` with `0 <= x < width` and `0 <= y < height`.
   The coordinate frame is the one stated to the policy: 1000x1000 for relative
   coordinates, or the processed screenshot size for absolute coordinates.

3. One `if`/`then` rule per tool call `name` requiring the arguments its PyAutoGUI
   conversion uses, which the tool definition only states in prose:

   | Action                                                                                       | Required arguments |
   | -------------------------------------------------------------------------------------------- | ------------------ |
   | `key`                                                                                        | `keys`             |
   | `type`                                                                                       | `text`             |
   | `mouse_move`, `left_click`, `left_click_drag`, `right_click`, `middle_click`, `double_click` | `coordinate`       |
   | `scroll`                                                                                     | `pixels`           |
   | `terminate`                                                                                  | `status`           |
   | `wait`                                                                                       | none               |

Invalid elements are logged with the schema error message and discarded. A
valid element whose `tool_call` (`name` and `arguments`) equals that of an
earlier candidate is logged and discarded, even when its `thoughts` or `action`
sentence differ, so each distinct action is simulated only once. Only validated,
distinct candidates are simulated by CUWM. If none remain, CUWM and the
selection request are skipped and the raw candidate response is returned.

`parse_selection` validates the selection object against a schema requiring an
`integer` `action_idx` between 1 and the number of validated candidates.
JSON Schema integers exclude `true` and `false`, so a boolean is rejected
instead of being read as index 1 or 0. A whole-number float such as `2.0` is a
JSON Schema integer and selects option 2.

A step without a validated candidate or selection returns no commands and
leaves the history unchanged, so the next step observes the same screenshot, as
the base Qwen3-VL agent does after a malformed response. Arguments outside the
tool definition are rejected even when the base agent's converter reads them,
such as `duration` for `left_click_drag`, because the policy is never told about
them. A coordinate-free click, which the base agent executes at the current
cursor position, is rejected because CUWM cannot simulate an unknown target.

One step costs at most `N + 2` chat completions and `N` image edits, one
completion and one edit per distinct candidate plus generation and selection.

## Prompts

All agent prompts are Jinja2 templates in
[`src/prompts/templates/cuwm-qwen3vl/`](../../prompts/templates/cuwm-qwen3vl/), rendered with the shared
`prompts.render` environment and `StrictUndefined`, so a missing variable fails
instead of producing a partial prompt. The directory also holds the CUWM
transition prompt used by this agent, described in the
[CUWM changes report](../../world_models/cuwm/README.md).

| Template                      | Role   | Context                                                            |
| ----------------------------- | ------ | ------------------------------------------------------------------ |
| `action_option_generation.j2` | System | `has_history`, `instruction`, `actions` (tool dict), `num_options` |
| `action_selection.j2`         | System | `has_history`, `instruction`, `num_options`                        |
| `history_step.j2`             | User   | `step`, `action` (tool call dict), after each history screenshot   |
| `current_screenshot.j2`       | User   | none, before `s(t)`                                                |
| `action_option.j2`            | User   | `index`, `action` (tool call dict), before each predicted image    |

Message assembly and image attachments stay in Python. The tool definition and
tool calls are passed as dictionaries and serialized with the `tojson` filter,
which the shared environment defines as `json.dumps` with `ensure_ascii=False`.
Keys keep their insertion order, and quotes, `<`, `>`, `&`, and non-ASCII text
stay literal.
The two system prompts are adapted copies of the paper prompts in
[`cuwm/`](../../prompts/templates/cuwm/). Each is a single template whose
history input is enclosed in `{% if has_history %}` blocks. The agent sets
`has_history` to `K > 0`. With `K = 0`, every reference to previous steps is
omitted: the example thoughts describe the current screenshot instead of the
previous step, the drag instruction asks for an earlier `mouse_move` action,
and the messages contain no history parts. With `K > 0`, the history input is
described and, at the first step of a task, contains no parts, which the prompt
states explicitly ("This input is empty at the first step of a task").

Changes to the action option generation prompt:

- Office applications are replaced with desktop applications.
- The annotated screenshot and accessibility inputs are removed, together with
  the instruction to prefer `control_label` over coordinates. The prompt instead
  asks for the coordinate of the element center in the stated resolution.
- A previous steps input is added when `has_history` is true, and the reasoning
  checklist asks how the previous steps changed the UI.
- The `{a11y}` placeholder is removed and `{{ actions }}` receives the
  `computer_use` tool definition, including the environment description and the
  coordinate resolution of `qwen3vl.py`.
- Each element has an additional `action` field with one imperative sentence.
  `tool_call` has `name`, the action such as `left_click`, and `arguments`
  without the `action` parameter, instead of `function`, `args`, and `status`.
- A section stresses that the options are mutually exclusive alternatives for
  the next action rather than a multistep plan, with a self-check and a
  wrong and correct example.
- The click, type, and drag examples use `left_click`, `type`, and
  `left_click_drag`. A `key` shortcut example is added. The `FINISH` example is
  replaced by `terminate` with status `success` or `failure`.
- The duplicated "Annotated current screenshot" line of the published listing
  is not kept.

Changes to the action selection prompt:

- Office applications are replaced with desktop applications.
- The image markers (`[Current State Screenshot]`, `[Predicted Image i]`) and
  the five hard-coded options are removed from the prompt. The images and the
  options are sent in the user message with the paper's labels
  (`Action Option i`, `- Action:`, `- Predicted State Image: see below.`).
- The response format lists `thought` before `action_idx` inside a `json`
  code fence.
- The bound of `action_idx` is `{{ num_options }}`, the number of parsed
  candidates, instead of 5.
- A previous steps input is added when `has_history` is true, together with a
  `Progress` rule that discourages actions repeating an earlier action without
  changing the UI.
- The `predicted_state` references are renamed to `predicted_state_image`,
  matching the field described above them.

## Configuration

[`config/agent-qwen3vl-cuwm.toml`](../../../config/agent-qwen3vl-cuwm.toml)
contains the agent options. Omitted keys use the defaults below.

| Key               | Default              | Meaning                                                          |
| ----------------- | -------------------- | ---------------------------------------------------------------- |
| `model`           | `"qwen3-vl"`         | Served Qwen3-VL identifier for generation and selection          |
| `max_tokens`      | `32768`              | Maximum generated tokens per policy request                      |
| `temperature`     | `0.0`                | Policy sampling temperature                                      |
| `top_p`           | `0.9`                | Policy nucleus sampling mass                                     |
| `history_n`       | `4`                  | `K`, previous screenshot and action pairs; `0` disables history  |
| `coordinate_type` | `"relative"`         | `relative` 1000x1000 grid or `absolute` processed pixels         |
| `num_options`     | `5`                  | `N`, candidates requested per step                               |
| `cuwm_config`     | `"config/cuwm.toml"` | CUWM TOML configuration; its `prompt_set` becomes `cuwm-qwen3vl` |

The policy client reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`. CUWM reads the
endpoints configured in `config/cuwm.toml`, by default `OPENAI_BASE_URL` for the
textual stage and `IMAGE_BASE_URL` with `IMAGE_API_KEY` for the visual stage.
Run the benchmark from the repository root:

```bash
uv run python scripts/benchmark.py \
  --agent-path agents.qwen3vl_cuwm.Qwen3VLCUWMAgent \
  --agent-config config/agent-qwen3vl-cuwm.toml \
  --tasks evaluation_examples/test_small.json
```

## Differences from the CUWM paper

The paper evaluates world-model-guided action search offline (Section 3.4 and
Appendix A.4): a frozen VLM agent receives the current screenshot and the task
instruction, proposes five candidate actions with the Appendix A.5 prompt, each
candidate is simulated by CUWM, and the agent selects the action whose
predicted outcome best fits the goal. The selected action is compared with the
GUI-360 ground truth on 339 single-step samples. This agent changes that
procedure as follows:

01. **Environment and evaluation.** The agent runs closed loop in OSWorld on an
    Ubuntu desktop with applications such as Chrome, LibreOffice, and GIMP. The
    selected action is executed, the next real screenshot starts the next step,
    and the OSWorld evaluator scores the final state. The paper uses recorded
    Microsoft Word, Excel, and PowerPoint transitions on Windows and scores one
    step by Function, Status, and Args Match against the ground-truth action.
02. **Action space.** Candidates are tool calls named after the `computer_use` actions of `qwen3vl.py`
    (`key`, `type`, `mouse_move`, `left_click`, `left_click_drag`,
    `right_click`, `middle_click`, `double_click`, `scroll`, `wait`, and
    `terminate`) with coordinates on a relative 1000x1000 grid by default. The
    paper uses `function`, `args`, and `status` with accessibility control labels,
    `select_text`, and `drag` with start and end coordinates. Task completion is
    the `terminate` action instead of `status = "FINISH"`.
03. **Observation.** The policy sees only screenshots. The paper's agent also
    receives an annotated screenshot with numeric markers and accessibility
    information with control labels and the application name.
04. **History.** Generation and selection receive the `K` most recent
    screenshots and executed actions. The paper conditions both steps only on the
    current screenshot and the task instruction, which corresponds to `K = 0`
    with the history-free prompt variants.
05. **Action description.** Each candidate carries a natural-language `action`
    sentence besides its JSON call. CUWM receives the JSON call as the action and
    the sentence as the GUI action description. The paper describes actions
    only through the structured JSON command.
06. **Number of candidates.** `N` is configurable with a default of 5, and the
    selection prompt is rendered for the number of valid candidates. The paper
    always uses five options.
07. **Policy model.** One served Qwen3-VL model (the 4B variant in this project)
    generates candidates, predicts CUWM transitions, and selects the action. The
    paper's agent backbones are Qwen3-VL-8B, GPT-4.1-mini, GPT-4o, and
    Gemini-2.0-Flash, and its textual transition model is a separately trained
    Qwen2.5-VL-7B.
08. **Selection input.** The selector sees each option's tool call and
    predicted screenshot, but not its description or `Δ_i(t)`. This matches the paper's
    image-only setting, which outperformed image plus text with CUWM for all four
    agents in Table 5. The transitions are only recorded in the returned trace.
09. **Goal in simulation.** The task instruction is not passed to CUWM, so
    `Δ_i(t)` depends only on `s(t)` and `a_i(t)`. Appendix A.4.1 states that the
    instruction is passed to the textual model, although the published
    transition prompt has no slot for it.
10. **Prompt placement.** The adapted prompts are system prompts, and the
    screenshots are user message content. The paper's listings place the
    images inline at bracketed markers in a single prompt.
11. **Execution.** Only the selected tool call is converted into PyAutoGUI
    code. Candidates and selections are validated as described in
    [Validation](#validation). Invalid candidates are discarded, and a step
    without a valid candidate or selection yields no action. The paper scores
    invalid output as zero.
12. **Simulated candidates.** Every validated candidate with a distinct tool
    call is simulated, including `wait` and `terminate`. Candidates repeating an
    earlier tool call are dropped before simulation; no other pruning is applied.
