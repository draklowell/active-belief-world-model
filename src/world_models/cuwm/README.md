# CUWM changes for the Qwen3-VL CUWM agent

This report describes the changes to the CUWM world-model pipeline in
[`cuwm.py`](../cuwm.py) that support the
[Qwen3-VL CUWM agent](../../agents/qwen3vl_cuwm/README.md), and how the
resulting pipeline differs from
[Guan et al.](https://arxiv.org/abs/2602.17365). The agent calls
`CUWM.predict` once per candidate action. The general API, configuration, and
backend contracts are documented in the [world model README](../README.md).

## Changes to the pipeline

### Prompt sets

CUWM renders its two prompts from a selectable template directory under
[`src/prompts/templates/`](../../prompts/templates/), its prompt set:

| Prompt set     | Transition prompt                                             | Image-editing prompt                        | `app_name`                        |
| -------------- | ------------------------------------------------------------- | ------------------------------------------- | --------------------------------- |
| `cuwm`         | Complete Appendix A.5.2 paper prompt                          | Project preservation instruction            | Required                          |
| `cuwm-qwen3vl` | Desktop adaptation without an application name (listed below) | Includes `cuwm/image_preservation.j2` as is | Not used, and ignored if supplied |

`CUWM(text_model, image_model, prompt_set="cuwm")` selects the set, and
`build_cuwm` reads it from the optional root `prompt_set` key of the TOML
configuration. Omitting the key keeps the paper prompts, so `config/cuwm.toml`
and existing callers behave as before. The paper templates in
[`cuwm/`](../../prompts/templates/cuwm/) are unchanged.

The agent loads `config/cuwm.toml` and replaces its `prompt_set` with
`cuwm-qwen3vl` before building CUWM, because the agent supplies no application
name. Its own prompts live in the same
[`cuwm-qwen3vl/`](../../prompts/templates/cuwm-qwen3vl/) directory.

### Interface

`app_name` is now an optional keyword argument:

```python
cuwm.generate_text_desc(screenshot, action, *, gui_description, app_name=None)
cuwm.predict(screenshot, action, *, gui_description, app_name=None)
```

- `action` is the structured candidate action as text. The agent passes the
  tool call JSON from the Qwen3-VL action space, named after the action, for example
  `{"name": "left_click", "arguments": {"coordinate": [512, 64]}}`.
- `gui_description` is the description of that action. The agent passes the
  candidate's one-sentence `Action:` description, for example
  `Click the column H header.`.
- `app_name` is passed to the template only when supplied, and a supplied value
  must be nonblank. With the paper set, omitting it raises a Jinja2
  `UndefinedError` before any inference request, so the paper prompt is never
  rendered without its application. With `cuwm-qwen3vl`, the textual model
  identifies the application from the screenshot.

`scripts/run_cuwm.py` keeps `--app-name` but makes it optional; it is required
in practice only by the paper prompt set.

### Textual state transition prompt

[`cuwm-qwen3vl/textual_state_transition.j2`](../../prompts/templates/cuwm-qwen3vl/textual_state_transition.j2)
keeps the paper's task statement, inputs, `{{ image }}`, `{{ action }}`, and
`{{ gui_description }}` slots, with the action inside a `json` code fence, three-part response structure, top-down region
order, change phrasing, quoting rule, and single-paragraph output. It changes
the following text:

| Paper listing (Appendix A.5.2)                                                      | Desktop prompt                                                                       |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| World Model assistant for Office applications such as `{{ app_name }}`              | World Model assistant for desktop applications                                       |
| A screenshot of the current Office UI                                               | A screenshot of the current desktop UI                                               |
| A function-level description of the GUI action and its arguments                    | A natural-language description of the GUI action                                     |
| Start by stating which Office application this is ("This is Microsoft PowerPoint.") | Start by stating which application is in focus ("This is Google Chrome.")            |
| Unchanged example "Ribbon unchanged"                                                | Unchanged example "Toolbar unchanged"                                                |
| No desktop shell region                                                             | Desktop Panel (top bar, dock, launcher, open windows)                                |
| Title Bar (document name, window state, changes if any)                             | Title Bar (window or document name, window state, changes if any)                    |
| Ribbon (active tab, visible groups, detailed controls and icons)                    | Menu Bar / Toolbar (active menu or tab, visible groups, detailed controls and icons) |
| Navigation Area (slide thumbnails, focus changes)                                   | Navigation Area (tabs, address bar, thumbnails, focus changes)                       |
| Dropdown / Popout                                                                   | Dropdown / Popout / Dialog                                                           |
| Examples with the "Insert" tab and a "Design Ideas" panel                           | Examples with the "Insert" menu and a "Layers" panel                                 |
| Quoted text example "Home", "File", "New Slide"                                     | Quoted text example "File", "Edit", "View"                                           |
| Terminology and layout conventions consistent with `{{ app_name }}`                 | Conventions consistent with the application shown in the current screenshot          |

### Unchanged parts

The two-stage order, the empty system message of the textual request, the
image-editing instruction, the backends, `config/cuwm.toml`, and the retry
policy are unchanged. Each call remains stateless; CUWM keeps no trajectory
history and simulates one candidate per call.

## Differences from the CUWM paper

The paper's CUWM (Section 3 and Appendix A.2) predicts `Δ_t = f_text(s_t, a_t)`
with a trained Qwen2.5-VL and renders `ŝ_t+1 = f_image(s_t, Δ_t)` with a trained
Qwen-Image-Edit-2509 for Microsoft Office. The pipeline used by the agent, CUWM
with the `cuwm-qwen3vl` prompt set, keeps this factorization and changes the
following. Items 1 to 3 apply only to that prompt set; items 4 to 8 also apply to
the default paper prompt set.

1. **Application context.** With the `cuwm-qwen3vl` prompt set, the textual
   stage receives no application name. The paper's prompt names the Office
   application twice, and its agent observations include the application name
   from accessibility information. The desktop prompt asks the model to identify
   the application from the screenshot. This change was made for the agent.
2. **Domain.** The prompt targets Ubuntu desktop applications in OSWorld
   instead of Word, Excel, and PowerPoint, with the region and example changes
   listed above. This change was made for the agent.
3. **Action representation.** The action is a Qwen3-VL action tool call
   (`{"name": <action>, "arguments": {...}}`) with coordinates on a relative 1000x1000 grid by default, and the GUI
   description is a one-sentence imperative produced by the policy instead of a
   function-level description. The paper's
   actions are GUI-360 commands (`function`, `args`, `status`) that may target
   accessibility control labels, paired with a function-level description. The
   coordinate convention is not explained to CUWM; the description grounds the
   target. This change was made for the agent.
4. **Task instruction.** CUWM does not receive the task instruction. Appendix
   A.4.1 states that the instruction is passed to the textual model during
   agent evaluation, although the published transition prompt has no slot for
   it.
5. **Textual model.** The textual stage uses the untrained Qwen3-VL model that
   also serves as the agent policy, configured in `config/cuwm.toml`. The paper
   fine-tunes Qwen2.5-VL-7B-Instruct with LoRA on GPT-5 annotated GUI-360
   transitions and refines it with GRPO using an LLM judge and a length
   penalty. No training is performed here.
6. **Visual model.** The visual stage uses MageFlowEdit-Turbo served by
   stable-diffusion.cpp with 4 steps, guidance 1.0, and seed 2026. The paper
   fine-tunes the DiT of Qwen-Image-Edit-2509 with LoRA and an MSE
   reconstruction loss.
7. **Image-editing instruction.** The image stage receives `Δ_t` followed by the
   project-specific instruction in `cuwm/image_preservation.j2` to preserve
   unrelated layout, text, icons, and colors, in both prompt sets. The paper conditions its fine-tuned editor on the screenshot and the
   transition text and does not publish an editing instruction.
8. **Screenshot geometry.** The agent passes its processed screenshot, resized to
   multiples of 32 pixels, and the image client pads it to multiples of 16
   before editing and removes the padding afterwards. The paper trains
   Qwen-Image-Edit with at most 1,048,576 pixels per image.
