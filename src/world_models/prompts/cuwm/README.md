# CUWM changes for the Qwen3-VL CUWM agent

This report describes the changes to the CUWM world-model pipeline in
[`cuwm.py`](../../cuwm.py) that support the
[Qwen3-VL CUWM agent](../../../agents/qwen3vl_cuwm/README.md), and how the
resulting pipeline differs from
[Guan et al.](https://arxiv.org/abs/2602.17365). The agent calls
`CUWM.predict` once per candidate action. The general API, configuration, and
backend contracts are documented in the [world model README](../../README.md).

## Changes to the pipeline

### Interface

`generate_text_desc` and `predict` no longer accept `app_name`:

```python
cuwm.generate_text_desc(screenshot, action, *, gui_description)
cuwm.predict(screenshot, action, *, gui_description)
```

- `action` is the structured candidate action. The agent passes the
  `computer_use` tool call JSON from the Qwen3-VL action space, for example
  `{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [512, 64]}}`.
- `gui_description` is the natural-language description of that action. The
  agent passes the candidate's one-sentence `Action:` description, for example
  `Click the column H header.`. It previously had to be a function-level
  description of the action and its arguments.
- The application is no longer supplied by the caller. The textual model
  identifies it from the screenshot.

`render_textual_state_transition(action, *, gui_description)` in
[`__init__.py`](__init__.py) drops `app_name` accordingly and renders the
desktop prompt below instead of the paper listing. Blank values and unknown
placeholders are still rejected before inference. `scripts/run_cuwm.py` drops
the `--app-name` option.

### Textual state transition prompt

The runtime prompt is
[`textual_state_transition.md`](textual_state_transition.md), loaded by
`load_desktop_transition_prompt`. The paper listing in
[`paper/textual_state_transition.md`](paper/textual_state_transition.md) is
unchanged and still loadable with `load_paper_prompt`. The adapted prompt keeps
the paper's task statement, inputs, `{image}`, `{action}`, and
`{gui_description}` slots, three-part response structure, top-down region
order, change phrasing, quoting rule, and single-paragraph output. It changes
the following text:

| Paper listing (Appendix A.5.2)                                                      | Desktop prompt                                                                       |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| World Model assistant for Office applications such as `{app_name}`                  | World Model assistant for desktop applications                                       |
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
| Terminology and layout conventions consistent with `{app_name}`                     | Conventions consistent with the application shown in the current screenshot          |

The fence parser shared by the paper prompts and the desktop prompt moved to
`world_models.prompts.remove_presentation_fence`, which the agent also uses for
its own prompts.

### Unchanged parts

The two-stage order, the empty system message of the textual request, the
image-editing instruction, the backends, `config/cuwm.toml`, and the retry
policy are unchanged. Each call remains stateless; CUWM keeps no trajectory
history and simulates one candidate per call.

## Differences from the CUWM paper

The paper's CUWM (Section 3 and Appendix A.2) predicts `Δ_t = f_text(s_t, a_t)`
with a trained Qwen2.5-VL and renders `ŝ_t+1 = f_image(s_t, Δ_t)` with a trained
Qwen-Image-Edit-2509 for Microsoft Office. The pipeline used by the agent keeps
this factorization and changes the following:

1. **Application context.** The textual stage no longer receives an
   application name. The paper's prompt names the Office application twice, and
   its agent observations include the application name from accessibility
   information. The desktop prompt asks the model to identify the application
   from the screenshot. This change was made for the agent.
2. **Domain.** The prompt targets Ubuntu desktop applications in OSWorld
   instead of Word, Excel, and PowerPoint, with the region and example changes
   listed above. This change was made for the agent.
3. **Action representation.** The action is a Qwen3-VL `computer_use` tool call
   with coordinates on a relative 1000x1000 grid by default, and the GUI
   description is a one-sentence imperative produced by the policy. The paper's
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
7. **Image-editing instruction.** The image stage receives `Δ_t` followed by a
   project-specific instruction to preserve unrelated layout, text, icons, and
   colors. The paper conditions its fine-tuned editor on the screenshot and the
   transition text and does not publish an editing instruction.
8. **Screenshot geometry.** The agent passes its processed screenshot, resized to
   multiples of 32 pixels, and the image client pads it to multiples of 16
   before editing and removes the padding afterwards. The paper trains
   Qwen-Image-Edit with at most 1,048,576 pixels per image.
