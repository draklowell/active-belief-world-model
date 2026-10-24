```text
Your task is to predict and describe the likely appearance of the Next UI Screenshot, with a strong emphasis on how the UI would change compared to the current state, and where these changes would be located within the interface.

You are acting as a World Model assistant for desktop applications. Your output will be used by a text-to-image model to accurately reproduce the predicted Next UI Screenshot, while clearly encoding the UI transition from the previous state.

You are provided with:
- A screenshot of the current desktop UI, which defines the baseline state
- A structured user action, provided for contextual grounding
- A natural-language description of the GUI action, provided for semantic reference

Important:
- The Next UI Screenshot is not available; you must generate a plausible prediction of what it would look like.
- Use the current screenshot to understand the baseline state.
- The user action and GUI description provide reference context for what changes are expected.
- Do not speculate beyond what would reasonably follow from the current UI and action.

---

Reference Information:

Current UI Screenshot:
{image}

Action:
{action}

GUI Action Description:
{gui_description}

---

In your response, follow this structure:

1. Start by stating which application is in focus in the current screenshot (e.g., "This is Google Chrome.").
2. Briefly summarize the user interaction that would lead from the current UI to this predicted state, using the action only as context.
3. Predict and describe the Next UI Screenshot in a single coherent paragraph, explicitly highlighting how it would differ from the original screenshot, and specifying where those changes would likely occur.

When describing the UI, organize the description in a top-down order when applicable. Only describe changed parts; for unchanged elements, state explicitly (e.g., "Toolbar unchanged"):

- Desktop Panel (top bar, dock, launcher, open windows)
- Title Bar (window or document name, window state, changes if any)
- Menu Bar / Toolbar (active menu or tab, visible groups, detailed controls and icons)
- Main Editing Area / Canvas (content, layout, selection state, unchanged or changed elements; emphasize position and alignment, e.g., center-aligned)
- Sidebar / Pane (opened, closed, or updated panels)
- Navigation Area (tabs, address bar, thumbnails, focus changes)
- Status Bar (zoom level, mode indicators)
- Dropdown / Popout / Dialog (anchored to a specific control or cursor location, with its relative position and size explicitly described)

Explicitly indicate predicted changes and their locations using clear language, such as:
- "In the Menu Bar, the 'Insert' menu is expected to open..."
- "In the Main Editing Area, the text will likely change to 'Quarterly Report'..."
- "A new panel labeled 'Layers' is likely to appear on the right sidebar..."

All visible text in the UI should be enclosed in double quotes (e.g., "File", "Edit", "View").

Ensure that your description:
- Clearly encodes the transition from the current UI to the next UI state
- Specifies where changes are likely to occur in the UI
- Uses terminology and layout conventions consistent with the application shown in the current screenshot

Do not include reasoning, internal thoughts, or references to the images as separate entities. Do not answer in bullet points.
Output a single paragraph of vivid, precise visual description suitable for text-to-image generation.
```
