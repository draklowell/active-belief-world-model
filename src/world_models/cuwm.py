from prompts import render
from world_models.base import ImageTransitionModel, TextTransitionModel
from world_models.types import CUWMPrediction

IMAGE_ATTACHMENT_MARKER = "[Current UI screenshot attached as image content]"
PAPER_PROMPT_SET = "cuwm"


class CUWM:
    """Stateless two-stage world model for hypothetical GUI action consequences.

    Args:
        text_model: Backend that predicts a textual transition from a screenshot and action.
        image_model: Backend that renders that transition using the original screenshot.
        prompt_set: Template directory under `prompts/templates` holding the transition and
            image-editing prompts, such as `cuwm` for the paper or `cuwm-qwen3vl` for desktops.
    """

    def __init__(
        self,
        text_model: TextTransitionModel,
        image_model: ImageTransitionModel,
        prompt_set: str = PAPER_PROMPT_SET,
    ) -> None:
        self._text_model = text_model
        self._image_model = image_model
        self._prompt_set = prompt_set

    def generate_text_desc(
        self,
        screenshot: bytes,
        action: str,
        *,
        gui_description: str,
        app_name: str | None = None,
    ) -> str:
        """Predict UI changes using the configured transition prompt and explicit action context.

        The paper prompt set requires `app_name`, and omitting it raises a Jinja2 undefined
        variable error. The `cuwm-qwen3vl` set identifies the application from the screenshot.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Complete structured candidate GUI action represented as text.
            gui_description: Description of the action and its arguments.
            app_name: Application corresponding to the current screenshot, if the prompt uses it.

        Returns:
            Natural-language description of the expected UI transition.
        """
        context = {"action": action, "gui_description": gui_description}
        if app_name is not None:
            context["app_name"] = app_name
        for name, value in context.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"CUWM requires a nonempty {name}.")
        prompt = render(
            f"{self._prompt_set}/textual_state_transition.j2",
            image=IMAGE_ATTACHMENT_MARKER,
            **context,
        )
        return self._text_model.generate(screenshot, prompt, system_prompt="")

    def generate_image(self, screenshot: bytes, text_description: str) -> bytes:
        """Realize a predicted textual transition using the original screenshot.

        Args:
            screenshot: Encoded image of the current GUI state.
            text_description: Predicted UI changes to render.

        Returns:
            PNG-encoded image of the predicted next GUI state.
        """
        prompt = render(
            f"{self._prompt_set}/image_preservation.j2", text_description=text_description
        )
        return self._image_model.generate(screenshot, prompt)

    def predict(
        self,
        screenshot: bytes,
        action: str,
        *,
        gui_description: str,
        app_name: str | None = None,
    ) -> CUWMPrediction:
        """Predict a textual transition and then its visual realization.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Complete structured candidate GUI action represented as text.
            gui_description: Description of the action and its arguments.
            app_name: Application corresponding to the current screenshot, if the prompt uses it.

        Returns:
            Transition description and PNG-encoded predicted screenshot.
        """
        text_description = self.generate_text_desc(
            screenshot, action, app_name=app_name, gui_description=gui_description
        )
        predicted_screenshot = self.generate_image(screenshot, text_description)
        return CUWMPrediction(text_description, predicted_screenshot)
