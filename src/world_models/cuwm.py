from world_models.base import ImageTransitionModel, TextTransitionModel
from world_models.types import CUWMPrediction

IMAGE_ATTACHMENT_MARKER = "[Current UI screenshot attached as image content]"


class CUWM:
    """Stateless two-stage world model for hypothetical GUI action consequences.

    Args:
        text_model: Backend that predicts a textual transition from a screenshot and action.
        image_model: Backend that renders that transition using the original screenshot.
    """

    def __init__(
        self,
        text_model: TextTransitionModel,
        image_model: ImageTransitionModel,
    ) -> None:
        self._text_model = text_model
        self._image_model = image_model

    def generate_text_desc(
        self, screenshot: bytes, action: str, *, app_name: str, gui_description: str
    ) -> str:
        """Predict UI changes using the complete paper prompt and explicit action context.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Complete structured candidate GUI action represented as text.
            app_name: Application corresponding to the current screenshot.
            gui_description: Function-level description of the action and its arguments.

        Returns:
            Natural-language description of the expected UI transition.
        """
        from prompts import render

        context = {"action": action, "app_name": app_name, "gui_description": gui_description}
        for name, value in context.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"CUWM requires a nonempty {name}.")
        prompt = render(
            "cuwm/textual_state_transition.j2", image=IMAGE_ATTACHMENT_MARKER, **context
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
        from prompts import render

        prompt = render("cuwm/image_preservation.j2", text_description=text_description)
        return self._image_model.generate(screenshot, prompt)

    def predict(
        self, screenshot: bytes, action: str, *, app_name: str, gui_description: str
    ) -> CUWMPrediction:
        """Predict a textual transition and then its visual realization.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Complete structured candidate GUI action represented as text.
            app_name: Application corresponding to the current screenshot.
            gui_description: Function-level description of the action and its arguments.

        Returns:
            Transition description and PNG-encoded predicted screenshot.
        """
        text_description = self.generate_text_desc(
            screenshot, action, app_name=app_name, gui_description=gui_description
        )
        predicted_screenshot = self.generate_image(screenshot, text_description)
        return CUWMPrediction(text_description, predicted_screenshot)
