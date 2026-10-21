from world_models.base import ImageTransitionModel, TextTransitionModel
from world_models.types import CUWMPrediction

TRANSITION_PROMPT = (
    "You predict immediate GUI state transitions for a computer-using world model. "
    "Given the current screenshot and one hypothetical candidate action, describe the "
    "visible changes that would most likely follow that action. Identify the affected "
    "UI elements and their locations, including selection, focus, content, menus, or "
    "dialogs when relevant. Use the screenshot as the baseline and mention relevant "
    "unchanged elements concisely. If no visible change is expected, say so. "
    "Do not execute the action or claim it has happened. Do not choose another action, "
    "write a task plan, or invent hidden state or details that cannot reasonably be inferred. "
    "Return only a concise natural-language transition description for an image editor."
)


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

    def generate_text_desc(self, screenshot: bytes, action: str) -> str:
        """Predict the immediate UI changes for a candidate action.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Hypothetical action to interpret without executing it.

        Returns:
            Natural-language description of the expected UI transition.
        """
        prompt = (
            f"Hypothetical candidate action:\n{action}\n\n"
            "Predict only the immediate resulting UI transition."
        )
        return self._text_model.generate(screenshot, prompt, system_prompt=TRANSITION_PROMPT)

    def generate_image(self, screenshot: bytes, text_description: str) -> bytes:
        """Realize a predicted textual transition using the original screenshot.

        Args:
            screenshot: Encoded image of the current GUI state.
            text_description: Predicted UI changes to render.

        Returns:
            PNG-encoded image of the predicted next GUI state.
        """
        prompt = (
            f"{text_description}\n\n"
            "Preserve the layout, text, icons, colors, and all other GUI elements "
            "unrelated to this transition."
        )
        return self._image_model.generate(screenshot, prompt)

    def predict(self, screenshot: bytes, action: str) -> CUWMPrediction:
        """Predict a textual transition and then its visual realization.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Hypothetical action to interpret without executing it.

        Returns:
            Transition description and PNG-encoded predicted screenshot.
        """
        text_description = self.generate_text_desc(screenshot, action)
        predicted_screenshot = self.generate_image(screenshot, text_description)
        return CUWMPrediction(text_description, predicted_screenshot)
