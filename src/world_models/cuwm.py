from world_models.base import ImageTransitionModel, TextTransitionModel
from world_models.types import CUWMPrediction


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
        return self._text_model.generate(screenshot, action)

    def generate_image(self, screenshot: bytes, text_description: str) -> bytes:
        """Realize a predicted textual transition using the original screenshot.

        Args:
            screenshot: Encoded image of the current GUI state.
            text_description: Predicted UI changes to render.

        Returns:
            PNG-encoded image of the predicted next GUI state.
        """
        return self._image_model.generate(screenshot, text_description)

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
