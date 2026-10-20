from dataclasses import dataclass


@dataclass(frozen=True)
class CUWMPrediction:
    """Textual and visual predictions for one hypothetical GUI transition.

    Attributes:
        text_description: Predicted immediate UI changes.
        predicted_screenshot: PNG-encoded image of the predicted next GUI state.
    """

    text_description: str
    predicted_screenshot: bytes
