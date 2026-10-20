from world_models import CUWM, CUWMPrediction


def test_predict_passes_original_screenshot_through_both_stages() -> None:
    """Keep the original screenshot and ordered transition between both model stages."""
    calls: list[tuple[str, bytes, str]] = []

    class TextModel:
        """Lightweight textual transition substitute."""

        def generate(self, screenshot: bytes, action: str) -> str:
            """Record the textual input and return its predicted transition."""
            calls.append(("text", screenshot, action))
            return "Column H would become selected."

    class ImageModel:
        """Lightweight visual transition substitute."""

        def generate(self, screenshot: bytes, text_description: str) -> bytes:
            """Record the visual input and return its predicted screenshot."""
            calls.append(("image", screenshot, text_description))
            return b"predicted screenshot"

    cuwm = CUWM(text_model=TextModel(), image_model=ImageModel())

    prediction = cuwm.predict(screenshot=b"original screenshot", action="Select column H")

    assert calls == [
        ("text", b"original screenshot", "Select column H"),
        ("image", b"original screenshot", "Column H would become selected."),
    ]
    assert isinstance(prediction, CUWMPrediction)
    assert prediction.text_description == "Column H would become selected."
    assert prediction.predicted_screenshot == b"predicted screenshot"
