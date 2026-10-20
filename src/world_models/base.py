from typing import Protocol


class TextTransitionModel(Protocol):
    """Backend contract for predicting the immediate consequences of a GUI action."""

    def generate(self, screenshot: bytes, action: str) -> str:
        """Describe the predicted transition without executing the action.

        Args:
            screenshot: Encoded image of the current GUI state.
            action: Candidate action in its original textual representation.

        Returns:
            Natural-language description of the expected immediate UI changes.
        """
        ...


class ImageTransitionModel(Protocol):
    """Backend contract for realizing a textual transition as a predicted screenshot."""

    def generate(self, screenshot: bytes, text_description: str) -> bytes:
        """Edit the current screenshot according to a predicted transition.

        Args:
            screenshot: Encoded image of the current GUI state.
            text_description: Predicted UI changes to render.

        Returns:
            PNG-encoded image of the predicted next GUI state.
        """
        ...
