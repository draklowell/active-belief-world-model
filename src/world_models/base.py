from typing import Protocol


class TextTransitionModel(Protocol):
    """Backend contract for predicting the immediate consequences of a GUI action."""

    def generate(self, screenshot: bytes, prompt: str, *, system_prompt: str) -> str:
        """Generate text from prompts prepared by the world model.

        Args:
            screenshot: Encoded image of the current GUI state.
            prompt: User instruction prepared by the world model.
            system_prompt: Domain instructions prepared by the world model.

        Returns:
            Natural-language description of the expected immediate UI changes.
        """
        ...


class ImageTransitionModel(Protocol):
    """Backend contract for realizing a textual transition as a predicted screenshot."""

    def generate(self, screenshot: bytes, prompt: str) -> bytes:
        """Edit the current screenshot using a prepared instruction.

        Args:
            screenshot: Encoded image of the current GUI state.
            prompt: Image-editing instruction prepared by the world model.

        Returns:
            PNG-encoded image of the predicted next GUI state.
        """
        ...
