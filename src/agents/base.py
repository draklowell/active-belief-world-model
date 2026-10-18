from abc import ABC, abstractmethod


class Agent(ABC):
    """Base interface for screenshot agents using PyAutoGUI actions.

    Attributes:
        config: Agent configuration loaded from JSON.

    Args:
        config: Agent configuration, including model and generation settings.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def reset(self) -> None:
        """Reset task history after the environment has been reset."""

    @abstractmethod
    def predict(self, instruction: str, observation: dict) -> tuple[str, list[str]]:
        """Generate a response and actions for the current screenshot.

        Args:
            instruction: Task instruction to follow.
            observation: Environment observation containing screenshot bytes.

        Returns:
            Response text and ordered PyAutoGUI commands or WAIT, FAIL, and DONE actions.
        """
