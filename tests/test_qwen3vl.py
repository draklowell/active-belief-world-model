from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agents.qwen3vl import (
    Qwen3VLAgent,
    ceil_by_factor,
    floor_by_factor,
    process_image,
    round_by_factor,
    smart_resize,
)


class TestImageHelpers:
    def test_round_by_factor(self) -> None:
        assert round_by_factor(27.0, 28) == 28
        assert round_by_factor(15.0, 28) == 28
        assert round_by_factor(13.0, 28) == 0

    def test_ceil_by_factor(self) -> None:
        assert ceil_by_factor(29.0, 28) == 56
        assert ceil_by_factor(28.0, 28) == 28

    def test_floor_by_factor(self) -> None:
        assert floor_by_factor(29.0, 28) == 28
        assert floor_by_factor(27.0, 28) == 0

    def test_smart_resize_validation(self) -> None:
        with pytest.raises(ValueError, match="must be larger than factor"):
            smart_resize(1, 100)
        with pytest.raises(ValueError, match="absolute aspect ratio"):
            smart_resize(10, 3000)

    def test_smart_resize_within_bounds(self) -> None:
        h, w = smart_resize(1080, 1920, factor=32)
        assert h % 32 == 0
        assert w % 32 == 0

    def test_process_image(self) -> None:
        image = Image.new("RGB", (1920, 1080), color="blue")
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        encoded = process_image(buf.getvalue())
        assert isinstance(encoded, str)
        assert len(encoded) > 0


class TestQwen3VLAgent:
    @pytest.fixture
    def agent(self) -> Qwen3VLAgent:
        config = {
            "model": "qwen3-vl-test",
            "temperature": 0.0,
            "top_p": 0.9,
            "coord": "relative",
        }
        return Qwen3VLAgent(config)

    def test_init_defaults(self, agent: Qwen3VLAgent) -> None:
        assert agent.action_space == "pyautogui"
        assert agent.observation_type == "screenshot"
        assert agent.model == "qwen3-vl-test"
        assert agent.coordinate_type == "relative"
        assert agent.actions == []
        assert agent.responses == []
        assert agent.screenshots == []

    def test_reset_clears_history(self, agent: Qwen3VLAgent) -> None:
        agent.actions.append("click")
        agent.responses.append("response")
        agent.screenshots.append("b64")
        agent.reset()
        assert agent.actions == []
        assert agent.responses == []
        assert agent.screenshots == []

    def test_parse_response_click_relative(self, agent: Qwen3VLAgent) -> None:
        response = (
            "Action: Click the start button\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [500, 500]}}\n'
            "</tool_call>"
        )
        instruction, actions = agent.parse_response(
            response,
            original_width=1920,
            original_height=1080,
            processed_width=1280,
            processed_height=720,
        )
        assert instruction == "Action: Click the start button"
        expected_x = int(500 * 1920 / 999)
        expected_y = int(500 * 1080 / 999)
        assert actions == [f"pyautogui.click({expected_x}, {expected_y})"]

    def test_parse_response_click_absolute(self) -> None:
        agent = Qwen3VLAgent({"coordinate_type": "absolute"})
        response = (
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "right_click", '
            '"coordinate": [640, 360]}}\n'
            "</tool_call>"
        )
        instruction, actions = agent.parse_response(
            response,
            original_width=1920,
            original_height=1080,
            processed_width=1280,
            processed_height=720,
        )
        assert instruction == ""
        assert actions == ["pyautogui.rightClick(960, 540)"]

    def test_parse_response_type_and_key(self, agent: Qwen3VLAgent) -> None:
        response = (
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "hello\\nworld"}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "c"]}}\n'
            "</tool_call>"
        )
        _, actions = agent.parse_response(response, 1920, 1080, 1280, 720)
        assert "pyautogui.typewrite('hello', interval=0.03)" in actions
        assert "pyautogui.press('enter')" in actions
        assert "pyautogui.typewrite('world', interval=0.03)" in actions
        assert "pyautogui.hotkey('ctrl', 'c')" in actions

    def test_parse_response_wait_and_terminate(self, agent: Qwen3VLAgent) -> None:
        response = (
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "wait", "time": 2}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
            "</tool_call>"
        )
        _, actions = agent.parse_response(response, 1920, 1080, 1280, 720)
        assert actions == ["WAIT", "DONE"]

    def test_parse_response_action_tag_extracted(self, agent: Qwen3VLAgent) -> None:
        response = (
            "Action: Open the file manager\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "double_click", '
            '"coordinate": [100, 200]}}\n'
            "</tool_call>"
        )
        instruction, actions = agent.parse_response(response, 1920, 1080, 1280, 720)
        assert instruction == "Action: Open the file manager"
        assert len(actions) == 1
        assert "doubleClick" in actions[0]

    def test_predict_flow(self, agent: Qwen3VLAgent) -> None:
        image = Image.new("RGB", (1000, 1000), color="white")
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        obs = {"screenshot": buf.getvalue()}

        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(
                message=MagicMock(
                    content=(
                        "Action: Click at center\n"
                        "<tool_call>\n"
                        '{"name": "computer_use", "arguments": {"action": "left_click", '
                        '"coordinate": [500, 500]}}\n'
                        "</tool_call>"
                    )
                )
            )
        ]

        with patch.object(
            agent._client.chat.completions, "create", return_value=mock_completion
        ) as mock_create:
            response, actions = agent.predict("Click the center", obs)
            assert "Action: Click at center" in response
            assert len(actions) == 1
            assert actions[0].startswith("pyautogui.click(")
            mock_create.assert_called_once()

    def test_init_config_env_fallback(self) -> None:
        """Verify environment variable fallback for client credentials via decouple config."""
        with patch("agents.qwen3vl.config") as mock_config:
            mock_config.side_effect = lambda key, default="": (
                "https://custom.endpoint/v1" if key == "OPENAI_BASE_URL" else "custom-key"
            )
            agent = Qwen3VLAgent({})
            assert str(agent._client.base_url) == "https://custom.endpoint/v1/"
            assert agent._client.api_key == "custom-key"
