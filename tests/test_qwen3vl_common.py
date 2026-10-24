from io import BytesIO

import pytest
from PIL import Image

from agents.qwen3vl_common import (
    build_computer_use_tool,
    ceil_by_factor,
    convert_tool_call_to_pyautogui,
    floor_by_factor,
    prepare_screenshot,
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
        buf = BytesIO()
        image.save(buf, format="PNG")
        encoded = process_image(buf.getvalue())
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_prepare_screenshot_reports_both_sizes(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (1920, 1080), color="blue").save(buffer, format="PNG")
        encoded, original_size, processed_size = prepare_screenshot(buffer.getvalue())
        assert encoded == process_image(buffer.getvalue())
        assert original_size == (1920, 1080)
        assert processed_size == (1920, 1088)


class TestComputerUseTool:
    def test_resolution_follows_coordinate_type(self) -> None:
        relative = build_computer_use_tool("relative", 1920, 1088)
        absolute = build_computer_use_tool("absolute", 1920, 1088)
        assert "resolution is 1000x1000" in relative["function"]["description"]
        assert "resolution is 1920x1088" in absolute["function"]["description"]
        assert relative["function"]["name"] == "computer_use"
        assert relative["function"]["parameters"]["properties"]["action"]["enum"] == [
            "key",
            "type",
            "mouse_move",
            "left_click",
            "left_click_drag",
            "right_click",
            "middle_click",
            "double_click",
            "scroll",
            "wait",
            "terminate",
        ]

    @pytest.mark.parametrize(
        "coordinate_type, expected",
        [("relative", ["pyautogui.click(960, 540)"]), ("absolute", ["pyautogui.click(749, 396)"])],
    )
    def test_click_coordinates_are_mapped_to_native_pixels(
        self, coordinate_type: str, expected: list[str]
    ) -> None:
        arguments = {"action": "left_click", "coordinate": [499.5, 499.5]}
        assert (
            convert_tool_call_to_pyautogui(arguments, coordinate_type, (1920, 1080), (1280, 1360))
            == expected
        )

    def test_unknown_action_has_no_commands(self) -> None:
        assert convert_tool_call_to_pyautogui({"action": "fly"}, "relative", (1, 1), (1, 1)) == []
