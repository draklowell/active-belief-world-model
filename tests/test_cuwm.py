import base64
import json
from email import policy
from email.parser import BytesParser
from io import BytesIO

import httpx2
import pytest
from PIL import Image

from world_models import CUWM, CUWMPrediction
from world_models.backends.mage_flow import MageFlowImageModel
from world_models.backends.qwen_text import QwenTextModel


def test_predict_passes_original_screenshot_through_both_stages() -> None:
    """Keep the original screenshot and ordered transition between both model stages."""
    calls: list[tuple[str, bytes, str]] = []
    system_prompts: list[str] = []

    class TextModel:
        """Lightweight textual transition substitute."""

        def generate(self, screenshot: bytes, prompt: str, *, system_prompt: str) -> str:
            """Record the textual input and return its predicted transition."""
            calls.append(("text", screenshot, prompt))
            system_prompts.append(system_prompt)
            return "Column H would become selected."

    class ImageModel:
        """Lightweight visual transition substitute."""

        def generate(self, screenshot: bytes, prompt: str) -> bytes:
            """Record the visual input and return its predicted screenshot."""
            calls.append(("image", screenshot, prompt))
            return b"predicted screenshot"

    cuwm = CUWM(text_model=TextModel(), image_model=ImageModel())

    prediction = cuwm.predict(screenshot=b"original screenshot", action="Select column H")

    assert [call[0] for call in calls] == ["text", "image"]
    assert [call[1] for call in calls] == [b"original screenshot", b"original screenshot"]
    assert "Select column H" in calls[0][2]
    assert system_prompts[0]
    assert "Column H would become selected." in calls[1][2]
    assert "Preserve" in calls[1][2]
    assert isinstance(prediction, CUWMPrediction)
    assert prediction.text_description == "Column H would become selected."
    assert prediction.predicted_screenshot == b"predicted screenshot"


@pytest.fixture
def screenshot() -> bytes:
    """Encode a small screenshot whose dimensions require Mage alignment."""
    buffer = BytesIO()
    Image.new("RGB", (31, 17), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_qwen_sends_caller_prompts_to_the_text_endpoint(
    monkeypatch: pytest.MonkeyPatch, screenshot: bytes
) -> None:
    """Forward supplied prompts and the full screenshot through the real SDK request."""

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        assert request.url == "https://text.test/v1/chat/completions"
        body = json.loads(request.read())
        assert body["model"] == "text-model"
        assert body["messages"][0] == {"role": "system", "content": "Custom domain instruction"}
        content = body["messages"][1]["content"]
        assert content[1] == {"type": "text", "text": "Custom candidate action"}
        image_bytes = base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
        with Image.open(BytesIO(image_bytes)) as image:
            assert image.size == (31, 17)
        return httpx2.Response(
            200,
            request=request,
            json={
                "id": "text-response",
                "object": "chat.completion",
                "created": 0,
                "model": "text-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Predicted transition"},
                    }
                ],
            },
        )

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = QwenTextModel(
        model="text-model", base_url="https://text.test/v1", api_key_env="CUWM_TEST_API_KEY"
    )
    description = model.generate(
        screenshot, "Custom candidate action", system_prompt="Custom domain instruction"
    )
    assert description == "Predicted transition"


def test_mage_edits_remotely_and_removes_only_alignment_padding(
    monkeypatch: pytest.MonkeyPatch, screenshot: bytes
) -> None:
    """Send an aligned screenshot to the image API and preserve the source dimensions."""
    prediction_buffer = BytesIO()
    Image.new("RGB", (32, 32), color="green").save(prediction_buffer, format="PNG")
    encoded_prediction = base64.b64encode(prediction_buffer.getvalue()).decode("ascii")

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        assert request.url == "https://image.test/v1/images/edits"
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read()
        )
        fields = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()
        }
        assert fields["model"] == b"mage-edit"
        assert fields["prompt"] == b"Caller-provided edit instruction"
        assert fields["size"] == b"32x32"
        assert fields["num_inference_steps"] == b"8"
        assert fields["guidance_scale"] == b"2.0"
        assert fields["seed"] == b"2026"
        image_bytes = fields["image"]
        assert isinstance(image_bytes, bytes)
        with Image.open(BytesIO(image_bytes)) as image:
            assert image.size == (32, 32)
            assert image.getpixel((30, 16)) == (0, 0, 255)
            assert image.getpixel((31, 31)) == (255, 255, 255)
        return httpx2.Response(
            200, request=request, json={"created": 0, "data": [{"b64_json": encoded_prediction}]}
        )

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = MageFlowImageModel(
        model="mage-edit",
        base_url="https://image.test/v1",
        api_key_env="CUWM_TEST_API_KEY",
        steps=8,
        cfg=2.0,
    )
    result = model.generate(screenshot, "Caller-provided edit instruction")
    with Image.open(BytesIO(result)) as prediction:
        assert prediction.size == (31, 17)
        assert prediction.format == "PNG"
        assert prediction.getpixel((30, 16)) == (0, 128, 0)


def test_mage_rejects_an_api_response_without_image_data(
    monkeypatch: pytest.MonkeyPatch, screenshot: bytes
) -> None:
    """Reject missing image data instead of returning the unchanged screenshot."""

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        return httpx2.Response(200, request=request, json={"created": 0, "data": []})

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = MageFlowImageModel(base_url="https://image.test/v1", api_key_env="CUWM_TEST_API_KEY")
    with pytest.raises(ValueError, match="no usable image"):
        model.generate(screenshot, "Edit the screenshot")
