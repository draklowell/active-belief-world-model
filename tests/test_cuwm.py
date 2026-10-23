import base64
import binascii
import json
import sys
import time
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import httpx2
import openai
import pytest
from PIL import Image, UnidentifiedImageError

from scripts import run_cuwm
from world_models import CUWM, CUWMPrediction
from world_models.backends.openai_text import OpenAITextModel
from world_models.backends.stable_diffusion_image import StableDiffusionImageModel

CANDIDATE_ACTION = (
    '{"function": "click", "args": ' '{"control_label": 15, "coordinate": null, "button": "left"}}'
)
GUI_DESCRIPTION = (
    "Click the UI control identified by accessibility control_label 15 "
    "using the left mouse button."
)


def encode_png(size: tuple[int, int], color: str) -> bytes:
    """Encode a solid PNG fixture at explicitly chosen dimensions."""
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


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

    prediction = cuwm.predict(
        screenshot=b"original screenshot",
        action=CANDIDATE_ACTION,
        app_name="Microsoft Excel",
        gui_description=GUI_DESCRIPTION,
    )

    assert [call[0] for call in calls] == ["text", "image"]
    assert [call[1] for call in calls] == [b"original screenshot", b"original screenshot"]
    assert CANDIDATE_ACTION in calls[0][2]
    assert GUI_DESCRIPTION in calls[0][2]
    assert "Microsoft Excel" in calls[0][2]
    assert calls[0][2].startswith("Your task is to predict and describe")
    assert "Dropdown / Popout" in calls[0][2]
    assert system_prompts == [""]
    assert calls[1][2] == (
        "Column H would become selected.\n\n"
        "Preserve the layout, text, icons, colors, and all other GUI elements "
        "unrelated to this transition."
    )
    assert isinstance(prediction, CUWMPrediction)
    assert prediction.text_description == "Column H would become selected."
    assert prediction.predicted_screenshot == b"predicted screenshot"


@pytest.fixture
def screenshot() -> bytes:
    """Encode a small screenshot whose dimensions require configured alignment."""
    return encode_png((31, 17), "blue")


@pytest.fixture
def retry_pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays without sleeping during simulated API failures."""
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    return pauses


@pytest.mark.parametrize("served_model", ["text-model", "another-vlm"])
def test_text_model_preserves_caller_prompts_after_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: bytes,
    retry_pauses: list[float],
    served_model: str,
) -> None:
    """Recover after SDK retries while preserving supplied prompts and image dimensions."""
    retry_counts: list[str] = []

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        assert request.url == "https://text.test/v1/chat/completions"
        retry_counts.append(request.headers["x-stainless-retry-count"])
        if len(retry_counts) <= 3:
            return httpx2.Response(503, request=request, json={"error": {"message": "Busy"}})
        body = json.loads(request.read())
        assert body["model"] == served_model
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
    model = OpenAITextModel(
        model=served_model, base_url="https://text.test/v1", api_key_env="CUWM_TEST_API_KEY"
    )
    description = model.generate(
        screenshot, "Custom candidate action", system_prompt="Custom domain instruction"
    )
    assert description == "Predicted transition"
    assert retry_counts == ["0", "1", "2", "0"]
    assert len(retry_pauses) == 3


def test_stable_diffusion_edits_remotely_and_removes_only_alignment_padding(
    monkeypatch: pytest.MonkeyPatch, screenshot: bytes, retry_pauses: list[float]
) -> None:
    """Recover a remote image edit while preserving image alignment and source dimensions."""
    prediction_buffer = BytesIO()
    Image.new("RGB", (32, 32), color="green").save(prediction_buffer, format="PNG")
    encoded_prediction = base64.b64encode(prediction_buffer.getvalue()).decode("ascii")
    retry_counts: list[str] = []

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        assert request.url == "https://image.test/v1/images/edits"
        retry_counts.append(request.headers["x-stainless-retry-count"])
        if len(retry_counts) <= 3:
            return httpx2.Response(503, request=request, json={"error": {"message": "Busy"}})
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read()
        )
        fields = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()
        }
        assert fields["model"] == b"mage-edit"
        prompt = fields["prompt"]
        assert isinstance(prompt, bytes)
        instruction, extra_params = prompt.split(b"<sd_cpp_extra_args>")
        assert instruction == b"Caller-provided edit instruction"
        assert extra_params.endswith(b"</sd_cpp_extra_args>")
        assert json.loads(extra_params.removesuffix(b"</sd_cpp_extra_args>")) == {
            "sample_params": {
                "sample_steps": 8,
                "guidance": {"txt_cfg": 2.0, "img_cfg": 2.0},
            },
            "seed": 2026,
        }
        assert fields["size"] == b"32x32"
        assert fields["response_format"] == b"b64_json"
        assert fields["output_format"] == b"png"
        assert set(fields) == {
            "model",
            "image",
            "prompt",
            "size",
            "n",
            "response_format",
            "output_format",
        }
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
    model = StableDiffusionImageModel(
        model="mage-edit",
        base_url="https://image.test/v1",
        api_key_env="CUWM_TEST_API_KEY",
        padding_multiple=16,
        steps=8,
        cfg=2.0,
        seed=2026,
    )
    result = model.generate(screenshot, "Caller-provided edit instruction")
    assert retry_counts == ["0", "1", "2", "0"]
    assert len(retry_pauses) == 3
    with Image.open(BytesIO(result)) as prediction:
        assert prediction.size == (31, 17)
        assert prediction.format == "PNG"
        assert prediction.getpixel((30, 16)) == (0, 128, 0)


@pytest.mark.parametrize("response_data", [[], None, [{"url": "https://image.test/result.png"}]])
def test_image_model_rejects_an_api_response_without_inline_image_data(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: bytes,
    response_data: list[dict[str, str]] | None,
) -> None:
    """Reject missing image data instead of returning the unchanged screenshot."""

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        return httpx2.Response(200, request=request, json={"created": 0, "data": response_data})

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = StableDiffusionImageModel(
        model="image-model", base_url="https://image.test/v1", api_key_env="CUWM_TEST_API_KEY"
    )
    with pytest.raises(ValueError, match="no usable image"):
        model.generate(screenshot, "Edit the screenshot")


@pytest.mark.parametrize("status_code, expected_requests", [(503, 15), (400, 5)])
def test_image_model_stops_retrying_after_the_bounded_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: bytes,
    retry_pauses: list[float],
    status_code: int,
    expected_requests: int,
) -> None:
    """Propagate persistent API errors after exhausting SDK and application retries."""
    request_count = 0

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        return httpx2.Response(
            status_code, request=request, json={"error": {"message": "Request failed"}}
        )

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = StableDiffusionImageModel(
        model="image-model", base_url="https://image.test/v1", api_key_env="CUWM_TEST_API_KEY"
    )
    with pytest.raises(openai.APIStatusError):
        model.generate(screenshot, "Edit the screenshot")
    assert request_count == expected_requests
    assert len(retry_pauses) == expected_requests - 1


@pytest.mark.parametrize(
    "max_size, padding_multiple, uploaded_size, output_size",
    [(None, 1, (31, 17), (31, 17)), (16, 8, (16, 16), (16, 9))],
)
def test_image_model_uses_only_explicit_geometry_and_extensions(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: bytes,
    max_size: int | None,
    padding_multiple: int,
    uploaded_size: tuple[int, int],
    output_size: tuple[int, int],
) -> None:
    """Send another image model without automatic Mage parameters or padding."""
    encoded_prediction = base64.b64encode(encode_png(uploaded_size, "green")).decode("ascii")

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        assert request.url == "https://another-image.test/v1/images/edits"
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read()
        )
        fields = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()
        }
        assert set(fields) == {
            "model",
            "image",
            "prompt",
            "size",
            "n",
            "response_format",
            "output_format",
        }
        assert fields["model"] == b"another-image-model"
        assert fields["prompt"] == b"Edit this screenshot"
        assert fields["size"] == f"{uploaded_size[0]}x{uploaded_size[1]}".encode()
        assert fields["n"] == b"1"
        assert fields["response_format"] == b"b64_json"
        assert fields["output_format"] == b"png"
        image_bytes = fields["image"]
        assert isinstance(image_bytes, bytes)
        with Image.open(BytesIO(image_bytes)) as image:
            assert image.size == uploaded_size
            assert image.getpixel((output_size[0] - 1, output_size[1] - 1)) == (0, 0, 255)
            if uploaded_size != output_size:
                assert image.getpixel((uploaded_size[0] - 1, uploaded_size[1] - 1)) == (
                    255,
                    255,
                    255,
                )
        return httpx2.Response(
            200, request=request, json={"created": 0, "data": [{"b64_json": encoded_prediction}]}
        )

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = StableDiffusionImageModel(
        model="another-image-model",
        base_url="https://another-image.test/v1",
        api_key_env="CUWM_TEST_API_KEY",
        max_size=max_size,
        padding_multiple=padding_multiple,
    )
    result = model.generate(screenshot, "Edit this screenshot")
    with Image.open(BytesIO(result)) as prediction:
        assert prediction.size == output_size
        assert prediction.format == "PNG"
        assert prediction.getpixel((output_size[0] - 1, output_size[1] - 1)) == (0, 128, 0)


@pytest.mark.parametrize(
    "steps, cfg, seed, expected_params",
    [
        (23, None, None, {"sample_params": {"sample_steps": 23}}),
        (None, 0.0, None, {"sample_params": {"guidance": {"txt_cfg": 0.0, "img_cfg": 0.0}}}),
        (None, None, 0, {"seed": 0}),
    ],
)
def test_stable_diffusion_sends_only_configured_sampling_parameters(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: bytes,
    steps: int | None,
    cfg: float | None,
    seed: int | None,
    expected_params: dict[str, object],
) -> None:
    """Preserve zero values and omit native controls that were not configured."""
    encoded_prediction = base64.b64encode(encode_png((31, 17), "green")).decode("ascii")
    request_count = 0

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read()
        )
        fields = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()
        }
        prompt = fields["prompt"]
        assert isinstance(prompt, bytes)
        instruction, extra_params = prompt.split(b"<sd_cpp_extra_args>")
        assert instruction == b"Edit this screenshot"
        assert extra_params.endswith(b"</sd_cpp_extra_args>")
        assert json.loads(extra_params.removesuffix(b"</sd_cpp_extra_args>")) == expected_params
        return httpx2.Response(
            200, request=request, json={"created": 0, "data": [{"b64_json": encoded_prediction}]}
        )

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = StableDiffusionImageModel(
        model="another-image-model",
        base_url="https://image.test/v1",
        api_key_env="CUWM_TEST_API_KEY",
        steps=steps,
        cfg=cfg,
        seed=seed,
    )
    result = model.generate(screenshot, "Edit this screenshot")
    with Image.open(BytesIO(result)) as prediction:
        assert prediction.size == (31, 17)
        assert prediction.format == "PNG"
    assert request_count == 1


@pytest.mark.parametrize("is_text_only", [False, True])
def test_cli_uses_the_configured_mage_request_and_preserves_text_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    screenshot: bytes,
    is_text_only: bool,
) -> None:
    """Build both configured adapters and execute only the requested CLI stages."""
    requested_paths: list[str] = []
    encoded_prediction = base64.b64encode(encode_png((32, 32), "green")).decode("ascii")

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            assert request.url.host == "text.test"
            assert request.headers["authorization"] == "Bearer text-key"
            body = json.loads(request.read())
            assert body["model"] == "qwen3-vl"
            assert body["messages"][0] == {"role": "system", "content": ""}
            prompt = body["messages"][1]["content"][1]["text"]
            assert CANDIDATE_ACTION in prompt
            assert GUI_DESCRIPTION in prompt
            assert "Microsoft Excel" in prompt
            assert prompt.count("Your task is to predict and describe") == 1
            assert (
                "Current UI Screenshot:\n[Current UI screenshot attached as image content]"
                in prompt
            )
            image_url = body["messages"][1]["content"][0]["image_url"]["url"]
            assert image_url.startswith("data:image/png;base64,")
            assert base64.b64decode(image_url.split(",", 1)[1]) == screenshot
            return httpx2.Response(
                200,
                request=request,
                json={
                    "id": "text-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "qwen3-vl",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Column H selected."},
                        }
                    ],
                },
            )
        assert request.url == "https://image.test/v1/images/edits"
        assert request.headers["authorization"] == "Bearer image-key"
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read()
        )
        fields = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()
        }
        assert fields["model"] == b"microsoft/Mage-Flow-Edit-Turbo"
        assert fields["size"] == b"32x32"
        assert fields["response_format"] == b"b64_json"
        assert fields["output_format"] == b"png"
        prompt = fields["prompt"]
        assert isinstance(prompt, bytes)
        assert b"Column H selected." in prompt
        assert b"<sd_cpp_extra_args>" in prompt
        extra_params = prompt.split(b"<sd_cpp_extra_args>", 1)[1]
        assert extra_params.endswith(b"</sd_cpp_extra_args>")
        assert json.loads(extra_params.removesuffix(b"</sd_cpp_extra_args>")) == {
            "sample_params": {
                "sample_steps": 4,
                "guidance": {"txt_cfg": 1.0, "img_cfg": 1.0},
            },
            "seed": 2026,
        }
        assert set(fields) == {
            "model",
            "image",
            "prompt",
            "size",
            "n",
            "response_format",
            "output_format",
        }
        return httpx2.Response(
            200, request=request, json={"created": 0, "data": [{"b64_json": encoded_prediction}]}
        )

    monkeypatch.setenv("OPENAI_BASE_URL", "https://text.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "text-key")
    monkeypatch.setenv("IMAGE_BASE_URL", "https://image.test/v1")
    monkeypatch.setenv("IMAGE_API_KEY", "image-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    screenshot_path = tmp_path / "source.png"
    screenshot_path.write_bytes(screenshot)
    output_dir = tmp_path / "output"
    arguments = [
        "run_cuwm.py",
        "--screenshot",
        str(screenshot_path),
        "--action",
        CANDIDATE_ACTION,
        "--app-name",
        "Microsoft Excel",
        "--gui-description",
        GUI_DESCRIPTION,
        "--config",
        "config/cuwm.toml",
        "--output-dir",
        str(output_dir),
    ]
    if is_text_only:
        arguments.append("--text-only")
    monkeypatch.setattr(sys, "argv", arguments)
    run_cuwm.main()
    assert (output_dir / "transition.txt").read_text() == "Column H selected.\n"
    image_path = output_dir / "predicted_next_state.png"
    if is_text_only:
        assert requested_paths == ["/v1/chat/completions"]
        assert not image_path.exists()
    else:
        assert requested_paths == ["/v1/chat/completions", "/v1/images/edits"]
        with Image.open(image_path) as prediction:
            assert prediction.size == (31, 17)
            assert prediction.format == "PNG"


@pytest.mark.parametrize(
    "encoded_prediction, expected_error",
    [
        ("invalid-base64!", binascii.Error),
        (base64.b64encode(b"not an image").decode("ascii"), UnidentifiedImageError),
        (base64.b64encode(encode_png((32, 32), "green")).decode("ascii"), ValueError),
    ],
)
def test_image_model_rejects_invalid_response_data_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: bytes,
    retry_pauses: list[float],
    encoded_prediction: str,
    expected_error: type[Exception],
) -> None:
    """Reject invalid base64, undecodable images, and unexpected response dimensions."""

    def send_response(
        client: httpx2.Client, request: httpx2.Request, **options: object
    ) -> httpx2.Response:
        return httpx2.Response(
            200, request=request, json={"created": 0, "data": [{"b64_json": encoded_prediction}]}
        )

    monkeypatch.setenv("CUWM_TEST_API_KEY", "test-key")
    monkeypatch.setattr(httpx2.Client, "send", send_response)
    model = StableDiffusionImageModel(
        model="image-model", base_url="https://image.test/v1", api_key_env="CUWM_TEST_API_KEY"
    )
    with pytest.raises(expected_error):
        model.generate(screenshot, "Edit the screenshot")
    assert retry_pauses == []


@pytest.mark.parametrize("max_size, padding_multiple", [(0, 1), (-1, 1), (None, 0), (None, -1)])
def test_image_model_rejects_nonpositive_geometry(
    max_size: int | None, padding_multiple: int
) -> None:
    """Reject invalid client geometry before attempting inference."""
    with pytest.raises(ValueError, match="positive"):
        StableDiffusionImageModel(
            model="image-model", max_size=max_size, padding_multiple=padding_multiple
        )


@pytest.mark.parametrize("missing_option", ["--app-name", "--gui-description"])
def test_cli_requires_the_paper_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing_option: str
) -> None:
    """Explain missing paper inputs before loading files or constructing model clients."""
    arguments = {
        "--screenshot": "missing.png",
        "--action": CANDIDATE_ACTION,
        "--app-name": "Microsoft Excel",
        "--gui-description": GUI_DESCRIPTION,
        "--output-dir": "unused-output",
    }
    arguments.pop(missing_option)
    monkeypatch.setattr(
        sys, "argv", ["run_cuwm.py", *(value for pair in arguments.items() for value in pair)]
    )
    with pytest.raises(SystemExit) as error:
        run_cuwm.main()
    assert error.value.code == 2
    assert missing_option in capsys.readouterr().err


@pytest.mark.parametrize("method_name", ["generate_text_desc", "predict"])
@pytest.mark.parametrize("missing_argument", ["app_name", "gui_description"])
def test_facade_requires_explicit_paper_context(method_name: str, missing_argument: str) -> None:
    """Reject omitted context through both public textual inference entry points."""
    cuwm = CUWM(text_model=Mock(), image_model=Mock())
    context = {"app_name": "Microsoft Excel", "gui_description": GUI_DESCRIPTION}
    context.pop(missing_argument)
    method = getattr(cuwm, method_name)
    with pytest.raises(TypeError, match=missing_argument):
        method(b"original screenshot", CANDIDATE_ACTION, **context)
