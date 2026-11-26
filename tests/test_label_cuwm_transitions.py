import inspect
import json
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import polars as pl
import pytest
from openai.types import Model

from scripts import label_cuwm_transitions
from world_models.backends.openai_text import OpenAITextModel

REPO_ROOT = Path(__file__).parent.parent

TRANSITION = {
    "task_id": "a",
    "index": 0,
    "split": "validation",
    "image": "a0.png",
    "next_image": "a1.png",
    "tool_arguments": '{"action": "left_click", "coordinate": [407, 530]}',
    "gui_description": "Click the toggle switch.",
    "next_observation": "The toggle switch is on.",
}


def sample_frame() -> pl.DataFrame:
    """Build a sample of five transitions over two splits, deliberately out of order."""
    rows = [
        {**TRANSITION, "task_id": task_id, "index": index, "split": split}
        for task_id, index, split in [
            ("b", 1, "validation"),
            ("a", 2, "validation"),
            ("c", 0, "train"),
            ("a", 1, "validation"),
            ("b", 0, "validation"),
        ]
    ]
    return pl.DataFrame(rows)


def test_render_annotation_prompt_uses_inference_action_and_ordered_markers() -> None:
    """Check the action form, the screenshot markers and that the hint is off by default."""
    prompt = label_cuwm_transitions.render_annotation_prompt(TRANSITION, False)
    action = (
        '{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [407, 530]}}'
    )
    assert action in prompt
    assert "Click the toggle switch." in prompt
    assert prompt.index(label_cuwm_transitions.CURRENT_IMAGE_MARKER) < prompt.index(
        label_cuwm_transitions.NEXT_IMAGE_MARKER
    )
    assert "The toggle switch is on." not in prompt


def test_render_annotation_prompt_adds_next_observation_hint_on_request() -> None:
    """Check that the AgentNet observation of the next step is inserted when requested."""
    prompt = label_cuwm_transitions.render_annotation_prompt(TRANSITION, True)
    assert "The toggle switch is on." in prompt


def test_read_labelled_keys_returns_nothing_without_output(tmp_path: Path) -> None:
    """Check that a first run starts with no labelled transitions."""
    assert label_cuwm_transitions.read_labelled_keys(tmp_path / "labels.jsonl") == set()


def test_select_pending_cuts_the_run_before_dropping_labelled_transitions() -> None:
    """Check split filtering, ordering and that the limit fixes the transitions of a run."""
    pending = label_cuwm_transitions.select_pending(
        sample_frame(), {("a", 1)}, split="validation", limit=3
    )
    assert list(zip(pending["task_id"], pending["index"])) == [("a", 2), ("b", 0)]


def test_select_pending_covers_every_split_without_limits() -> None:
    """Check that omitting the split and the limit selects the whole sample."""
    pending = label_cuwm_transitions.select_pending(sample_frame(), set(), split=None, limit=None)
    assert pending.height == 5


def served_models(*entries: tuple[str, str]) -> MagicMock:
    """Build a text model client whose server lists the given served names and checkpoints."""
    model = MagicMock()
    model.model = "cuwm-teacher"
    model.client.models.list.return_value = [
        Model.model_validate(
            {"id": name, "object": "model", "created": 0, "owned_by": "vllm", "root": root}
        )
        for name, root in entries
    ]
    return model


def test_read_served_checkpoint_returns_the_root_of_the_served_name() -> None:
    """Check that the checkpoint comes from the vLLM model listing."""
    model = served_models(("other", "Qwen/Other"), ("cuwm-teacher", "Qwen/Teacher"))
    assert label_cuwm_transitions.read_served_checkpoint(model) == "Qwen/Teacher"


def test_read_served_checkpoint_rejects_a_server_without_the_name() -> None:
    """Check that a server serving another model name stops the run."""
    with pytest.raises(ValueError, match="does not serve 'cuwm-teacher'"):
        label_cuwm_transitions.read_served_checkpoint(served_models(("other", "Qwen/Other")))


SERVER_READING_BEFORE = {
    ("vllm:e2e_request_latency_seconds", "sum"): 10.0,
    ("vllm:e2e_request_latency_seconds", "count"): 4.0,
}
SERVER_READING_AFTER = {
    ("vllm:e2e_request_latency_seconds", "sum"): 16.0,
    ("vllm:e2e_request_latency_seconds", "count"): 7.0,
    ("vllm:request_prompt_tokens", "sum"): 15000.0,
    ("vllm:request_prompt_tokens", "count"): 3.0,
}
METRICS_TEXT = """# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.3",model_name="cuwm-teacher"} 0.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="cuwm-teacher"} 12.5
vllm:e2e_request_latency_seconds_count{engine="0",model_name="cuwm-teacher"} 5.0
vllm:e2e_request_latency_seconds_sum{engine="1",model_name="cuwm-teacher"} 2.5
vllm:e2e_request_latency_seconds_count{engine="1",model_name="cuwm-teacher"} 1.0
vllm:num_requests_running{engine="0",model_name="cuwm-teacher"} 0.0
vllm:request_generation_tokens_sum 900.0
"""


def test_read_server_histograms_sums_sums_and_counts_over_label_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that Prometheus histogram sums and counts are read and buckets and gauges ignored."""
    urls: list[str] = []

    def get_metrics(url: str, timeout: float) -> httpx.Response:
        urls.append(url)
        return httpx.Response(200, text=METRICS_TEXT, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get_metrics)
    histograms = label_cuwm_transitions.read_server_histograms("http://teacher-server:8000/metrics")
    assert histograms == {
        ("vllm:e2e_request_latency_seconds", "sum"): 15.0,
        ("vllm:e2e_request_latency_seconds", "count"): 6.0,
        ("vllm:request_generation_tokens", "sum"): 900.0,
    }
    assert urls == ["http://teacher-server:8000/metrics"]


def test_summarize_server_requests_averages_over_the_run_and_forgives_missing_histograms() -> None:
    """Check per request means between two readings, with zero for histograms without requests."""
    summary = label_cuwm_transitions.summarize_server_requests(
        SERVER_READING_BEFORE, SERVER_READING_AFTER
    )
    assert summary["requests"] == 3.0
    assert summary["e2e_seconds"] == 2.0
    assert summary["prompt_tokens"] == 5000.0
    assert summary["decode_seconds"] == 0.0


class FakeTeacher:
    """Text model stand-in that labels every transition except one with a truncated answer.

    Args:
        model: Served model name.
        max_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature.
        chat_template_kwargs: Chat template options of the request.
        **_: Remaining client options, ignored.
    """

    requests: list[tuple[list[bytes], str]] = []
    metrics_urls: list[str] = []

    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: dict | None = None,
        **_: object,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_template_kwargs = chat_template_kwargs
        self.client = served_models(("cuwm-teacher", "Qwen/Teacher")).client
        self.client.base_url = httpx.URL("http://teacher-server:8000/v1/")

    def generate_from_screenshots(
        self, screenshots: list[bytes], prompt: str, *, system_prompt: str
    ) -> str:
        """Return a label naming the screenshots, or fail for the truncated transition."""
        FakeTeacher.requests.append((screenshots, prompt))
        if screenshots[0] == b"truncated":
            raise ValueError("The text response was truncated; increase max_tokens.")
        return f"Label of {screenshots[0].decode()} to {screenshots[1].decode()}."


@pytest.fixture
def labelling_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare a sample, screenshots, configuration and fake teacher, and return the output."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows = [
        {**TRANSITION, "task_id": "a", "index": 0, "image": "a0.png", "next_image": "a1.png"},
        {**TRANSITION, "task_id": "a", "index": 1, "image": "a1.png", "next_image": "a2.png"},
        {**TRANSITION, "task_id": "b", "index": 0, "image": "bad.png", "next_image": "b1.png"},
    ]
    for name in ["a0.png", "a1.png", "a2.png", "b1.png"]:
        (images_dir / name).write_bytes(name.encode())
    (images_dir / "bad.png").write_bytes(b"truncated")
    pl.DataFrame(rows).write_parquet(tmp_path / "sample.parquet")
    (tmp_path / "server.json").write_text(
        json.dumps({"model": "Qwen/Teacher", "revision": "abc123"})
    )
    (tmp_path / "labels.toml").write_text(
        f'transitions = "{tmp_path / "sample.parquet"}"\n'
        f'images_dir = "{images_dir}"\n'
        f'server_config = "{tmp_path / "server.json"}"\n'
        "workers = 2\n\n"
        "[text_model]\n"
        'model = "cuwm-teacher"\n'
        "max_tokens = 512\n"
        "temperature = 0.0\n"
    )
    FakeTeacher.requests = []
    FakeTeacher.metrics_urls = []
    readings = iter([SERVER_READING_BEFORE, SERVER_READING_AFTER] * 4)

    def read_fake_histograms(metrics_url: str) -> dict[tuple[str, str], float]:
        FakeTeacher.metrics_urls.append(metrics_url)
        return next(readings)

    monkeypatch.setattr(label_cuwm_transitions, "OpenAITextModel", FakeTeacher)
    monkeypatch.setattr(label_cuwm_transitions, "read_server_histograms", read_fake_histograms)
    return tmp_path / "out" / "labels.jsonl"


def run_labelling(monkeypatch: pytest.MonkeyPatch, output: Path, *options: str) -> list[dict]:
    """Run the labelling script with the fixture configuration and return the output records."""
    config_path = output.parent.parent / "labels.toml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["label", "--config", str(config_path), "--output", str(output), "--gpu", "G4", *options],
    )
    label_cuwm_transitions.main()
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_main_writes_labels_with_teacher_metadata_and_skips_failures(
    labelling_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check record contents, screenshot order and that a truncated label is left out."""
    records = run_labelling(monkeypatch, labelling_run)
    labels = {(record["task_id"], record["index"]): record for record in records}
    assert set(labels) == {("a", 0), ("a", 1)}
    record = labels[("a", 0)]
    assert record["label"] == "Label of a0.png to a1.png."
    assert record["split"] == "validation"
    assert record["has_next_observation_hint"] is False
    assert record["teacher"] == {
        "checkpoint": "Qwen/Teacher",
        "revision": "abc123",
        "served_model": "cuwm-teacher",
        "gpu": "G4",
        "max_tokens": 512,
        "temperature": 0.0,
        "chat_template_kwargs": None,
        "template": "cuwm-qwen3vl/transition_annotation.j2",
    }
    assert record["screenshot_bytes"] == len(b"a0.png") + len(b"a1.png")
    assert record["request_seconds"] >= 0
    assert all("The toggle switch is on." not in prompt for _, prompt in FakeTeacher.requests)


def test_main_appends_a_run_summary_that_splits_engine_and_outside_time(
    labelling_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the run summary, which counts failed requests and reads the server's metrics."""
    run_labelling(monkeypatch, labelling_run)
    runs = [
        json.loads(line)
        for line in labelling_run.with_suffix(".runs.jsonl").read_text().splitlines()
    ]
    assert len(runs) == 1
    run = runs[0]
    assert (run["requests"], run["labels"]) == (3, 2)
    assert run["server"]["e2e_seconds"] == 2.0
    assert run["mean_seconds_outside_engine"] == pytest.approx(
        run["mean_request_seconds"] - 2.0, abs=1e-3
    )
    assert run["teacher"]["gpu"] == "G4"
    assert FakeTeacher.metrics_urls == ["http://teacher-server:8000/metrics"] * 2


def test_main_resumes_by_requesting_only_unlabelled_transitions(
    labelling_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that a rerun keeps earlier labels and requests only the failed transition again."""
    run_labelling(monkeypatch, labelling_run)
    FakeTeacher.requests = []
    records = run_labelling(monkeypatch, labelling_run)
    assert len(records) == 2
    assert [screenshots[0] for screenshots, _ in FakeTeacher.requests] == [b"truncated"]


def test_main_adds_the_hint_and_honours_split_and_limit(
    labelling_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the hint option and that the limit selects the first transitions of the split."""
    records = run_labelling(
        monkeypatch,
        labelling_run,
        "--split",
        "validation",
        "--limit",
        "1",
        "--include-next-observation",
    )
    assert [(record["task_id"], record["index"]) for record in records] == [("a", 0)]
    assert records[0]["has_next_observation_hint"] is True
    assert "The toggle switch is on." in FakeTeacher.requests[0][1]


def test_main_stops_before_requests_when_screenshots_are_missing(
    labelling_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that an incomplete screenshot download stops the run without any request."""
    (labelling_run.parent.parent / "images" / "a2.png").unlink()
    with pytest.raises(FileNotFoundError, match="1 screenshots of the run are missing"):
        run_labelling(monkeypatch, labelling_run)
    assert FakeTeacher.requests == []


def test_main_rejects_a_server_running_another_checkpoint(
    labelling_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that labels are never written under the metadata of a different checkpoint."""
    server_config = labelling_run.parent.parent / "server.json"
    server_config.write_text(json.dumps({"model": "Qwen/Other"}))
    with pytest.raises(ValueError, match="runs Qwen/Teacher"):
        run_labelling(monkeypatch, labelling_run)
    assert FakeTeacher.requests == []


@pytest.mark.parametrize(
    "config_path",
    sorted((REPO_ROOT / "config").glob("cuwm-labels*.toml")),
    ids=lambda path: path.name,
)
def test_repository_labelling_config_matches_the_client_and_the_teacher_server(
    config_path: Path,
) -> None:
    """Check that the supplied configurations name client options and a teacher server."""
    with config_path.open("rb") as config_file:
        settings = tomllib.load(config_file)
    client_options = set(inspect.signature(OpenAITextModel.__init__).parameters) - {"self"}
    assert set(settings["text_model"]) <= client_options
    server_config = json.loads((REPO_ROOT / settings["server_config"]).read_text())
    assert settings["text_model"]["model"] == server_config["served_model_name"]
