import base64
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from experiment_log import ExperimentLog, TaskLog, span


class Usage(BaseModel):
    """Token usage returned by a model endpoint."""

    prompt_tokens: int
    completion_tokens: int


@pytest.fixture
def experiment(tmp_path: Path) -> ExperimentLog:
    """Provide an experiment log in a temporary directory."""
    return ExperimentLog(tmp_path / "experiment")


def read_spans(task_log: TaskLog) -> list[dict]:
    """Read the finished spans of a task.

    Args:
        task_log: Log of the task.

    Returns:
        Span records in completion order.
    """
    lines = (task_log.directory / "spans.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_store_image_deduplicates_by_content(experiment: ExperimentLog) -> None:
    """Write identical images once and keep distinct images apart."""
    first = experiment.store_image(b"screenshot")
    assert experiment.store_image(b"screenshot") == first
    second = experiment.store_image(b"other")
    assert second != first
    assert first.startswith("images/") and first.endswith(".png")
    assert (experiment.root / first).read_bytes() == b"screenshot"
    assert len(list((experiment.root / "images").iterdir())) == 2


def test_serialize_replaces_images_and_dumps_models(experiment: ExperimentLog) -> None:
    """Store bytes and data URLs as shared image references and dump Pydantic models."""
    data_url = f"data:image/png;base64,{base64.b64encode(b'screenshot').decode('ascii')}"
    serialized = experiment.serialize(
        {
            "messages": [{"image_url": {"url": data_url}}],
            "image": ("screenshot.png", b"screenshot", "image/png"),
            "usage": Usage(prompt_tokens=3, completion_tokens=2),
            "path": Path("recording.mp4"),
            "text": "plain",
        }
    )
    reference = {"image": experiment.store_image(b"screenshot")}
    assert serialized == {
        "messages": [{"image_url": {"url": reference}}],
        "image": ["screenshot.png", reference, "image/png"],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "path": "recording.mp4",
        "text": "plain",
    }
    assert len(list((experiment.root / "images").iterdir())) == 1


def test_span_records_nested_parents_in_completion_order(experiment: ExperimentLog) -> None:
    """Link child spans to their parent and append each span when it finishes."""
    with experiment.start_task("chrome", "task-1") as task_log:
        with span("task", "CHAIN", inputs={"id": "task-1"}) as task_span:
            with span("predict", "AGENT") as predict_span:
                predict_span.outputs = {"actions": ["DONE"]}
            task_span.outputs = {"score": 1.0}
    child, parent = read_spans(task_log)
    assert (child["name"], parent["name"]) == ("predict", "task")
    assert parent["parent_id"] is None
    assert child["parent_id"] == parent["span_id"]
    assert child["span_type"] == "AGENT"
    assert child["outputs"] == {"actions": ["DONE"]}
    assert parent["inputs"] == {"id": "task-1"}
    assert parent["status"] == child["status"] == "OK"
    task_record = json.loads((task_log.directory / "task.json").read_text(encoding="utf-8"))
    assert task_record["domain"] == "chrome"
    assert task_record["task_id"] == "task-1"
    assert task_record["status"] == "OK"
    assert "error" not in task_record


def test_span_records_error_and_propagates(experiment: ExperimentLog) -> None:
    """Record the traceback of failed work on the span and task, then re-raise."""
    with (
        pytest.raises(ValueError, match="truncated"),
        experiment.start_task("chrome", "task-1") as task_log,
        span("chat_completion", "CHAT_MODEL"),
    ):
        raise ValueError("truncated")
    (record,) = read_spans(task_log)
    assert record["status"] == "ERROR"
    assert "ValueError: truncated" in record["error"]
    task_record = json.loads((task_log.directory / "task.json").read_text(encoding="utf-8"))
    assert task_record["status"] == "ERROR"


def test_span_outside_task_writes_nothing(tmp_path: Path) -> None:
    """Allow agents and world models to run without an active benchmark task."""
    with span("chat_completion", "CHAT_MODEL", inputs={"image": b"screenshot"}) as model_span:
        model_span.outputs = "response"
    assert model_span.outputs == "response"
    assert list(tmp_path.iterdir()) == []


def test_write_metadata_and_summary(experiment: ExperimentLog) -> None:
    """Write experiment metadata and the aggregate summary as JSON."""
    experiment.write_metadata({"name": "benchmark", "tags": {"agent": "agents.A"}})
    experiment.write_summary(
        {"average_score": 0.5}, [{"domain": "chrome", "task_id": "a", "score": 0.5, "steps": 2}]
    )
    metadata = json.loads((experiment.root / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((experiment.root / "summary.json").read_text(encoding="utf-8"))
    assert metadata["tags"] == {"agent": "agents.A"}
    assert summary["metrics"] == {"average_score": 0.5}
    assert summary["task_results"][0]["task_id"] == "a"
