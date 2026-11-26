import base64
import datetime as dt
import hashlib
import json
import traceback
import uuid
from collections.abc import Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel


def write_json(path: Path, value: object) -> None:
    """Write a value as indented UTF-8 JSON, converting unsupported values to strings.

    Args:
        path: Destination file.
        value: JSON-compatible value to write.
    """
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def format_timestamp() -> str:
    """Return the current UTC time in ISO 8601 format.

    Returns:
        Current UTC timestamp.
    """
    return dt.datetime.now(dt.UTC).isoformat()


def format_error(error: BaseException) -> str:
    """Format an exception with its traceback.

    Args:
        error: Exception to format.

    Returns:
        Traceback text ending with the exception message.
    """
    return "".join(traceback.format_exception(error))


class ExperimentLog:
    """Directory of one benchmark experiment with per-task records and deduplicated images.

    The directory contains `experiment.json`, `summary.json`, `images/{sha256}.png`, and
    `tasks/{task_id}/` with `task.json`, `spans.jsonl`, and the task recording.

    Attributes:
        root: Experiment directory.

    Args:
        root: Experiment directory, created on first write.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_metadata(self, metadata: dict) -> None:
        """Write the experiment metadata to `experiment.json`.

        Args:
            metadata: Experiment name, creation time, tags, settings, and agent configuration.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "experiment.json", metadata)

    def write_summary(self, metrics: dict, task_results: list[dict]) -> None:
        """Write aggregate metrics and per-task results to `summary.json`.

        Args:
            metrics: Metric names mapped to their values.
            task_results: One result record per completed task.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "summary.json", {"metrics": metrics, "task_results": task_results})

    def store_image(self, content: bytes) -> str:
        """Store PNG bytes once under their SHA-256 digest, reusing an existing identical file.

        Args:
            content: PNG image bytes.

        Returns:
            Image path relative to the experiment directory.
        """
        relative_path = f"images/{hashlib.sha256(content).hexdigest()}.png"
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileExistsError), path.open("xb") as image_file:
            image_file.write(content)
        return relative_path

    def serialize(self, value: object) -> object:
        """Convert a span value to JSON-compatible data, storing images as file references.

        Bytes and base64 image data URLs are stored as PNG images and replaced by
        `{"image": "images/{sha256}.png"}`. Pydantic models are dumped, containers are
        converted recursively, and other unsupported values become strings.

        Args:
            value: Span input or output value.

        Returns:
            JSON-compatible representation of the value.
        """
        match value:
            case bytes():
                return {"image": self.store_image(value)}
            case str() if value.startswith("data:image/") and ";base64," in value:
                return {"image": self.store_image(base64.b64decode(value.partition(",")[2]))}
            case BaseModel():
                return self.serialize(value.model_dump(mode="json"))
            case dict():
                return {str(key): self.serialize(item) for key, item in value.items()}
            case list() | tuple():
                return [self.serialize(item) for item in value]
            case None | bool() | int() | float() | str():
                return value
            case _:
                return str(value)

    @contextmanager
    def start_task(self, domain: str, task_id: str) -> Generator[TaskLog]:
        """Record one task in `tasks/{task_id}/` and make it the target of new spans.

        On exit, `task.json` stores the task status, timestamps, and the traceback of an
        exception, which is propagated.

        Args:
            domain: OSWorld domain of the task.
            task_id: OSWorld task identifier.

        Yields:
            Log of the active task.
        """
        task_log = TaskLog(self, self.root / "tasks" / task_id)
        task_log.directory.mkdir(parents=True, exist_ok=True)
        record = {
            "domain": domain,
            "task_id": task_id,
            "status": "OK",
            "started_at": format_timestamp(),
        }
        token = CURRENT_TASK.set(task_log)
        try:
            yield task_log
        except BaseException as error:
            record |= {"status": "ERROR", "error": format_error(error)}
            raise
        finally:
            CURRENT_TASK.reset(token)
            record["ended_at"] = format_timestamp()
            write_json(task_log.directory / "task.json", record)


@dataclass
class TaskLog:
    """Directory of one task within an experiment.

    Attributes:
        experiment: Experiment that stores the task and its images.
        directory: Task directory containing `task.json`, `spans.jsonl`, and the recording.
    """

    experiment: ExperimentLog
    directory: Path

    def append_span(self, record: dict) -> None:
        """Serialize a finished span and append it as one line of `spans.jsonl`.

        Args:
            record: Span identifiers, timing, status, inputs, and outputs.
        """
        with (self.directory / "spans.jsonl").open("a", encoding="utf-8") as spans_file:
            spans_file.write(json.dumps(self.experiment.serialize(record)) + "\n")


@dataclass
class Span:
    """Unit of traced work whose inputs and outputs are recorded when it finishes.

    Attributes:
        name: Span name, such as `predict` or `step_1`.
        span_type: Span category, such as `CHAIN`, `AGENT`, `TOOL`, or `CHAT_MODEL`.
        inputs: Values consumed by the work.
        outputs: Values produced by the work, assigned before the span finishes.
        parent_id: Identifier of the enclosing span, or None for a root span.
        span_id: Unique span identifier.
    """

    name: str
    span_type: str
    inputs: object = None
    outputs: object = None
    parent_id: str | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)


CURRENT_TASK: ContextVar[TaskLog | None] = ContextVar("current_task", default=None)
CURRENT_SPAN: ContextVar[str | None] = ContextVar("current_span", default=None)


@contextmanager
def span(name: str, span_type: str, inputs: object = None) -> Generator[Span]:
    """Trace work nested in the current span and append it to the active task on exit.

    Outside an active task the span is yielded but not recorded. An exception is recorded
    with its traceback and propagated.

    Args:
        name: Span name.
        span_type: Span category, such as `CHAIN`, `AGENT`, `TOOL`, or `CHAT_MODEL`.
        inputs: Values consumed by the work.

    Yields:
        Span whose `outputs` can be assigned before it finishes.
    """
    current_span = Span(name, span_type, inputs, parent_id=CURRENT_SPAN.get())
    task_log = CURRENT_TASK.get()
    record = {"status": "OK", "start_time": format_timestamp()}
    token = CURRENT_SPAN.set(current_span.span_id)
    try:
        yield current_span
    except BaseException as error:
        record |= {"status": "ERROR", "error": format_error(error)}
        raise
    finally:
        CURRENT_SPAN.reset(token)
        if task_log is not None:
            task_log.append_span(
                {
                    "span_id": current_span.span_id,
                    "parent_id": current_span.parent_id,
                    "name": current_span.name,
                    "span_type": current_span.span_type,
                    **record,
                    "end_time": format_timestamp(),
                    "inputs": current_span.inputs,
                    "outputs": current_span.outputs,
                }
            )
