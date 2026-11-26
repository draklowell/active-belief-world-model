#!/usr/bin/env python
"""Label sampled CUWM transitions with textual state changes written by a teacher VLM.

Reads the transitions and screenshots written by build_cuwm_transitions.py and
download_agentnet_images.py, sends the current and next screenshot of every transition with the
cuwm-qwen3vl/transition_annotation.j2 prompt to the teacher server, and appends one JSON record
per labelled transition to the output file. Rerunning skips transitions already in the output, so
an interrupted run resumes. A transition whose label is truncated or rejected is logged and left
out, and a rerun requests it again. Every run appends its timing summary, which separates the
time spent in the vLLM engine from the time spent outside it, such as uploading screenshots, to a
sibling file with the suffix .runs.jsonl.
"""

import argparse
import datetime as dt
import json
import logging
import re
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import polars as pl
from openai import BadRequestError

from prompts import render
from world_models.backends.openai_text import OpenAITextModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

ANNOTATION_TEMPLATE = "cuwm-qwen3vl/transition_annotation.j2"
CURRENT_IMAGE_MARKER = "[Current UI screenshot attached as the first image]"
NEXT_IMAGE_MARKER = "[Next UI screenshot attached as the second image]"
BASE64_EXPANSION = 4 / 3
SERVER_HISTOGRAMS = {
    "e2e_seconds": "vllm:e2e_request_latency_seconds",
    "queue_seconds": "vllm:request_queue_time_seconds",
    "prefill_seconds": "vllm:request_prefill_time_seconds",
    "decode_seconds": "vllm:request_decode_time_seconds",
    "prompt_tokens": "vllm:request_prompt_tokens",
    "generation_tokens": "vllm:request_generation_tokens",
}
HISTOGRAM_LINE_PATTERN = r"^(vllm:[\w:]+?)_(sum|count)(?:\{[^}]*\})?\s+(\S+)$"


def render_annotation_prompt(transition: dict, should_include_next_observation: bool) -> str:
    """Render the teacher prompt of one transition with its action in the inference form.

    Args:
        transition: Sample row with tool_call, gui_description and next_observation.
        should_include_next_observation: Whether to add the AgentNet observation of the next step
            as a hint.

    Returns:
        Prompt text that refers to the two attached screenshots through their markers.
    """
    return render(
        ANNOTATION_TEMPLATE,
        image=CURRENT_IMAGE_MARKER,
        next_image=NEXT_IMAGE_MARKER,
        action=transition["tool_call"],
        gui_description=transition["gui_description"],
        next_observation=(
            transition["next_observation"] if should_include_next_observation else None
        ),
    )


def read_labelled_keys(output: Path) -> set[tuple[str, int]]:
    """Read the transitions that already have a label in the output file.

    Args:
        output: JSON Lines file of earlier labels, which may not exist yet.

    Returns:
        Task id and step index of every labelled transition.
    """
    if not output.exists():
        return set()
    with output.open() as labels:
        records = [json.loads(line) for line in labels if line.strip()]
    return {(record["task_id"], record["index"]) for record in records}


def select_pending(
    transitions: pl.DataFrame,
    labelled: set[tuple[str, int]],
    split: str | None,
    limit: int | None,
) -> pl.DataFrame:
    """Select the transitions of a run and drop those already labelled.

    The run covers the transitions of the split, ordered by task id and step index and cut to the
    limit, so a resumed run with the same arguments covers the same transitions.

    Args:
        transitions: Sample with task_id, index and split columns.
        labelled: Task id and step index of transitions that already have a label.
        split: Split to label, or None for every split.
        limit: Maximum number of transitions of the run, or None for all of them.

    Returns:
        Transitions of the run without a label, in task id and step index order.
    """
    selected = transitions.sort("task_id", "index")
    if split is not None:
        selected = selected.filter(pl.col("split") == split)
    if limit is not None:
        selected = selected.head(limit)
    is_labelled = [
        (task_id, index) in labelled
        for task_id, index in zip(selected["task_id"], selected["index"])
    ]
    return selected.filter(~pl.Series(is_labelled, dtype=pl.Boolean))


def read_served_checkpoint(model: OpenAITextModel) -> str:
    """Read the checkpoint that the server loaded under the configured model name.

    Args:
        model: Client of the teacher server.

    Returns:
        Checkpoint identifier reported by vLLM as the root of the served model.
    """
    served = {entry.id: str(entry.to_dict()["root"]) for entry in model.client.models.list()}
    if model.model not in served:
        raise ValueError(f"The teacher server does not serve {model.model!r}: {sorted(served)}.")
    return served[model.model]


def read_server_histograms(metrics_url: str) -> dict[tuple[str, str], float]:
    """Read the sums and counts of every vLLM histogram from its Prometheus endpoint.

    Args:
        metrics_url: URL of the server's /metrics endpoint.

    Returns:
        Mapping from metric name and `sum` or `count` to its total over all label sets.
    """
    response = httpx.get(metrics_url, timeout=30)
    response.raise_for_status()
    totals: dict[tuple[str, str], float] = {}
    for line in response.text.splitlines():
        match = re.match(HISTOGRAM_LINE_PATTERN, line)
        if match:
            key = (match[1], match[2])
            totals[key] = totals.get(key, 0.0) + float(match[3])
    return totals


def summarize_server_requests(
    before: dict[tuple[str, str], float], after: dict[tuple[str, str], float]
) -> dict[str, float]:
    """Average the server's request histograms over the requests between two readings.

    Args:
        before: Histogram totals read before the run.
        after: Histogram totals read after the run.

    Returns:
        Number of finished requests and the mean of every histogram in SERVER_HISTOGRAMS, zero
        when no request finished.
    """

    def increase(name: str, part: str) -> float:
        return after.get((name, part), 0.0) - before.get((name, part), 0.0)

    summary = {"requests": increase(SERVER_HISTOGRAMS["e2e_seconds"], "count")}
    for key, name in SERVER_HISTOGRAMS.items():
        count = increase(name, "count")
        summary[key] = increase(name, "sum") / count if count else 0.0
    return summary


def label_transition(
    model: OpenAITextModel,
    transition: dict,
    images_dir: Path,
    should_include_next_observation: bool,
) -> tuple[str | None, float]:
    """Request the textual state change of one transition from the teacher and time it.

    A truncated or rejected answer is logged and gives no label.

    Args:
        model: Client of the teacher server.
        transition: Sample row with both screenshot names and the action fields.
        images_dir: Directory holding the screenshots.
        should_include_next_observation: Whether to add the AgentNet observation of the next step
            as a hint.

    Returns:
        Label text written by the teacher, or None without a label, and the seconds of the
        request from encoding the screenshots to receiving the answer, including retries.
    """
    screenshots = [
        (images_dir / transition["image"]).read_bytes(),
        (images_dir / transition["next_image"]).read_bytes(),
    ]
    prompt = render_annotation_prompt(transition, should_include_next_observation)
    started = time.monotonic()
    try:
        label = model.generate_from_screenshots(screenshots, prompt, system_prompt="")
    except (ValueError, BadRequestError) as error:
        logger.warning("No label for %s/%d: %s", transition["task_id"], transition["index"], error)
        label = None
    return label, time.monotonic() - started


def main() -> None:
    """Label the transitions of a run and append the labels with teacher metadata to the output.

    Every screenshot of the run must be on disk before a request is sent. Requests run in
    parallel threads, and each label is written and flushed as soon as it arrives. The server's
    request histograms are read before and after the requests, so the mean request time seen by
    the client can be split into time inside the vLLM engine and time outside it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/cuwm-labels.toml"))
    parser.add_argument("--output", type=Path, required=True, help="JSON Lines file of labels")
    parser.add_argument("--gpu", required=True, help="GPU of the teacher server, such as G4")
    parser.add_argument("--split", help="Split to label, every split when omitted")
    parser.add_argument("--limit", type=int, help="Maximum number of transitions of the run")
    parser.add_argument(
        "--include-next-observation",
        action="store_true",
        help="Add the AgentNet observation of the next step to the prompt as a hint",
    )
    args = parser.parse_args()
    with args.config.open("rb") as config_file:
        settings = tomllib.load(config_file)
    server_config = json.loads(Path(settings["server_config"]).read_text())
    images_dir = Path(settings["images_dir"])

    transitions = pl.read_parquet(settings["transitions"])
    pending = select_pending(
        transitions, read_labelled_keys(args.output), args.split, args.limit
    ).to_dicts()
    missing = {
        name
        for transition in pending
        for name in (transition["image"], transition["next_image"])
        if not (images_dir / name).exists()
    }
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} screenshots of the run are missing from {images_dir}; "
            "run download_agentnet_images.py first."
        )
    if not pending:
        logger.info("Every transition of the run is already labelled in %s", args.output)
        return

    model = OpenAITextModel(**settings["text_model"])
    checkpoint = read_served_checkpoint(model)
    if checkpoint != server_config["model"]:
        raise ValueError(
            f"The teacher server runs {checkpoint}, but {settings['server_config']} "
            f"names {server_config['model']}."
        )
    teacher = {
        "checkpoint": checkpoint,
        "revision": server_config.get("revision"),
        "served_model": model.model,
        "gpu": args.gpu,
        "max_tokens": model.max_tokens,
        "temperature": model.temperature,
        "chat_template_kwargs": model.chat_template_kwargs,
        "template": ANNOTATION_TEMPLATE,
    }
    metrics_url = str(model.client.base_url.join("/metrics"))
    logger.info("Labelling %d transitions with %s on %s", len(pending), checkpoint, args.gpu)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    histograms_before = read_server_histograms(metrics_url)
    started = time.monotonic()
    request_seconds = []
    upload_bytes = []
    labels_written = 0
    with (
        ThreadPoolExecutor(settings["workers"]) as pool,
        args.output.open("a") as labels,
    ):
        futures = {
            pool.submit(
                label_transition, model, transition, images_dir, args.include_next_observation
            ): transition
            for transition in pending
        }
        for future in as_completed(futures):
            transition = futures[future]
            label, seconds = future.result()
            screenshot_bytes = sum(
                (images_dir / transition[name]).stat().st_size for name in ("image", "next_image")
            )
            request_seconds.append(seconds)
            upload_bytes.append(screenshot_bytes * BASE64_EXPANSION)
            if label is None:
                continue
            record = {
                "task_id": transition["task_id"],
                "index": transition["index"],
                "split": transition["split"],
                "label": label,
                "has_next_observation_hint": args.include_next_observation,
                "request_seconds": round(seconds, 3),
                "screenshot_bytes": screenshot_bytes,
                "teacher": teacher,
                "labelled_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
            labels.write(json.dumps(record) + "\n")
            labels.flush()
            labels_written += 1
    elapsed = time.monotonic() - started
    server = summarize_server_requests(histograms_before, read_server_histograms(metrics_url))
    mean_request_seconds = sum(request_seconds) / len(request_seconds)
    run = {
        "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "limit": args.limit,
        "workers": settings["workers"],
        "has_next_observation_hint": args.include_next_observation,
        "teacher": teacher,
        "requests": len(request_seconds),
        "labels": labels_written,
        "wall_seconds": round(elapsed, 1),
        "requests_per_minute": round(len(request_seconds) / elapsed * 60, 2),
        "mean_request_seconds": round(mean_request_seconds, 3),
        "mean_upload_megabytes": round(sum(upload_bytes) / len(upload_bytes) / 1e6, 3),
        "server": {key: round(value, 3) for key, value in server.items()},
        "mean_seconds_outside_engine": round(mean_request_seconds - server["e2e_seconds"], 3),
    }
    with args.output.with_suffix(".runs.jsonl").open("a") as runs:
        runs.write(json.dumps(run) + "\n")
    logger.info(
        "%d labels from %d requests in %.0f s (%.1f per minute) into %s; mean request %.2f s, "
        "of which %.2f s in the engine (queue %.2f, prefill %.2f, decode %.2f) and %.2f s "
        "outside it for %.2f MB of screenshots",
        labels_written,
        run["requests"],
        elapsed,
        run["requests_per_minute"],
        args.output,
        mean_request_seconds,
        server["e2e_seconds"],
        server["queue_seconds"],
        server["prefill_seconds"],
        server["decode_seconds"],
        run["mean_seconds_outside_engine"],
        run["mean_upload_megabytes"],
    )


if __name__ == "__main__":
    main()
