# Based on https://github.com/xlang-ai/OSWorld/blob/main/scripts/python/run_multienv_qwen3vl.py

import argparse
import datetime as dt
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import dagshub
import mlflow
import polars as pl
from mlflow.entities import SpanType
from mlflow.tracing.attachments import Attachment

from agents.base import Agent

if TYPE_CHECKING:
    from desktop_env.desktop_env import DesktopEnv

logger = logging.getLogger(__name__)


def init_mlflow(mlflow_tracking_uri: str) -> None:
    """Initialize MLflow tracking server and autologging.

    Args:
        mlflow_tracking_uri: Optional MLflow tracking URI.
            If not provided, uses the default DagsHub tracking server.
    """

    if mlflow_tracking_uri == "dagshub":
        dagshub.init(repo_owner="draklowell", repo_name="active-belief-world-model", mlflow=True)
        mlflow.openai.autolog()
    elif mlflow_tracking_uri == "memory":
        mlflow.set_tracking_uri("sqlite:///:memory:")
        mlflow.openai.autolog()
    elif mlflow_tracking_uri == "disabled":
        mlflow.tracing.disable()
    else:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.openai.autolog()


class SettingsDict(TypedDict):
    max_steps: int
    sleep_after_execution: float
    provider_name: str
    aws_region: str
    path_to_vm: str
    headless: bool
    enable_proxy: bool
    password: str
    num_envs: int
    mlflow_tracking_uri: str
    mlflow_enabled: bool


class TaskResultDict(TypedDict):
    domain: str
    task_id: str
    score: float
    steps: int


def parse_args() -> argparse.Namespace:
    """Parse benchmark settings, leaving model settings in the agent configuration."""
    parser = argparse.ArgumentParser(description="Benchmark a screenshot agent on OSWorld tasks.")
    parser.add_argument(
        "--agent-config", type=Path, required=True, help="Agent configuration JSON."
    )
    parser.add_argument("--agent-path", required=True, help="Import path to an Agent subclass.")
    parser.add_argument(
        "--provider-name",
        choices=["aws", "virtualbox", "vmware", "docker", "azure", "aliyun", "pyromind"],
        default="docker",
    )
    parser.add_argument("--password", default="")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="Domain-to-task-ID JSON manifest, beside examples/<domain>/<task-id>.json.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--path-to-vm")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--sleep-after-execution", type=float, default=0.0)
    parser.add_argument(
        "--enable-proxy", action="store_true", help="Enable proxy for agent's network access."
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional suffix for the MLflow experiment name.",
    )
    parser.add_argument(
        "--experiment-tags",
        nargs="+",
        metavar="KEY=VALUE",
        default=[],
        help="Optional tags for the MLflow experiment.",
    )
    parser.add_argument(
        "--mlflow",
        default="dagshub",
        help="MLflow tracking server to use ('dagshub', 'disabled' or a custom URI).",
    )

    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 1 or args.sleep_after_execution < 0:
        parser.error(
            "num-envs and max-steps must be positive; sleep-after-execution cannot be negative"
        )
    return args


def load_agent_class(agent_path: str) -> type[Agent]:
    """Import an agent class and require the benchmark interface.

    Args:
        agent_path: Dotted module and class name, such as agents.example.ExampleAgent.

    Returns:
        The imported Agent subclass.
    """
    module_name, separator, class_name = agent_path.rpartition(".")
    if not separator or not module_name or not class_name:
        raise ValueError("agent-path must have the form module.Class")
    agent_class = getattr(import_module(module_name), class_name)
    if not isinstance(agent_class, type) or not issubclass(agent_class, Agent):
        raise TypeError(f"{agent_path} must inherit from agents.base.Agent")
    return agent_class


def run_task(agent: Agent, env: "DesktopEnv", task: dict, settings: SettingsDict) -> TaskResultDict:
    """Run one task and save its trajectory, screenshots, recording, and score.

    Args:
        agent: Agent reused across tasks in this worker.
        env: Desktop environment owned by this worker.
        task: OSWorld task configuration, including its instruction and evaluator.
        settings: Benchmark settings.

    Returns:
        The environment's evaluation score.
    """
    with mlflow.start_span("reset", span_type=SpanType.TOOL) as span:
        observation = env.reset(task_config=task)
        observation_attachment = Attachment(
            content_type="image/png", content_bytes=observation["screenshot"]
        )

        span.set_outputs({"observation": observation_attachment})

    agent.reset()

    env.controller.start_recording()

    try:
        for step in range(1, settings["max_steps"] + 1):
            with mlflow.start_span(f"step_{step}", span_type=SpanType.CHAIN):

                with mlflow.start_span("predict", span_type=SpanType.AGENT) as span:
                    response, actions = agent.predict(task["instruction"], observation)
                    span.set_inputs(
                        {"instruction": task["instruction"], "observation": observation_attachment}
                    )
                    span.set_outputs({"response": response, "actions": actions})

                is_done = False
                for action_index, action in enumerate(actions, start=1):
                    logger.info("Task %s, step %d: %s", task["id"], step, action)

                    with mlflow.start_span(
                        f"action_{action_index}", span_type=SpanType.TOOL
                    ) as span:
                        observation, reward, is_done, info = env.step(
                            action, settings["sleep_after_execution"]
                        )
                        observation_attachment = Attachment(
                            content_type="image/png", content_bytes=observation["screenshot"]
                        )

                        span.set_inputs(
                            {"action": action, "pause": settings["sleep_after_execution"]}
                        )
                        span.set_outputs(
                            {
                                "reward": reward,
                                "done": is_done,
                                "info": info,
                                "observation": observation_attachment,
                            }
                        )

                    if is_done:
                        break

                if is_done:
                    break

        return {
            "domain": task["snapshot"],
            "task_id": task["id"],
            "score": env.evaluate(),
            "steps": step + 1,
        }

    finally:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
            env.controller.end_recording(temp_file.name)
            temp_file.flush()

            video_attachment = Attachment(content_type="video/mp4", content_bytes=temp_file.read())

        with mlflow.start_span("record_video", span_type=SpanType.TOOL) as span:
            span.set_outputs({"video": video_attachment})


def run_tasks(
    settings: SettingsDict,
    agent_class: type[Agent],
    agent_config: dict,
    tasks: list[tuple[str, str, dict]],
    experiment_id: str,
) -> list[TaskResultDict]:
    """Reuse one agent and environment to evaluate a worker's assigned tasks.

    Args:
        settings: Benchmark settings.
        agent_class: Agent subclass to instantiate with the configuration dictionary.
        agent_config: Agent settings loaded from JSON without modification.
        tasks: Domain, task ID, and task configuration triples assigned to this worker.
        experiment_id: The ID of the MLflow experiment.

    Returns:
        One score per task, including zero for tasks that failed with an exception.
    """
    from desktop_env.desktop_env import DesktopEnv

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    init_mlflow(settings["mlflow_tracking_uri"])
    if settings["mlflow_enabled"]:
        mlflow.set_experiment(experiment_id=experiment_id)

    agent = agent_class(agent_config)
    snapshot_name = "init_state"
    if settings["provider_name"] == "aws":
        os.environ["AWS_REGION"] = settings["aws_region"]
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        snapshot_name = IMAGE_ID_MAP[settings["aws_region"]][(1920, 1080)]

    env = DesktopEnv(
        provider_name=settings["provider_name"],
        region=settings["aws_region"],
        path_to_vm=settings["path_to_vm"],
        snapshot_name=snapshot_name,
        action_space="pyautogui",
        headless=settings["headless"],
        require_a11y_tree=False,
        enable_proxy=settings["enable_proxy"],
        client_password=settings["password"],
    )

    try:
        task_results = []
        for domain, task_id, task in tasks:
            logger.info("Running %s/%s", domain, task_id)

            try:
                with (
                    mlflow.tracing.context(
                        tags={"domain": domain, "task_id": task_id},
                        enabled=settings["mlflow_enabled"],
                    ),
                    mlflow.start_span("task", span_type=SpanType.CHAIN) as task_span,
                ):
                    task_span.set_inputs(
                        {"task": task, "settings": settings, "agent_config": agent_config}
                    )

                    task_result = run_task(agent, env, task, settings)

                    logger.info(
                        "Results for %s/%s: score=%.4f steps=%d",
                        domain,
                        task_id,
                        task_result["score"],
                        task_result["steps"],
                    )
                    task_results.append(task_result)

                    task_span.set_outputs(task_result)
            except Exception:
                logger.exception("Task %s/%s failed", domain, task_id)

        return task_results
    finally:
        env.close()


def main() -> None:
    """Load task definitions and evaluate them across isolated worker processes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    args = parse_args()

    init_mlflow(args.mlflow)

    agent_class = load_agent_class(args.agent_path)
    agent_config = json.loads(args.agent_config.read_text(encoding="utf-8"))
    if not isinstance(agent_config, dict):
        raise ValueError("agent-config must contain a JSON object")
    manifest = json.loads(args.tasks.read_text(encoding="utf-8"))

    tasks = [
        (
            domain,
            task_id,
            json.loads(
                (args.tasks.parent / "examples" / domain / f"{task_id}.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        for domain, task_ids in manifest.items()
        for task_id in task_ids
    ]

    if not tasks:
        logger.info("No tasks selected. Average score: 0.0000")
        return

    experiment_name = f"benchmark_{args.agent_path}"
    if args.experiment_name:
        experiment_name = f"{experiment_name}_{args.experiment_name}"
    experiment_name = f"{experiment_name}_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}"

    num_envs = min(args.num_envs, len(tasks))
    logger.info(
        "Running %d tasks in %d environments. Results: %s",
        len(tasks),
        num_envs,
        experiment_name,
    )

    settings = cast(
        SettingsDict,
        {
            "provider_name": args.provider_name,
            "aws_region": args.aws_region,
            "path_to_vm": args.path_to_vm,
            "headless": args.headless,
            "enable_proxy": args.enable_proxy,
            "password": args.password,
            "max_steps": args.max_steps,
            "sleep_after_execution": args.sleep_after_execution,
            "mlflow_tracking_uri": args.mlflow,
            "mlflow_enabled": args.mlflow != "disabled",
        },
    )

    if settings["mlflow_enabled"]:
        experiment_id = mlflow.create_experiment(
            tags={"agent": args.agent_path}, name=experiment_name
        )

        for tag in args.experiment_tags:
            key, value = tag.split("=", 1)
            mlflow.set_experiment_tag(key, value)
    else:
        experiment_id = ""

    with ProcessPoolExecutor(max_workers=num_envs, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(
                run_tasks,
                settings,
                agent_class,
                agent_config,
                tasks[index::num_envs],
                experiment_id,
            )
            for index in range(num_envs)
        ]
        task_results = [task_result for future in futures for task_result in future.result()]

    dataframe = pl.DataFrame(task_results)
    success_rate = dataframe.select(pl.col("score").mean()).item()
    average_steps = dataframe.select(pl.col("steps").mean()).item()
    minimum_steps = dataframe.select(pl.col("steps").min()).item()

    logger.info("Success rate: %.4f (%d tasks)", success_rate, len(task_results))
    logger.info("Average steps: %.2f", average_steps)
    logger.info("Minimum steps: %d", minimum_steps)

    if settings["mlflow_enabled"]:
        with mlflow.start_run(experiment_id=experiment_id, run_name="summary"):
            mlflow.log_metrics(
                {
                    "average_score": success_rate,
                    "average_steps": average_steps,
                    "minimum_steps": minimum_steps,
                    "num_tasks": len(task_results),
                }
            )
            mlflow.log_table(dataframe.to_dict(as_series=False), artifact_file="task_results.json")


if __name__ == "__main__":
    main()
