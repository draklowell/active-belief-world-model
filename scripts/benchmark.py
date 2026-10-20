# Based on https://github.com/xlang-ai/OSWorld/blob/main/scripts/python/run_multienv_qwen3vl.py

import argparse
import datetime
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

from agents.base import Agent

if TYPE_CHECKING:
    from desktop_env.desktop_env import DesktopEnv

logger = logging.getLogger(__name__)


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
    parser.add_argument("--result-dir", type=Path, default=Path(".artifacts/benchmark"))
    parser.add_argument(
        "--enable-proxy", action="store_true", help="Enable proxy for agent's network access."
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


def run_task(
    agent: Agent, env: "DesktopEnv", task: dict, result_dir: Path, args: argparse.Namespace
) -> float:
    """Run one task and save its trajectory, screenshots, recording, and score.

    Args:
        agent: Agent reused across tasks in this worker.
        env: Desktop environment owned by this worker.
        task: OSWorld task configuration, including its instruction and evaluator.
        result_dir: Directory for this task's artifacts.
        args: Parsed benchmark settings.

    Returns:
        The environment's evaluation score.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    observation = env.reset(task_config=task)
    agent.reset()
    (result_dir / "initial.png").write_bytes(observation["screenshot"])
    env.controller.start_recording()
    try:
        with (result_dir / "traj.jsonl").open("w", encoding="utf-8") as trajectory:
            for step in range(1, args.max_steps + 1):
                response, actions = agent.predict(task["instruction"], observation)
                is_done = False
                for action_index, action in enumerate(actions, start=1):
                    logger.info("Task %s, step %d: %s", task["id"], step, action)
                    observation, reward, is_done, info = env.step(
                        action, args.sleep_after_execution
                    )
                    screenshot_file = f"step_{step}_{action_index}.png"
                    (result_dir / screenshot_file).write_bytes(observation["screenshot"])
                    trajectory.write(
                        json.dumps(
                            {
                                "step_num": step,
                                "action": action,
                                "response": response,
                                "reward": reward,
                                "done": is_done,
                                "info": info,
                                "screenshot_file": screenshot_file,
                            }
                        )
                        + "\n"
                    )
                    if is_done:
                        break
                if is_done:
                    break
        score = float(env.evaluate())
    finally:
        env.controller.end_recording(str(result_dir / "recording.mp4"))
    (result_dir / "result.txt").write_text(f"{score}\n", encoding="utf-8")
    return score


def run_tasks(
    args: argparse.Namespace,
    agent_class: type[Agent],
    agent_config: dict,
    tasks: list[tuple[str, str, dict]],
) -> list[float]:
    """Reuse one agent and environment to evaluate a worker's assigned tasks.

    Args:
        args: Parsed benchmark settings.
        agent_class: Agent subclass to instantiate with the configuration dictionary.
        agent_config: Agent settings loaded from JSON without modification.
        tasks: Domain, task ID, and task configuration triples assigned to this worker.

    Returns:
        One score per task, including zero for tasks that failed with an exception.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    from desktop_env.desktop_env import DesktopEnv

    agent = agent_class(agent_config)
    snapshot_name = "init_state"
    if args.provider_name == "aws":
        os.environ["AWS_REGION"] = args.aws_region
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        snapshot_name = IMAGE_ID_MAP[args.aws_region][(1920, 1080)]
    env = DesktopEnv(
        provider_name=args.provider_name,
        region=args.aws_region,
        path_to_vm=args.path_to_vm,
        snapshot_name=snapshot_name,
        action_space="pyautogui",
        headless=args.headless,
        require_a11y_tree=False,
        enable_proxy=args.enable_proxy,
        client_password=args.password,
    )
    try:
        scores = []
        for domain, task_id, task in tasks:
            result_dir = args.result_dir / domain / task_id
            logger.info("Running %s/%s", domain, task_id)
            try:
                score = run_task(agent, env, task, result_dir, args)
            except Exception as error:
                logger.exception("Task %s/%s failed", domain, task_id)
                result_dir.mkdir(parents=True, exist_ok=True)
                (result_dir / "error.txt").write_text(str(error) + "\n", encoding="utf-8")
                score = 0.0
            scores.append(score)
            logger.info("Score for %s/%s: %.4f", domain, task_id, score)
        return scores
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

    run_id = (
        args.agent_path + "_" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    args.result_dir = args.result_dir / run_id
    num_envs = min(args.num_envs, len(tasks))
    logger.info(
        "Running %d tasks in %d environments. Results: %s", len(tasks), num_envs, args.result_dir
    )
    with ProcessPoolExecutor(max_workers=num_envs, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(run_tasks, args, agent_class, agent_config, tasks[index::num_envs])
            for index in range(num_envs)
        ]
        scores = [score for future in futures for score in future.result()]
    logger.info("Average score: %.4f (%d tasks)", sum(scores) / len(scores), len(scores))


if __name__ == "__main__":
    main()
