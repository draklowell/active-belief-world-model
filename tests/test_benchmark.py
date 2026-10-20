import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, call

import mlflow
import pytest

from agents.base import Agent
from scripts import benchmark

AGENT_CONFIG = {
    "model": "qwen3-vl",
    "temperature": 0,
    "top_p": 0.9,
    "max_tokens": 32768,
    "stop_token": None,
    "coord": "relative",
    "add_thought_prefix": False,
    "actions": ["DONE"],
}


class DummyAgent(Agent):
    """Deterministic agent that returns configured actions."""

    def reset(self) -> None:
        """Reset task history."""

    def predict(self, instruction: str, observation: dict) -> tuple[str, list[str]]:
        """Return the configured response and actions.

        Args:
            instruction: Task instruction.
            observation: Current screenshot observation.

        Returns:
            Fixed response text and configured actions.
        """
        return "response", self.config["actions"]


@pytest.fixture
def settings(args: argparse.Namespace) -> benchmark.SettingsDict:
    """Provide benchmark settings in the current worker interface."""
    return cast(
        benchmark.SettingsDict,
        {
            "max_steps": args.max_steps,
            "sleep_after_execution": args.sleep_after_execution,
            "provider_name": args.provider_name,
            "aws_region": args.aws_region,
            "path_to_vm": args.path_to_vm,
            "headless": args.headless,
            "enable_proxy": args.enable_proxy,
            "password": args.password,
            "mlflow_tracking_uri": args.mlflow,
            "mlflow_enabled": args.mlflow != "disabled",
        },
    )


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    """Provide benchmark settings using temporary output paths."""
    return argparse.Namespace(
        agent_config=tmp_path / "agent.json",
        agent_path="tests.test_benchmark.DummyAgent",
        provider_name="docker",
        password="test-password",
        aws_region="us-east-1",
        num_envs=1,
        tasks=tmp_path / "tasks.json",
        headless=True,
        path_to_vm=None,
        max_steps=3,
        sleep_after_execution=0.25,
        enable_proxy=True,
        experiment_name=None,
        experiment_tags=[],
        mlflow="disabled",
    )


@pytest.fixture
def task() -> dict:
    """Provide a minimal task with a deterministic instruction."""
    return {"id": "task-1", "snapshot": "chrome", "instruction": "Complete the task"}


@pytest.fixture
def env() -> MagicMock:
    """Provide a desktop environment without starting a VM."""
    environment = MagicMock()
    environment.reset.return_value = {"screenshot": b"initial"}
    environment.step.return_value = ({"screenshot": b"final"}, 0, True, {})
    environment.evaluate.return_value = 1.0
    return environment


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the requested CLI names and leave model settings in JSON."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--agent-config",
            "agent.json",
            "--agent-path",
            "agents.example.ExampleAgent",
            "--provider-name",
            "aws",
            "--password",
            "secret",
            "--aws-region",
            "ap-east-1",
            "--num-envs",
            "2",
            "--tasks",
            "evaluation_examples/test.json",
            "--headless",
            "--path-to-vm",
            "machine.vmx",
            "--max-steps",
            "5",
            "--sleep-after-execution",
            "0.5",
            "--enable-proxy",
        ],
    )
    settings = benchmark.parse_args()
    assert settings.agent_config == Path("agent.json")
    assert settings.agent_path == "agents.example.ExampleAgent"
    assert settings.provider_name == "aws"
    assert settings.password == "secret"
    assert settings.aws_region == "ap-east-1"
    assert settings.num_envs == 2
    assert settings.tasks == Path("evaluation_examples/test.json")
    assert settings.headless is True
    assert settings.path_to_vm == "machine.vmx"
    assert settings.max_steps == 5
    assert settings.sleep_after_execution == 0.5
    assert settings.enable_proxy is True
    assert "model" not in vars(settings)
    assert "action_space" not in vars(settings)
    assert "observation_type" not in vars(settings)


@pytest.mark.parametrize(
    ("option", "value"),
    [("--num-envs", "0"), ("--max-steps", "0"), ("--sleep-after-execution", "-1")],
)
def test_parse_args_rejects_invalid_limits(
    monkeypatch: pytest.MonkeyPatch, option: str, value: str
) -> None:
    """Reject invalid execution limits before provisioning environments."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--agent-config",
            "a.json",
            "--agent-path",
            "a.A",
            "--tasks",
            "t.json",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit):
        benchmark.parse_args()


def test_load_agent_class() -> None:
    """Resolve agent subclasses from dotted import paths."""
    assert benchmark.load_agent_class("tests.test_benchmark.DummyAgent") is DummyAgent


@pytest.mark.parametrize("agent_path", ["pathlib.Path", "tests.test_benchmark.AGENT_CONFIG"])
def test_load_agent_class_rejects_other_objects(agent_path: str) -> None:
    """Reject classes and values that do not implement Agent."""
    with pytest.raises(TypeError, match="must inherit"):
        benchmark.load_agent_class(agent_path)


def test_load_agent_class_requires_dotted_path() -> None:
    """Require a module and class name."""
    with pytest.raises(ValueError, match="module.Class"):
        benchmark.load_agent_class("DummyAgent")


def test_run_task_tracks_actions_and_stops_when_done(
    args: argparse.Namespace, env: MagicMock, task: dict, settings: benchmark.SettingsDict
) -> None:
    """Preserve action steps and stop as soon as the environment marks the task complete."""
    agent = DummyAgent({"actions": ["pyautogui.click()", "DONE", "unused"]})
    env.step.side_effect = [
        ({"screenshot": b"clicked"}, 0, False, {}),
        ({"screenshot": b"done"}, 0, True, {"done": True}),
    ]
    mlflow.tracing.disable()
    result = benchmark.run_task(agent, env, task, settings)
    assert result["score"] == 1.0
    assert result["domain"] == "chrome"
    assert result["task_id"] == "task-1"
    assert result["steps"] == 2
    env.reset.assert_called_once_with(task_config=task)
    assert env.step.call_args_list == [
        call("pyautogui.click()", 0.25),
        call("DONE", 0.25),
    ]
    env.controller.start_recording.assert_called_once_with()
    env.controller.end_recording.assert_called_once()
    env.evaluate.assert_called_once_with()


def test_run_task_respects_prediction_limit(
    args: argparse.Namespace, env: MagicMock, task: dict, settings: benchmark.SettingsDict
) -> None:
    """Evaluate after the prediction budget is exhausted."""
    env.step.return_value = ({"screenshot": b"step"}, 0, False, {})
    agent = DummyAgent({"actions": ["WAIT"]})
    mlflow.tracing.disable()
    benchmark.run_task(agent, env, task, settings)
    assert env.step.call_count == args.max_steps
    env.evaluate.assert_called_once_with()


@pytest.mark.parametrize("failing_method", ["step", "evaluate"])
def test_run_task_ends_recording_on_error(
    args: argparse.Namespace,
    env: MagicMock,
    task: dict,
    settings: benchmark.SettingsDict,
    failing_method: str,
) -> None:
    """Close recordings without marking failed tasks complete."""
    getattr(env, failing_method).side_effect = RuntimeError("failed")
    mlflow.tracing.disable()
    with pytest.raises(RuntimeError, match="failed"):
        benchmark.run_task(DummyAgent(AGENT_CONFIG), env, task, settings)
    env.controller.end_recording.assert_called_once()


@pytest.mark.parametrize("provider_name", ["docker", "aws"])
def test_run_tasks_reuses_environment_and_continues_after_error(
    args: argparse.Namespace,
    env: MagicMock,
    task: dict,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    settings: benchmark.SettingsDict,
) -> None:
    """Reuse the VM and keep successful task results while logging task failures."""
    desktop_module = ModuleType("desktop_env.desktop_env")
    environment_class = MagicMock(return_value=env)
    setattr(desktop_module, "DesktopEnv", environment_class)
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", desktop_module)
    aws_module = ModuleType("desktop_env.providers.aws.manager")
    setattr(aws_module, "IMAGE_ID_MAP", {"us-east-1": {(1920, 1080): "test-ami"}})
    monkeypatch.setitem(sys.modules, "desktop_env.providers.aws.manager", aws_module)
    monkeypatch.setenv("AWS_REGION", "previous-region")
    args.provider_name = provider_name
    settings["provider_name"] = provider_name

    env.reset.side_effect = [RuntimeError("task setup failed"), {"screenshot": b"initial"}]
    task_results = benchmark.run_tasks(
        settings,
        DummyAgent,
        AGENT_CONFIG,
        [("chrome", "failed", task), ("chrome", "passed", task)],
        "test-experiment",
    )
    assert [result["score"] for result in task_results] == [1.0]
    environment_class.assert_called_once_with(
        provider_name=provider_name,
        region="us-east-1",
        path_to_vm=None,
        snapshot_name="test-ami" if provider_name == "aws" else "init_state",
        action_space="pyautogui",
        headless=True,
        require_a11y_tree=False,
        enable_proxy=True,
        client_password="test-password",
    )
    env.close.assert_called_once_with()


def test_main_distributes_tasks_across_workers(
    args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load configuration unchanged and assign each selected task to one worker."""
    args.agent_config.write_text(json.dumps(AGENT_CONFIG))
    args.tasks.write_text(json.dumps({"chrome": ["a", "b", "c"]}))
    task_dir = args.tasks.parent / "examples/chrome"
    task_dir.mkdir(parents=True)
    for task_id in ["a", "b", "c"]:
        (task_dir / f"{task_id}.json").write_text(json.dumps({"id": task_id, "snapshot": "chrome"}))
    args.num_envs = 2
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    executor_class = MagicMock()
    executor = executor_class.return_value.__enter__.return_value
    executor.submit.return_value.result.return_value = [
        {"domain": "chrome", "task_id": "a", "score": 1.0, "steps": 1}
    ]
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", executor_class)
    benchmark.main()
    assert executor_class.call_args.kwargs["max_workers"] == 2
    submissions = executor.submit.call_args_list
    assert len(submissions) == 2
    assert [task_id for _, task_id, _ in submissions[0].args[4]] == ["a", "c"]
    assert [task_id for _, task_id, _ in submissions[1].args[4]] == ["b"]
    assert submissions[0].args[1] is not None
    assert submissions[0].args[2] is DummyAgent
    assert submissions[0].args[3] == AGENT_CONFIG


def test_main_skips_environment_for_empty_manifest(
    args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish empty selections without allocating VMs or dividing by zero."""
    args.agent_config.write_text(json.dumps(AGENT_CONFIG))
    args.tasks.write_text("{}")
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    executor = MagicMock()
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", executor)
    benchmark.main()
    executor.assert_not_called()


def test_main_rejects_non_object_configuration(
    args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid agent configuration before creating workers."""
    args.agent_config.write_text("[]")
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="JSON object"):
        benchmark.main()
