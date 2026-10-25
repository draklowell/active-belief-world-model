import polars as pl
import pytest

from agents.qwen3vl_common import convert_tool_call_to_pyautogui
from cuwm_dataset import (
    allocate_quota,
    assign_splits,
    build_action_validator,
    build_transitions,
    convert_code_to_arguments,
    find_osworld_overlap,
    sample_transitions,
)

SCREEN_SIZE = (1920, 1080)
PROCESSED_SIZE = (1920, 1088)


@pytest.mark.parametrize(
    ("code", "arguments"),
    [
        ("pyautogui.click(x=0.5, y=0.25)", {"action": "left_click", "coordinate": [500, 250]}),
        ("pyautogui.doubleClick(x=0, y=1.0)", {"action": "double_click", "coordinate": [0, 999]}),
        ("pyautogui.rightClick(x=0.1, y=0.2)", {"action": "right_click", "coordinate": [100, 200]}),
        ("pyautogui.moveTo(x=0.394, y=0.604)", {"action": "mouse_move", "coordinate": [394, 603]}),
        ("pyautogui.write(message='a\\nb')", {"action": "type", "text": "a\nb"}),
        ("pyautogui.press('pagedown')", {"action": "key", "keys": ["pagedown"]}),
        ("pyautogui.hotkey(['ctrl', 'o'])", {"action": "key", "keys": ["ctrl", "o"]}),
        ("pyautogui.scroll(-8)", {"action": "scroll", "pixels": -8}),
        (
            "pyautogui.moveTo(x=0.1, y=0.1)\npyautogui.moveTo(x=0.072, y=0.506)\n"
            "pyautogui.scroll(-4)\npyautogui.scroll(-2)",
            {"action": "scroll", "coordinate": [72, 505], "pixels": -6},
        ),
        ("computer.wait()", {"action": "wait"}),
        ("computer.terminate(status='failure')", {"action": "terminate", "status": "failure"}),
    ],
)
def test_convert_code_to_arguments(code: str, arguments: dict) -> None:
    """Check the mapping of every supported AgentNet call to `computer_use` arguments."""
    assert convert_code_to_arguments(code) == arguments


@pytest.mark.parametrize(
    "code",
    [
        "pyautogui.moveTo(x=0.078, y=0.381)\npyautogui.dragTo(x=0.085, y=0.361, button='left')",
        "computer.tripleClick(x=0.226, y=0.311)",
        "pyautogui.click(x=0.1, y=0.1)\npyautogui.click(x=0.2, y=0.2)",
        "pyautogui.scroll(-2)\npyautogui.moveTo(x=0.1, y=0.1)",
        "pyautogui.click(x=position, y=0.1)",
        "x = 1",
        "pyautogui.click(",
        "",
    ],
)
def test_convert_code_to_arguments_rejects_unsupported_code(code: str) -> None:
    """Check that drags, triple clicks, call combinations and non-literal code are unsupported."""
    assert convert_code_to_arguments(code) is None


@pytest.mark.parametrize("fraction", [0.0, 0.018, 0.5, 0.981, 1.0])
def test_relative_coordinates_round_trip_to_native_pixels(fraction: float) -> None:
    """Check that the agent's PyAutoGUI conversion maps converted coordinates back to the screen."""
    arguments = convert_code_to_arguments(f"pyautogui.click(x={fraction}, y={fraction})")
    assert arguments is not None
    command = convert_tool_call_to_pyautogui(arguments, "relative", SCREEN_SIZE, PROCESSED_SIZE)
    x, y = (int(value) for value in command[0].removeprefix("pyautogui.click(")[:-1].split(","))
    assert abs(x - fraction * SCREEN_SIZE[0]) <= 2
    assert abs(y - fraction * SCREEN_SIZE[1]) <= 2


def test_action_validator_uses_agent_schema() -> None:
    """Check that converted actions pass the agent schema and out-of-screen points fail it."""
    validator = build_action_validator()
    option = {
        "action": "Click the button.",
        "tool_call": {
            "name": "computer_use",
            "arguments": convert_code_to_arguments("pyautogui.click(x=0.5, y=0.5)"),
        },
    }
    assert validator.is_valid(option)
    option["tool_call"]["arguments"] = convert_code_to_arguments("pyautogui.click(x=0.9, y=-0.01)")
    assert not validator.is_valid(option)


def step(index: int, image: str, code: str) -> dict:
    """Build one AgentNet trajectory step with an action description."""
    value = {"code": code, "action": f"Do step {index}.", "reflection": None}
    return {"index": index, "image": image, "value": value}


def test_build_transitions_assigns_exclusion_reasons() -> None:
    """Check pairing, the order of exclusion reasons and duplicate detection."""
    click = "pyautogui.click(x=0.5, y=0.5)"
    trajectories = pl.LazyFrame(
        [
            {
                "task_id": "a",
                "domain": "Chrome",
                "instruction": "Open a tab.",
                "task_completed": True,
                "traj": [
                    step(0, "a0", click),
                    step(1, "a1", click),
                    step(2, "a1", "computer.tripleClick(x=0.1, y=0.1)"),
                    step(3, "a2", "computer.terminate(status='success')"),
                    step(5, "a3", click),
                    step(6, "missing", click),
                    step(7, "a4", "computer.terminate(status='success')"),
                ],
            },
            {
                "task_id": "b",
                "domain": "Chrome",
                "instruction": "Open a tab again.",
                "task_completed": None,
                "traj": [step(0, "b0", click), step(1, "b1", click)],
            },
            {
                "task_id": "c",
                "domain": "OS",
                "instruction": "Copied task.",
                "task_completed": False,
                "traj": [step(0, "c0", click), step(1, "c1", click)],
            },
        ]
    )
    image_index = pl.DataFrame(
        {
            "name": ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "c0", "c1"],
            "crc32": [1, 2, 3, 4, 5, 1, 6, 7, 8],
            "uncompressed_bytes": [10, 20, 30, 40, 50, 10, 60, 70, 80],
        }
    )
    transitions = build_transitions(trajectories, image_index, ["c"])
    reasons = dict(
        zip(
            transitions["task_id"] + "/" + transitions["index"].cast(pl.String),
            transitions["exclusion_reason"],
        )
    )
    assert reasons == {
        "a/0": None,
        "a/1": "unchanged_screenshot",
        "a/2": "unsupported_action",
        "a/3": "non_consecutive_index",
        "a/5": "missing_image",
        "a/6": "missing_image",
        "a/7": "last_step",
        "b/0": "duplicate_transition",
        "b/1": "last_step",
        "c/0": "osworld_overlap",
        "c/1": "osworld_overlap",
    }
    first = transitions.row(0, named=True)
    assert first["next_image"] == "a1"
    assert first["tool_arguments"] == '{"action": "left_click", "coordinate": [500, 500]}'
    assert first["gui_description"] == "Do step 0."
    assert transitions["judge_status"].unique().sort().to_list() == [
        "completed",
        "failed",
        "unscored",
    ]


def test_build_transitions_excludes_terminate_with_next_step() -> None:
    """Check that a termination followed by another step is not a transition to simulate."""
    trajectories = pl.LazyFrame(
        [
            {
                "task_id": "a",
                "domain": "OS",
                "instruction": "Finish.",
                "task_completed": True,
                "traj": [
                    step(0, "a0", "computer.terminate(status='success')"),
                    step(1, "a1", "computer.wait()"),
                ],
            }
        ]
    )
    image_index = pl.DataFrame(
        {"name": ["a0", "a1"], "crc32": [1, 2], "uncompressed_bytes": [1, 2]}
    )
    transitions = build_transitions(trajectories, image_index, [])
    assert transitions["exclusion_reason"].to_list() == ["terminate", "last_step"]


def overlap_tasks(*instructions: str | None) -> pl.DataFrame:
    """Build AgentNet tasks with one instruction each and no rewritten task fields."""
    return pl.DataFrame(
        {
            "task_id": [f"t{i}" for i in range(len(instructions))],
            "instruction": list(instructions),
            "natural_language_task": [None] * len(instructions),
            "actual_task": [None] * len(instructions),
        },
        schema={
            "task_id": pl.String,
            "instruction": pl.String,
            "natural_language_task": pl.String,
            "actual_task": pl.String,
        },
    )


def test_find_osworld_overlap_matches_copies_and_near_copies() -> None:
    """Check exact copies with a task prefix, near copies, and stopword-only similarity."""
    osworld_tasks = pl.DataFrame(
        {
            "task_id": ["exact", "near", "short"],
            "instruction": [
                "Change the first row of table to T1, T2, T3 on slide 4.",
                "Please help me install the autoDocstring extension in VS Code.",
                "Can you help me?",
            ],
        }
    )
    tasks = overlap_tasks(
        "Task1：Change the first row of table to T1, T2, T3 on slide 4.",
        "Install the autoDocstring extension in VS Code and reload the window.",
        "Can you help me set the mouse speed? I want it to be fast.",
        "Can you help me?",
    )
    overlap = find_osworld_overlap(tasks, osworld_tasks, threshold=0.8)
    assert overlap.to_dicts() == [
        {"task_id": "t0", "osworld_id": "exact", "is_exact": True, "containment": 1.0},
        {"task_id": "t1", "osworld_id": "near", "is_exact": False, "containment": 1.0},
        {"task_id": "t3", "osworld_id": "short", "is_exact": True, "containment": 1.0},
    ]


def test_find_osworld_overlap_uses_every_instruction_field() -> None:
    """Check that rewritten task fields are compared as well as the instruction."""
    osworld_tasks = pl.DataFrame(
        {"task_id": ["os"], "instruction": ["Turn on Bluetooth in the system settings."]}
    )
    tasks = overlap_tasks("Do something unrelated.").with_columns(
        pl.lit("Open settings and turn on bluetooth system wide.").alias("actual_task")
    )
    overlap = find_osworld_overlap(tasks, osworld_tasks, threshold=0.8)
    assert overlap["task_id"].to_list() == ["t0"]
    assert overlap["containment"].to_list() == [pytest.approx(1.0)]


def test_assign_splits_is_deterministic_and_keeps_instruction_groups_together() -> None:
    """Check reproducibility, split shares and that duplicate instructions share a split."""
    tasks = pl.DataFrame(
        {
            "task_id": [f"t{i}" for i in range(400)] + ["copy"],
            "domain": ["Chrome", "OS"] * 200 + ["Chrome"],
            "instruction": [f"Instruction {i}" for i in range(400)] + ["TASK2: instruction 0!"],
        }
    )
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    splits = assign_splits(tasks, fractions, seed=2026)
    assert splits.sort("task_id").equals(assign_splits(tasks, fractions, seed=2026).sort("task_id"))
    split_of = dict(splits.iter_rows())
    assert split_of["copy"] == split_of["t0"]
    counts = dict(splits.filter(pl.col("task_id") != "copy").group_by("split").len().iter_rows())
    assert counts == {"train": 320, "validation": 40, "test": 40}


def test_allocate_quota_is_proportional_and_exact() -> None:
    """Check that quotas sum to the total and follow the weights."""
    quota = allocate_quota(10, {"Chrome": 2, "OS": 1, "VLC": 1})
    assert quota == {"Chrome": 5, "OS": 3, "VLC": 2}
    assert sum(allocate_quota(2876, {"a": 46, "b": 26, "c": 101, "d": 15}).values()) == 2876


def test_sample_transitions_respects_quotas_and_eligibility() -> None:
    """Check that sampling draws only eligible transitions up to each quota, reproducibly."""
    transitions = pl.DataFrame(
        {
            "task_id": [f"t{i}" for i in range(12)],
            "index": [0] * 12,
            "split": ["train"] * 8 + ["test"] * 4,
            "domain": ["Chrome", "OS"] * 6,
            "exclusion_reason": [None] * 10 + ["last_step"] * 2,
        },
        schema_overrides={"exclusion_reason": pl.String},
    )
    quotas = pl.DataFrame(
        {
            "split": ["train", "train", "test", "test"],
            "domain": ["Chrome", "OS", "Chrome", "OS"],
            "quota": [3, 1, 5, 0],
        }
    )
    sample = sample_transitions(transitions, quotas, seed=2026)
    assert sample.equals(sample_transitions(transitions, quotas, seed=2026))
    assert sample["exclusion_reason"].null_count() == sample.height
    counts = dict(
        sample.group_by(pl.concat_str("split", "domain", separator="/")).len().iter_rows()
    )
    assert counts == {"train/Chrome": 3, "train/OS": 1, "test/Chrome": 1}
