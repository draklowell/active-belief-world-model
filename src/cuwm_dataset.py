import ast
import json
import math
from collections.abc import Iterable
from itertools import accumulate
from pathlib import Path

import polars as pl
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

from agents.qwen3vl_common import build_computer_use_tool
from agents.qwen3vl_cuwm import RELATIVE_COORDINATE_SPACE, build_option_schema

COORDINATE_SCALE = 999
POINTER_ACTIONS = {
    "pyautogui.click": "left_click",
    "pyautogui.doubleClick": "double_click",
    "pyautogui.rightClick": "right_click",
    "pyautogui.moveTo": "mouse_move",
}
TASK_PREFIX_PATTERN = r"^\s*task\s*\d+\s*[:：]\s*"
TOKEN_PATTERN = r"\w+"
OVERLAP_TEXT_COLUMNS = ["instruction", "natural_language_task", "actual_task"]
MIN_CONTENT_TOKENS = 3
EXACT_MATCH_MIN_CHARS = 30
STOPWORDS = (
    "a an the to of in on for and or my me i you it is be can could please help want would "
    "like this that with from as at by into all your our we us do does how what which some any "
    "its are then also there their them these those so if not no new current"
).split()


def parse_calls(code: str) -> list[tuple[str, list, dict]] | None:
    """Parse AgentNet action code into function calls with literal arguments, without executing it.

    Args:
        code: Python source of one AgentNet step, one call per statement.

    Returns:
        Called function name, positional arguments and keyword arguments of each statement, or
        None when the code is not a sequence of calls with literal arguments.
    """
    try:
        statements = ast.parse(code).body
    except SyntaxError:
        return None
    calls = []
    for statement in statements:
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
            return None
        call = statement.value
        try:
            positional = [ast.literal_eval(argument) for argument in call.args]
            keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        except ValueError:
            return None
        calls.append((ast.unparse(call.func), positional, keywords))
    return calls


def relative_coordinate(keywords: dict) -> list[int]:
    """Map fractional AgentNet pointer coordinates to the relative grid of the Qwen3-VL agent.

    The scale is the inverse of `adjust_coordinates`, which maps a relative coordinate to native
    pixels by multiplying it with the screen size divided by 999.

    Args:
        keywords: Call keywords holding `x` and `y` as fractions of the screen width and height.

    Returns:
        Relative x and y coordinates between 0 and 999.
    """
    return [round(keywords["x"] * COORDINATE_SCALE), round(keywords["y"] * COORDINATE_SCALE)]


def convert_code_to_arguments(code: str) -> dict | None:
    """Convert the action code of one AgentNet step into `computer_use` tool call arguments.

    Clicks, pointer moves, typing, key presses, hotkeys, waits and terminations map to one action.
    Pointer moves followed by scrolls become one scroll at the last pointer position with the
    summed amount. Drags and triple clicks have no equivalent in the action space and, like any
    other call combination, are unsupported.

    Args:
        code: Python source of one AgentNet step.

    Returns:
        Tool call arguments including the `action` name, or None for unsupported code.
    """
    calls = parse_calls(code)
    if not calls:
        return None
    names = [name for name, _, _ in calls]
    moves = [keywords for name, _, keywords in calls if name == "pyautogui.moveTo"]
    scrolls = [positional[0] for name, positional, _ in calls if name == "pyautogui.scroll"]
    if scrolls and len(moves) + len(scrolls) == len(calls):
        if names.index("pyautogui.scroll") != len(moves):
            return None
        arguments: dict = {"action": "scroll"}
        if moves:
            arguments["coordinate"] = relative_coordinate(moves[-1])
        arguments["pixels"] = sum(scrolls)
        return arguments
    if len(calls) != 1:
        return None
    name, positional, keywords = calls[0]
    if name in POINTER_ACTIONS:
        return {"action": POINTER_ACTIONS[name], "coordinate": relative_coordinate(keywords)}
    match name:
        case "pyautogui.write":
            return {"action": "type", "text": keywords["message"]}
        case "pyautogui.press":
            return {"action": "key", "keys": [positional[0]]}
        case "pyautogui.hotkey":
            return {"action": "key", "keys": list(positional[0])}
        case "computer.wait":
            return {"action": "wait"}
        case "computer.terminate":
            return {"action": "terminate", "status": keywords["status"]}
    return None


def build_action_validator() -> Validator:
    """Build the validator of candidate actions used by the Qwen3-VL CUWM agent.

    Returns:
        Validator of an option with an `action` description and a relative `computer_use` call.
    """
    computer_use_tool = build_computer_use_tool("relative", *RELATIVE_COORDINATE_SPACE)
    return Draft202012Validator(build_option_schema(computer_use_tool, RELATIVE_COORDINATE_SPACE))


def normalize_text(text: pl.Expr) -> pl.Expr:
    """Normalize instructions for matching by Unicode form, case, task prefix and punctuation.

    Args:
        text: String expression holding an instruction.

    Returns:
        Lowercase word tokens joined by single spaces, without a leading `TaskN:` prefix.
    """
    return (
        text.str.normalize("NFKC")
        .str.to_lowercase()
        .str.replace(TASK_PREFIX_PATTERN, "")
        .str.extract_all(TOKEN_PATTERN)
        .list.join(" ")
    )


def load_osworld_instructions(manifest_path: Path) -> pl.DataFrame:
    """Load the instructions of the OSWorld tasks listed in a task manifest.

    Args:
        manifest_path: OSWorld manifest mapping each domain to task ids, next to `examples/`.

    Returns:
        One row per task with its task_id, domain and instruction.
    """
    manifest = json.loads(manifest_path.read_text())
    examples_dir = manifest_path.parent / "examples"
    return pl.DataFrame(
        [
            {
                "task_id": task_id,
                "domain": domain,
                "instruction": json.loads((examples_dir / domain / f"{task_id}.json").read_text())[
                    "instruction"
                ],
            }
            for domain, task_ids in manifest.items()
            for task_id in task_ids
        ]
    )


def find_osworld_overlap(
    tasks: pl.DataFrame, osworld_tasks: pl.DataFrame, threshold: float
) -> pl.DataFrame:
    """Find AgentNet tasks whose instructions copy or nearly copy an OSWorld instruction.

    Every instruction field of an AgentNet task is compared with every OSWorld instruction after
    normalization. A pair matches exactly when the texts are equal or the OSWorld text of at least
    30 characters appears verbatim. Otherwise the containment is the share of OSWorld content
    tokens, words outside a stopword list, that appear in the AgentNet text. OSWorld instructions
    with fewer than three content tokens only match exactly.

    Args:
        tasks: AgentNet tasks with task_id and the instruction, natural_language_task and
            actual_task text columns.
        osworld_tasks: OSWorld tasks with task_id and instruction.
        threshold: Minimum containment of a near match.

    Returns:
        One row per matching AgentNet task with its best OSWorld match, whether it is exact, and
        the containment, sorted by task_id.
    """
    texts = (
        tasks.unpivot(OVERLAP_TEXT_COLUMNS, index="task_id", value_name="text")
        .drop_nulls("text")
        .select("task_id", normalize_text(pl.col("text")).alias("text"))
        .unique()
        .with_row_index("text_id")
    )
    osworld_texts = osworld_tasks.select(
        pl.col("task_id").alias("osworld_id"),
        normalize_text(pl.col("instruction")).alias("osworld_text"),
    )
    content_tokens = (
        osworld_texts.select(
            "osworld_id", pl.col("osworld_text").str.extract_all(TOKEN_PATTERN).alias("token")
        )
        .explode("token")
        .unique()
        .filter(~pl.col("token").is_in(STOPWORDS))
        .with_columns(pl.len().over("osworld_id").alias("content_tokens"))
        .filter(pl.col("content_tokens") >= MIN_CONTENT_TOKENS)
    )
    near_pairs = (
        texts.select("text_id", pl.col("text").str.extract_all(TOKEN_PATTERN).alias("token"))
        .explode("token")
        .unique()
        .join(content_tokens, on="token")
        .group_by("text_id", "osworld_id")
        .agg((pl.len() / pl.col("content_tokens").first()).alias("containment"))
    )
    equal_pairs = texts.join(osworld_texts, left_on="text", right_on="osworld_text").select(
        "text_id", "osworld_id", pl.lit(1.0).alias("containment")
    )
    return (
        pl.concat([near_pairs, equal_pairs])
        .join(texts, on="text_id")
        .join(osworld_texts, on="osworld_id")
        .with_columns(
            (
                (pl.col("text") == pl.col("osworld_text"))
                | (
                    (pl.col("osworld_text").str.len_chars() >= EXACT_MATCH_MIN_CHARS)
                    & pl.col("text").str.contains(pl.col("osworld_text"), literal=True)
                )
            ).alias("is_exact")
        )
        .filter(pl.col("is_exact") | (pl.col("containment") >= threshold))
        .sort(["is_exact", "containment", "osworld_id"], descending=[True, True, False])
        .group_by("task_id", maintain_order=True)
        .first()
        .select("task_id", "osworld_id", "is_exact", "containment")
        .sort("task_id")
    )


def build_transitions(
    trajectories: pl.LazyFrame, image_index: pl.DataFrame, excluded_task_ids: Iterable[str]
) -> pl.DataFrame:
    """Pair every AgentNet step with the next screenshot and label why a transition is excluded.

    A step forms a transition with the step whose index follows it directly. Actions are converted
    to `computer_use` arguments and validated with the agent's option schema. Screenshots are
    compared through the CRC32 and size of the archive index, so identical frames are found
    without downloading them. Exclusion reasons are assigned in this order: `osworld_overlap`,
    `last_step`, `non_consecutive_index`, `missing_image`, `unsupported_action`, `invalid_action`,
    `terminate`, `unchanged_screenshot`, and `duplicate_transition` for repeats of an earlier
    eligible screenshot and action.

    Args:
        trajectories: AgentNet trajectories with task fields and a `traj` list column.
        image_index: Archive index with name, crc32 and uncompressed_bytes of every screenshot.
        excluded_task_ids: Tasks excluded as a whole because they overlap OSWorld.

    Returns:
        One row per step with task fields, both screenshots, the original code, the JSON tool
        arguments, the action description, the AgentNet observation of the next step, the judge
        status and a null or exclusion reason.
    """
    steps = (
        trajectories.select("task_id", "domain", "instruction", "task_completed", "traj")
        .explode("traj")
        .unnest("traj")
        .unnest("value")
        .select(
            "task_id",
            "domain",
            "instruction",
            "task_completed",
            "index",
            "image",
            "code",
            pl.col("action").alias("gui_description"),
            "reflection",
            "observation",
        )
        .sort("task_id", "index")
        .with_columns(
            pl.col("index").shift(-1).over("task_id").alias("next_index"),
            pl.col("image").shift(-1).over("task_id").alias("next_image"),
            pl.col("observation").shift(-1).over("task_id").alias("next_observation"),
        )
        .drop("observation")
        .collect()
    )
    validator = build_action_validator()
    arguments = [convert_code_to_arguments(code) for code in steps["code"]]
    is_valid_action = [
        argument is not None
        and validator.is_valid(
            {
                "action": description,
                "tool_call": {"name": "computer_use", "arguments": argument},
            }
        )
        for argument, description in zip(arguments, steps["gui_description"])
    ]
    checksums = image_index.select("name", "crc32", "uncompressed_bytes")
    reason = (
        pl.when(pl.col("task_id").is_in(list(excluded_task_ids)))
        .then(pl.lit("osworld_overlap"))
        .when(pl.col("next_index").is_null())
        .then(pl.lit("last_step"))
        .when(pl.col("next_index") != pl.col("index") + 1)
        .then(pl.lit("non_consecutive_index"))
        .when(pl.col("image_crc32").is_null() | pl.col("next_image_crc32").is_null())
        .then(pl.lit("missing_image"))
        .when(pl.col("tool_arguments").is_null())
        .then(pl.lit("unsupported_action"))
        .when(~pl.col("is_valid_action"))
        .then(pl.lit("invalid_action"))
        .when(pl.col("action_name") == "terminate")
        .then(pl.lit("terminate"))
        .when(
            (pl.col("image_crc32") == pl.col("next_image_crc32"))
            & (pl.col("image_bytes") == pl.col("next_image_bytes"))
        )
        .then(pl.lit("unchanged_screenshot"))
    )
    is_repeat = (
        ~pl.struct("image_crc32", "image_bytes", "tool_arguments")
        .is_first_distinct()
        .over(pl.col("exclusion_reason").is_null())
    )
    return (
        steps.with_columns(
            pl.Series(
                "tool_arguments",
                [json.dumps(argument) if argument else None for argument in arguments],
                dtype=pl.String,
            ),
            pl.Series("is_valid_action", is_valid_action),
            pl.when(pl.col("task_completed").is_null())
            .then(pl.lit("unscored"))
            .when(pl.col("task_completed"))
            .then(pl.lit("completed"))
            .otherwise(pl.lit("failed"))
            .alias("judge_status"),
        )
        .with_columns(pl.col("tool_arguments").str.json_path_match("$.action").alias("action_name"))
        .join(
            checksums.rename(
                {"name": "image", "crc32": "image_crc32", "uncompressed_bytes": "image_bytes"}
            ),
            on="image",
            how="left",
        )
        .join(
            checksums.rename(
                {
                    "name": "next_image",
                    "crc32": "next_image_crc32",
                    "uncompressed_bytes": "next_image_bytes",
                }
            ),
            on="next_image",
            how="left",
        )
        .sort("task_id", "index")
        .with_columns(reason.alias("exclusion_reason"))
        .with_columns(
            pl.when(pl.col("exclusion_reason").is_null() & is_repeat)
            .then(pl.lit("duplicate_transition"))
            .otherwise(pl.col("exclusion_reason"))
            .alias("exclusion_reason")
        )
        .drop("task_completed", "next_index", "is_valid_action")
    )


def assign_splits(tasks: pl.DataFrame, fractions: dict[str, float], seed: int) -> pl.DataFrame:
    """Assign tasks to splits, stratified by domain and grouped by normalized instruction.

    Instruction groups are shuffled with the seed inside each domain and cut at the cumulative
    fractions, so tasks sharing a normalized instruction never land in different splits.

    Args:
        tasks: AgentNet tasks with task_id, domain and instruction.
        fractions: Ordered mapping of split names to their share of instruction groups, summing
            to one.
        seed: Seed of the shuffle.

    Returns:
        One row per task with its task_id and split.
    """
    keyed = tasks.select(
        "task_id", "domain", normalize_text(pl.col("instruction")).alias("instruction_key")
    )
    breaks = accumulate(list(fractions.values())[:-1])
    groups = (
        keyed.select("domain", "instruction_key")
        .unique()
        .sort("domain", "instruction_key")
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .with_columns(
            (pl.int_range(pl.len()).over("domain") / pl.len().over("domain")).alias("position")
        )
        .with_columns(
            pl.sum_horizontal(pl.lit(0), *[pl.col("position") >= bound for bound in breaks])
            .replace_strict(dict(enumerate(fractions)), return_dtype=pl.String)
            .alias("split")
        )
    )
    return keyed.join(groups, on=["domain", "instruction_key"]).select("task_id", "split")


def allocate_quota(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split a sample size across domains in proportion to their weights.

    The largest remainder method keeps the quotas integral and summing to the total.

    Args:
        total: Number of transitions to allocate.
        weights: Nonnegative weight of every domain.

    Returns:
        Number of transitions of every domain.
    """
    weight_sum = sum(weights.values())
    shares = {domain: total * weight / weight_sum for domain, weight in weights.items()}
    quota = {domain: math.floor(share) for domain, share in shares.items()}
    remainder = total - sum(quota.values())
    for domain in sorted(shares, key=lambda name: shares[name] - quota[name], reverse=True)[
        :remainder
    ]:
        quota[domain] += 1
    return quota


def sample_transitions(transitions: pl.DataFrame, quotas: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Draw a seeded uniform sample of eligible transitions for every split and domain.

    Args:
        transitions: Transitions with split, domain and exclusion_reason columns.
        quotas: Number of transitions to draw, with split, domain and quota columns.
        seed: Seed of the shuffle.

    Returns:
        Sampled eligible transitions, at most the quota of each split and domain, sorted by split,
        task_id and index.
    """
    return (
        transitions.filter(pl.col("exclusion_reason").is_null())
        .sort("task_id", "index")
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .with_columns(pl.int_range(pl.len()).over("split", "domain").alias("rank"))
        .join(quotas, on=["split", "domain"])
        .filter(pl.col("rank") < pl.col("quota"))
        .drop("rank", "quota")
        .sort("split", "task_id", "index")
    )
