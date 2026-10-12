"""Reproducible experiment allocation schedules using NumPy and Polars.

The allocation unit must match the unit of independent treatment assignment.
Repeated observations within a task or episode do not create independent units.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl


def normalize_ratio(arms: Sequence[str], ratio: Sequence[int] | None) -> list[str]:
    """Expand an allocation ratio into one complete-block template."""
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("Use at least two distinct treatment arms")
    weights = [1] * len(arms) if ratio is None else list(ratio)
    if len(weights) != len(arms) or any(
        isinstance(weight, bool) or not isinstance(weight, (int, np.integer)) or weight < 1
        for weight in weights
    ):
        raise ValueError("Allocation ratios must be positive integers per arm")
    return [arm for arm, weight in zip(arms, weights, strict=True) for _ in range(weight)]


def simple_randomization(
    n: int,
    arms: Sequence[str] = ("baseline", "candidate"),
    ratio: Sequence[int] | None = None,
    seed: int = 2026,
) -> pl.DataFrame:
    """Assign treatment arms independently; arm counts need not balance."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    template = normalize_ratio(arms, ratio)
    probabilities = np.array([template.count(arm) for arm in arms], dtype=float)
    probabilities /= probabilities.sum()
    assignments = np.random.default_rng(seed).choice(list(arms), size=n, p=probabilities).tolist()
    return pl.DataFrame(
        {"unit_id": list(range(1, n + 1)), "arm": assignments},
        schema={"unit_id": pl.Int64, "arm": pl.String},
    )


def block_randomization(
    n: int,
    arms: Sequence[str] = ("baseline", "candidate"),
    block_size: int | None = None,
    ratio: Sequence[int] | None = None,
    seed: int = 2026,
) -> pl.DataFrame:
    """Shuffle fixed-size blocks; completed blocks preserve the requested ratio."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    template = normalize_ratio(arms, ratio)
    size = len(template) * 2 if block_size is None else block_size
    if size < 1 or size % len(template):
        raise ValueError("block_size must be a positive multiple of sum(ratio)")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | str]] = []
    block = 0
    while len(rows) < n:
        labels = template * (size // len(template))
        rng.shuffle(labels)
        start = len(rows)
        rows.extend(
            {"unit_id": start + offset + 1, "block": block, "arm": arm}
            for offset, arm in enumerate(labels)
        )
        block += 1
    return pl.DataFrame(
        rows[:n],
        schema={"unit_id": pl.Int64, "block": pl.Int64, "arm": pl.String},
    )


def stratified_block_randomization(
    strata: dict[str, int] | Sequence[str],
    arms: Sequence[str] = ("baseline", "candidate"),
    block_size: int | None = None,
    ratio: Sequence[int] | None = None,
    seed: int = 2026,
) -> pl.DataFrame:
    """Randomize within each stratum, preserving existing unit positions if supplied."""
    if isinstance(strata, dict):
        if any(count < 0 for count in strata.values()):
            raise ValueError("Stratum sizes must be nonnegative")
        positions = {}
        next_id = 1
        for label, count in strata.items():
            positions[label] = list(range(next_id, next_id + count))
            next_id += count
    else:
        positions: dict[str, list[int]] = {}
        for position, label in enumerate(strata, start=1):
            positions.setdefault(label, []).append(position)

    rows: list[dict[str, int | str]] = []
    for index, (label, unit_ids) in enumerate(positions.items()):
        allocation = block_randomization(len(unit_ids), arms, block_size, ratio, seed + index)
        for unit_id, record in zip(unit_ids, allocation.iter_rows(named=True), strict=True):
            rows.append(
                {
                    "unit_id": unit_id,
                    "stratum": label,
                    "block": record["block"],
                    "arm": record["arm"],
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "unit_id": pl.Int64,
            "stratum": pl.String,
            "block": pl.Int64,
            "arm": pl.String,
        },
    ).sort("unit_id")


def cluster_randomization(
    clusters: int | Sequence[str],
    arms: Sequence[str] = ("baseline", "candidate"),
    ratio: Sequence[int] | None = None,
    block_size: int | None = None,
    seed: int = 2026,
) -> pl.DataFrame:
    """Assign entire task/application clusters, not individual observations."""
    if isinstance(clusters, int):
        if clusters < 0:
            raise ValueError("Number of clusters must be nonnegative")
        cluster_ids = [f"cluster_{index}" for index in range(1, clusters + 1)]
    else:
        cluster_ids = list(clusters)
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("Cluster IDs must be unique")
    allocation = block_randomization(len(cluster_ids), arms, block_size, ratio, seed)
    return allocation.with_columns(pl.Series("cluster_id", cluster_ids)).select(
        "cluster_id", "block", "arm"
    )


def assign_factorial_runs(design: pl.DataFrame, seed: int = 2026) -> pl.DataFrame:
    """Assign and sort a randomized global execution order."""
    order = (np.random.default_rng(seed).permutation(design.height) + 1).tolist()
    return (
        design.with_columns(pl.Series("run_order", order))
        .sort("run_order")
        .select("run_order", pl.exclude("run_order"))
    )


def arm_balance(
    allocation: pl.DataFrame, arm_col: str = "arm", by: str | None = None
) -> pl.DataFrame:
    """Return long-form arm counts, optionally grouped within a stratum or block."""
    groups = [arm_col] if by is None else [by, arm_col]
    return allocation.group_by(groups).agg(pl.len().alias("count")).sort(groups)
