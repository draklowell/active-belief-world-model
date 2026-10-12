"""Regression tests for standalone experimental-design helpers."""

import numpy as np
import polars as pl
from doe_designs import (
    box_behnken,
    central_composite,
    fractional_factorial,
    full_factorial,
    latin_hypercube,
    plackett_burman,
    two_level_factorial,
)
from randomization import (
    arm_balance,
    block_randomization,
    cluster_randomization,
    simple_randomization,
    stratified_block_randomization,
)


def test_simple_randomization_reproducible() -> None:
    first = simple_randomization(30, seed=2026)
    second = simple_randomization(30, seed=2026)
    assert first.equals(second)
    assert first.get_column("unit_id").to_list() == list(range(1, 31))


def test_block_randomization_balances_complete_blocks() -> None:
    schedule = block_randomization(24, block_size=4, seed=2026)
    groups = arm_balance(schedule, by="block")
    assert groups.get_column("count").to_list() == [2] * 12
    assert schedule.get_column("unit_id").to_list() == list(range(1, 25))


def test_stratified_allocation_preserves_original_positions() -> None:
    schedule = stratified_block_randomization(
        ["browser", "editor", "browser", "settings", "editor"], seed=2026
    )
    assert schedule.get_column("unit_id").to_list() == [1, 2, 3, 4, 5]
    assert schedule.get_column("stratum").to_list() == [
        "browser",
        "editor",
        "browser",
        "settings",
        "editor",
    ]


def test_cluster_allocation_retains_ids() -> None:
    schedule = cluster_randomization(["app_a", "app_b", "app_c"], seed=2026)
    assert schedule.get_column("cluster_id").to_list() == ["app_a", "app_b", "app_c"]


def test_full_factorial() -> None:
    design = full_factorial({"modality": ["text", "image"], "belief": ["off", "on"]})
    assert design.height == 4
    assert design.get_column("run_order").to_list() == [1, 2, 3, 4]


def test_two_level_factorial() -> None:
    design = two_level_factorial({"horizon": (1, 4), "threshold": (0.2, 0.8)}, randomize=False)
    assert design.height == 4
    assert design.get_column("horizon").unique().sort().to_list() == [1.0, 4.0]


def test_fractional_factorial_generator() -> None:
    factors = {name: (0.0, 1.0) for name in "abcd"}
    matrix = fractional_factorial(factors, "a b c abc", randomize=False)
    assert matrix.height == 8
    coded = matrix.select(list(factors)).to_numpy() * 2 - 1
    assert np.array_equal(coded[:, 3], np.prod(coded[:, :3], axis=1))


def test_screening_designs_are_orthogonal() -> None:
    for count, expected_runs in [(5, 8), (11, 12), (17, 20), (25, 32)]:
        design = plackett_burman(
            {f"factor_{index}": (0.0, 1.0) for index in range(count)},
            randomize=False,
        )
        coded = design.to_numpy() * 2 - 1
        assert design.height == expected_runs
        assert np.allclose(coded.T @ coded, np.eye(count) * expected_runs)
        assert np.allclose(coded.sum(axis=0), 0)


def test_response_surface_designs_stay_inside_bounds() -> None:
    ccd = central_composite({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    assert ccd.height == 11
    assert ccd.select(pl.all().exclude("run_order")).to_numpy().min() >= 0
    assert ccd.select(pl.all().exclude("run_order")).to_numpy().max() <= 1
    bb = box_behnken({"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)})
    assert bb.height == 15
    coded = bb.select("a", "b", "c").to_numpy() * 2 - 1
    assert np.all(np.count_nonzero(coded, axis=1) <= 2)


def test_latin_hypercube_stratification() -> None:
    n = 12
    design = latin_hypercube({"a": (0.0, 1.0), "b": (0.0, 1.0)}, n_samples=n, seed=2026)
    for column in design.columns:
        bins = np.floor(design.get_column(column).to_numpy() * n).astype(int)
        assert sorted(bins.tolist()) == list(range(n))
