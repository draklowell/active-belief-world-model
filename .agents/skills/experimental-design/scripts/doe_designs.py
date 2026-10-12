"""Reproducible factorial, screening, response-surface, and space-filling designs.

All design functions return Polars DataFrames in real factor units. Designs use
only NumPy and Polars; no experimental-design package is installed implicitly.
"""

from itertools import combinations, product
from math import ceil, log2, sqrt

import numpy as np
import polars as pl
from randomization import assign_factorial_runs


def decode_levels(coded: np.ndarray, factors: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Map coded coordinates in [-1, 1] to real factor values."""
    names = list(factors)
    data = {
        name: ((low + high) / 2 + coded[:, index] * (high - low) / 2).tolist()
        for index, (name, (low, high)) in enumerate(factors.items())
    }
    return pl.DataFrame(data, schema={name: pl.Float64 for name in names})


def numeric_ranges(factors: dict[str, tuple[float, float]]) -> None:
    """Check caller-supplied factor boundaries before generating a matrix."""
    if not factors or "run_order" in factors:
        raise ValueError("Provide factors without the reserved name run_order")
    if any(
        len(levels) != 2 or not np.isfinite(levels).all() or levels[0] >= levels[1]
        for levels in factors.values()
    ):
        raise ValueError("Every factor needs a finite (low, high) range")


def with_run_order(design: pl.DataFrame, randomize: bool, seed: int) -> pl.DataFrame:
    return assign_factorial_runs(design, seed) if randomize else design


def full_factorial(
    factors: dict[str, list[str | int | float]],
    randomize: bool = True,
    seed: int = 2026,
) -> pl.DataFrame:
    """Generate every combination of explicit factor levels."""
    if not factors or any(
        not levels or len(set(levels)) != len(levels) for levels in factors.values()
    ):
        raise ValueError("Every factor must have distinct, nonempty levels")
    names = list(factors)
    rows = [dict(zip(names, values, strict=True)) for values in product(*factors.values())]
    design = pl.DataFrame(rows, infer_schema_length=None)
    return with_run_order(design, randomize, seed)


def two_level_factorial(
    factors: dict[str, tuple[float, float]],
    randomize: bool = True,
    seed: int = 2026,
) -> pl.DataFrame:
    """Generate the complete 2**k design with estimable interactions."""
    numeric_ranges(factors)
    coded = np.array(list(product((-1.0, 1.0), repeat=len(factors))))
    return with_run_order(decode_levels(coded, factors), randomize, seed)


def fractional_factorial(
    factors: dict[str, tuple[float, float]],
    generator: str,
    randomize: bool = True,
    seed: int = 2026,
) -> pl.DataFrame:
    """Generate a regular two-level fraction using Yates notation.

    Example: generator='a b c abc' creates 8 runs for 4 named factors.
    The factor order follows the dictionary order; inspect aliasing before use.
    """
    numeric_ranges(factors)
    tokens = generator.lower().split()
    if len(tokens) != len(factors):
        raise ValueError("Generator must specify one token per named factor")
    bases = [token for token in tokens if len(token) == 1 and token.isalpha()]
    if not bases or len(set(bases)) != len(bases):
        raise ValueError("Generator needs unique independent one-letter factors")
    if any(not token.isalpha() or not set(token).issubset(bases) for token in tokens):
        raise ValueError("Generated factor tokens must reference independent bases")
    independent = np.array(list(product((-1.0, 1.0), repeat=len(bases))))
    columns = [
        np.prod(independent[:, [bases.index(letter) for letter in token]], axis=1)
        for token in tokens
    ]
    coded = np.column_stack(columns)
    return with_run_order(decode_levels(coded, factors), randomize, seed)


def hadamard(size: int) -> np.ndarray:
    """Build a Sylvester Hadamard matrix of power-of-two order."""
    matrix = np.array([[1]], dtype=int)
    while len(matrix) < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def paley_hadamard(q: int) -> np.ndarray:
    """Build a Paley type-I Hadamard matrix for q = 11 or 19."""
    residues = {value * value % q for value in range(1, q)}
    difference = np.fromfunction(lambda row, col: (col - row) % q, (q, q), dtype=int)
    character = np.where(difference == 0, 0, np.where(np.isin(difference, list(residues)), 1, -1))
    matrix = np.ones((q + 1, q + 1), dtype=int)
    matrix[1:, 1:] = character - np.eye(q, dtype=int)
    return matrix


def plackett_burman(
    factors: dict[str, tuple[float, float]],
    randomize: bool = True,
    seed: int = 2026,
) -> pl.DataFrame:
    """Generate a two-level orthogonal screening design with minimal supported runs.

    Supported run counts include powers of two and 12/20 via Paley construction.
    Main effects can be confounded with interactions.
    """
    numeric_ranges(factors)
    count = len(factors)
    options = [2 ** max(2, ceil(log2(count + 1))), 12, 20]
    size = min(option for option in options if option > count)
    matrix = paley_hadamard(size - 1) if size in (12, 20) else hadamard(size)
    coded = matrix[:, 1 : count + 1].astype(float)
    return with_run_order(decode_levels(coded, factors), randomize, seed)


def central_composite(
    factors: dict[str, tuple[float, float]],
    center: tuple[int, int] = (0, 3),
    alpha: str | float = "rotatable",
    face: str = "inscribed",
    randomize: bool = True,
    seed: int = 2026,
) -> pl.DataFrame:
    """Generate a CCD with factorial, axial and replicated center runs.

    Inscribed and faced designs remain within the supplied factor bounds.
    Circumscribed designs may exceed the specified limits.
    """
    numeric_ranges(factors)
    k = len(factors)
    if k < 2 or min(center) < 0 or face not in {"inscribed", "faced", "circumscribed"}:
        raise ValueError("CCD needs >=2 factors, nonnegative center counts, and a valid face")
    n_factorial = 2**k
    n_total = n_factorial + 2 * k + sum(center)
    if alpha == "rotatable":
        distance = n_factorial**0.25
    elif alpha == "orthogonal":
        distance = sqrt((sqrt(n_total * n_factorial) - n_factorial) / 2)
    else:
        distance = float(alpha)
    if distance <= 0:
        raise ValueError("alpha must be positive")
    scale = 1 / distance if face == "inscribed" else 1.0
    axial_distance = 1.0 if face in {"inscribed", "faced"} else distance
    factorial = np.array(list(product((-scale, scale), repeat=k)))
    axial = np.concatenate([np.eye(k) * axial_distance, -np.eye(k) * axial_distance])
    coded = np.vstack([factorial, np.zeros((center[0], k)), axial, np.zeros((center[1], k))])
    return with_run_order(decode_levels(coded, factors), randomize, seed)


def box_behnken(
    factors: dict[str, tuple[float, float]],
    center: int = 3,
    randomize: bool = True,
    seed: int = 2026,
) -> pl.DataFrame:
    """Generate a quadratic Box-Behnken design without extreme corner points."""
    numeric_ranges(factors)
    k = len(factors)
    if k < 3 or center < 0:
        raise ValueError("Box-Behnken requires at least 3 factors and nonnegative center")
    rows = []
    for first, second in combinations(range(k), 2):
        for first_value, second_value in product((-1.0, 1.0), repeat=2):
            row = [0.0] * k
            row[first], row[second] = first_value, second_value
            rows.append(row)
    rows.extend([[0.0] * k for unused in range(center)])
    return with_run_order(decode_levels(np.array(rows), factors), randomize, seed)


def latin_hypercube(
    factors: dict[str, tuple[float, float]],
    n_samples: int,
    criterion: str = "maximin",
    seed: int = 2026,
    randomize: bool = False,
) -> pl.DataFrame:
    """Generate a stratified Latin hypercube; maximize spread with bounded restarts."""
    numeric_ranges(factors)
    if n_samples < 1 or criterion not in {
        "random",
        "center",
        "maximin",
        "centermaximin",
        "correlation",
    }:
        raise ValueError("Use a positive sample size and a supported criterion")
    rng = np.random.default_rng(seed)
    candidates = 12 if criterion in {"maximin", "centermaximin", "correlation"} else 1
    best = np.empty((n_samples, len(factors)))
    best_score = float("-inf")
    for candidate_index in range(candidates):
        coordinates = np.column_stack(
            [
                (
                    rng.permutation(n_samples)
                    + (0.5 if "center" in criterion else rng.random(n_samples))
                )
                / n_samples
                for unused_dimension in range(len(factors))
            ]
        )
        if criterion == "correlation" and len(factors) > 1 and n_samples > 2:
            correlation = np.corrcoef(coordinates, rowvar=False)
            score = -float(np.max(np.abs(correlation - np.eye(len(factors)))))
        elif criterion in {"maximin", "centermaximin"} and n_samples > 1:
            distances = np.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(distances, np.inf)
            score = float(np.min(distances))
        else:
            score = 0.0
        if candidate_index == 0 or score > best_score:
            best, best_score = coordinates, score
    data = {
        name: (low + best[:, index] * (high - low)).tolist()
        for index, (name, (low, high)) in enumerate(factors.items())
    }
    return with_run_order(pl.DataFrame(data), randomize, seed)
