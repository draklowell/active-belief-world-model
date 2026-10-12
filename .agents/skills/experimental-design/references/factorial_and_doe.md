# Factorial and Design-of-Experiments Methods for ML

Full factorial plans reveal interactions between research factors, while
reduced plans trade identifiable effects for fewer runs. The bundled
`scripts/doe_designs.py` uses **NumPy + Polars** without requiring pandas,
pydoe, SciPy, or an additional Python environment. It generates plans, not
training results. Do not assume every generated combination is feasible.

## Example research factors

| Factor | Levels | Intended question |
| --- | --- | --- |
| Model output | text, image, both | Which modalities add decision utility? |
| Active belief policy | off, on | Does active rollout selection help? |
| Rollout horizon | 1, 2, 4 | How does foresight trade off against cost? |
| Uncertainty threshold | 0.2, 0.5, 0.8 | How sensitive is selection? |
| Model seed | 2026, 2027, 2028 | How stable is training? |

Treat these as illustrative, not approved benchmark settings. Some factors may
be logically coupled, e.g. active belief may require world-model predictions;
an invalid combination must not silently be treated as a valid trial.

## Full factorial

A two-level design over `k` factors requires `2**k` combinations and permits
estimation of main effects and interactions. A saturated model with one
observation per combination has no residual degrees of freedom, so plan
independent replications or a predeclared reduced model before significance
claims.

```python
from doe_designs import full_factorial, two_level_factorial

modalities = full_factorial(
    {"modality": ["text", "image", "both"], "belief": ["off", "on"]},
    seed=2026,
)
continuous = two_level_factorial(
    {"horizon": (1, 4), "uncertainty": (0.2, 0.8)}, seed=2026
)
```

The `full_factorial` helper accepts explicit numeric or string levels.
`two_level_factorial` uses numeric `(low, high)` pairs. Both return Polars
DataFrames; randomize=True adds `run_order`.

## Fractional factorial and aliasing

A regular `2**(k-p)` fractional design saves compute but makes some effects
indistinguishable. The helper supports **positive Yates generator notation**
using independent lowercase single-letter factors, e.g. `a b c abc` for four
named columns in eight runs. It intentionally does not implement signed
generators. This is a narrow, auditable subset of the original `pydoe` API.

```python
from doe_designs import fractional_factorial

matrix = fractional_factorial(
    {"a": (0, 1), "b": (0, 1), "c": (0, 1), "d": (0, 1)},
    generator="a b c abc",
    seed=2026,
)
```

For `I = ABCD`, the example is resolution IV: main effects are not aliased
with two-factor interactions, but some two-factor interactions are aliased
with each other. Interpret the actual generator, not a generic label.

| Resolution | Main limitations |
| --- | --- |
| III | Main effects can alias with two-factor interactions |
| IV | Main effects separated from two-factor interactions; 2FI can alias with other 2FI |
| V | Main effects and 2FI separated from one another, assuming negligible higher interactions |

## Plackett-Burman-type screening

The `plackett_burman` helper creates orthogonal two-level screening arrays
using Sylvester Hadamard matrices (4, 8, 16, 32, ...) and Paley Hadamard
matrices (12, 20 runs). It selects the smallest supported run count with more
runs than factors. Other non-power-of-two run counts are not implemented.
For 9 to 11 factors, 12 runs are available; for 17 to 19, 20 runs are
available. More factors may require a higher power-of-two design.

These are **main-effect screening designs**; the arrays do not resolve all
interactions. If factors interact, estimated main effects may be misleading.
Prefer full/fractional factorial studies once key factors are identified.

## Response-surface design

Continuous factors may exhibit curvature that a two-level factorial cannot
estimate. The helpers offer:

- `central_composite`: factorial, axial, and replicated center points;
  default is `face="inscribed"`, `alpha="rotatable"`, `center=(0, 3)`.
- `box_behnken`: combinations of two off-center factors and center points;
  requires at least three factors and avoids all-extreme corners.

```python
from doe_designs import box_behnken, central_composite

ranges = {"horizon": (1, 5), "threshold": (0.2, 0.8)}
ccd = central_composite(ranges, face="inscribed", seed=2026)

other = box_behnken(
    {"horizon": (1, 5), "threshold": (0.2, 0.8), "temperature": (0.0, 0.6)},
    seed=2026,
)
```

**Important:** a CCD treats factors as continuous. An integer rollout horizon
must be mapped to permissible discrete values without accidental duplicates
or out-of-range settings; such rounding changes the design's geometry and can
invalidate an intended quadratic fit. Likewise, do not choose physically
infeasible combinations just because the matrix contains them.

For CCD, `circumscribed` can put axial points outside the `(low, high)` range;
`inscribed` and `faced` stay inside it. `orthogonal` and `rotatable` alpha
settings are supported, as is an explicit positive numeric alpha. Replication
at center points is not the same as independent training seeds unless that is
how the experiment is actually executed.

## Latin hypercube

Latin hypercube creates one stratified value per marginal interval, useful for
covering a continuous configuration space with fewer expensive evaluations.

```python
from doe_designs import latin_hypercube

sweep = latin_hypercube(
    {"threshold": (0.1, 0.9), "temperature": (0.0, 1.0)},
    n_samples=12,
    criterion="maximin",
    seed=2026,
)
```

Supported criteria are `random`, `center`, `maximin`, `centermaximin`, and
`correlation`. The last three compare **12 deterministic candidate designs**;
this is a heuristic, not an optimality guarantee. `maximin` uses a pairwise
minimum-distance score, which can be costly for very large `n_samples`.
A Latin hypercube alone is not a Sobol/Saltelli sensitivity analysis.

## Before running an expensive plan

Define hypotheses, units, valid factor combinations, primary metric, baseline,
resource cost per row, number of replicated seeds/tasks, nuisance blocks, and
stopping criteria. Record a run configuration and output a Polars table/CSV
under `.artifacts/` when appropriate. Ask before fetching any dependencies,
models, datasets, or launching substantial benchmark compute.
