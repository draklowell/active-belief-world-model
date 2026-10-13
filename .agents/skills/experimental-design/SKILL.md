---
name: experimental-design
description: >-
  Design reproducible ML and computer-use-agent experiments before execution.
  Use for research hypotheses, ablations, baselines, OSWorld evaluation, task
  pairing, randomization, blocking, factorial designs, uncertainty estimates,
  sequential plans, and compute budgets. Produce a reviewable experiment plan
  and do not start expensive runs without user authorization.
allowed-tools: Read Write Edit Bash Glob Grep WebSearch WebFetch Python Task
compatibility: >-
  Codex and compatible agents. Project target is Python 3.14, uv, NumPy,
  Polars, marimo, pytest, Black, isort, flake8, ty, and pre-commit.
  Tool names depend on the agent host. No automatic downloads or installations.
license: MIT
metadata:
  version: "3.0-project-adaptation"
  adapted-from: "K-Dense-AI/scientific-agent-skills/skills/experimental-design"
  upstream-author: "K-Dense Inc."
---

# Experimental Design for Computer-Use and ML Research

## Purpose

Design comparisons that distinguish real improvement from confounding, noise,
leakage, and selection bias. Use this skill for experimental planning, small
local allocation or DOE calculations, and evidence review. It does **not**
authorize autonomous benchmark execution or model training. Follow the root
`AGENTS.md` and the user's instructions.

Core principles: define a falsifiable hypothesis, choose the experimental unit,
select a comparable baseline, control nuisance variation, plan replication,
pre-specify metrics and evaluation budgets, then interpret observed results with
uncertainty. Keep hypotheses separate from measured evidence.

## Tools and permissions

Use available tools for reading local research papers and code, repository
search, editing files, terminal commands, scientific calculations, browsing
public technical documentation, and delegating independent analysis to
supported agents. `allowed-tools` is advisory; actual host permissions and
sandbox restrictions always take precedence.

**Mandatory approval before any new download or installation.** Before running
commands that fetch packages, models, datasets, pretrained weights, tokenizers,
PDFs, repositories, archives, benchmark assets, or external scripts, stop and
ask the user. This includes implicit downloads by `uv sync`, `uv add`, model hub
APIs, benchmark setup scripts, and first-run caches. Specify the resource,
source, reason, approximate size if known, destination, and proposed command.
Wait for explicit approval. Never work around a denied permission.

Public web search and viewing online documentation are allowed when browsing is
available. Saving remote content locally counts as a download and requires
approval. Ask separately before expensive GPU/CPU evaluation, paid APIs, or
runs exceeding the agreed resource budget. Do not stage, commit, push, or create
branches unless explicitly requested.

Use existing local tools without syncing when possible:

```bash
uv run --no-sync python --version
uv run --no-sync pytest
uv run --no-sync ty check
```

Check that the interpreter and dependencies already exist first. `--no-sync`
does not prevent a program from fetching data internally. Never assume an
uncached managed Python interpreter is available without downloading it.

## Project conventions

- Target **Python 3.14**. If root `.python-version`, `pyproject.toml`, lockfile,
  or `AGENTS.md` specify a different target, stop and ask which version is
  authoritative; do not silently edit the environment.
- Manage dependencies exclusively with `uv` and project `pyproject.toml`/
  `uv.lock`. No separate `.venv-design`, global pip, or implicit installs.
- Use NumPy for numerical design generation and Polars DataFrames for tables.
  Prefer Polars expressions to row-by-row Python data processing.
- Reuse available libraries. The bundled scripts require only NumPy and Polars;
  no pandas, pydoe, or SciPy dependency is required.
- Write modern type hints with built-in types. Do not add
  `from __future__ import annotations` or unnecessary `typing` imports.
- Keep Python simple and idiomatic; avoid redundant defensive checks, code
  comments, unused abstractions, and private-name prefixes for new helpers.
- Use four-digit seeds, e.g. `2026`, for new experiments. Preserve original
  seeds when reproducing previous findings.
- Use marimo `.py` files for new exploratory notebooks. Preserve existing
  `.ipynb` files unless explicitly asked to migrate them.
- Use `.artifacts/` for temporary plans, per-run results, CSVs, analysis,
  troubleshooting notes, and logs; ensure `.artifacts/` is gitignored. Do not
  store secrets or unnecessary copies of large datasets. Promote validated
  findings to tracked reports only when requested.

## Workflow

### 1. Specify the question

Record hypothesis, predicted mechanism, falsifying observation, primary
estimand, target task population, benchmark/version, and unit of independence.
In OSWorld, tasks, repeated episodes, action steps, and model seeds are distinct
levels and must not be conflated. Distinguish `observed`, `sourced`,
`hypothesized`, and `unknown` claims.

### 2. Build the comparison

Use a matched backbone, prompt, action space, observation protocol, world-model
inputs, task IDs, versions, and budget unless explicitly studying one of them.
An illustrative ablation for Active Belief World Models is:

| Condition     | World-model information      | Active belief policy |
| ------------- | ---------------------------- | -------------------- |
| Base agent    | None                         | Disabled             |
| Text          | Predicted textual transition | Disabled             |
| Image         | Predicted next observation   | Disabled             |
| Text + image  | Both outputs                 | Disabled             |
| Active belief | Matched text + image         | Enabled              |

Do not attribute a gain to belief-state reasoning if several conditions change
simultaneously. Report additional inference cost, calls, and latency alongside
quality. OSWorld 1.0 and 2.0 require separate protocols and analyses.

### 3. Choose the design

- **Paired benchmark evaluation:** Evaluate conditions on the same fixed tasks;
  randomize or counterbalance execution order where feasible. Preferred first
  choice for an agent ablation.
- **Randomized block/stratified design:** Balance task family, application,
  difficulty, environment, machine, or evaluation batch.
- **Full factorial:** Test a few toggles and their interactions.
- **Fractional factorial / Plackett-Burman:** Screen many factors cheaply, but
  document aliased interactions and supported matrix size.
- **Response surface / Latin hypercube:** Explore continuous rollout horizon,
  thresholds, weights, or compute budgets when appropriate.
- **Sequential/adaptive design:** Predefine interim decisions, stopping rules,
  and any multiple-testing correction before inspecting results.

See [design types](references/design_types.md), [randomization and blocking](references/randomization_and_blocking.md), [factorial DOE](references/factorial_and_doe.md), and [sequential designs](references/sequential_and_adaptive.md).

### 4. Control confounders and leakage

Keep the train/dev/test split, task set, task reset rules, model checkpoints,
software/VM images, prompts, seed policy, temperature, inference budget, and
outcome checker fixed where required. Guard against test-set tuning, selective
failure omission, correlated task episodes, caching, nondeterminism, and API
version drift. Predefine exclusions and the treatment of timeouts and crashes.

### 5. Plan uncertainty and replication

Specify independent tasks, seeds, replicates, run order, aggregation, and sample
size. Prefer task-paired differences and confidence intervals for agent success;
use hierarchical analysis if multiple episodes or model seeds are nested within
tasks. Do not treat UI action steps as independent replications. A single lucky
run or non-significant result is not evidence of superiority or equivalence.

### 6. Define metrics and budget

Primary: preselected success-rate difference (or another explicit objective).
Secondary: per-task outcomes, number of actions, latency, model calls, cost,
retries, timeouts, and error taxonomy. For world models, include transition
correctness, decision utility, and uncertainty calibration when meaningful.
Specify task IDs, time/step limits, model versions, number of repetitions,
compute resources, and acceptable cost **before** execution.

### 7. Produce the experiment specification

Save significant plans to `.artifacts/experiments/<experiment-id>/plan.md` or
`config.yaml` when the user asks for implementation or persistent local notes.
Use this minimal template:

```yaml
experiment_id: active_belief_ablation
hypothesis: "Belief-guided selection improves paired task success"
benchmark: "OSWorld, exact version TBD"
experimental_unit: task_id
conditions: [base_agent, text_image, active_belief]
primary_metric: paired_success_rate_difference
secondary_metrics: [actions, latency, model_calls]
seed: 2026
task_ids: TBD
controls: [backbone, prompt, action_budget, vm_image]
compute_budget: TBD
status: proposed
```

Do not invent task identifiers, model results, prices, or benchmark settings.
Ask when a missing input prevents meaningful planning.

### 8. Generate local design tables when needed

The bundled scripts return Polars DataFrames:

```python
from pathlib import Path

from doe_designs import two_level_factorial
from randomization import block_randomization

Path(".artifacts").mkdir(exist_ok=True)
allocations = block_randomization(12, arms=["baseline", "active"], seed=2026)
design = two_level_factorial(
    {"rollout_horizon": (1, 4), "belief_threshold": (0.2, 0.8)}, seed=2026
)

allocations.write_csv(".artifacts/allocations.csv")
design.write_csv(".artifacts/factorial.csv")
```

Run examples only when an existing local environment contains dependencies.
Never automatically install them. Review actual feasibility and alias structure
before turning a generated table into a costly execution schedule.

### 9. Verify and review

For implementation tasks, follow root project checks as applicable:

```bash
uv run --no-sync isort --check-only .
uv run --no-sync black --check .
uv run --no-sync flake8 src/ scripts/ tests/ app/
uv run --no-sync ty check
uv run --no-sync pytest
uv run --no-sync pytest .agents/skills/experimental-design/tests
```

Only run commands relevant to paths that exist; don't weaken quality rules to
make checks pass. After substantial changes, remove code made obsolete by the
implementation and run the full applicable test suite. Clearly report PASS,
FAIL, or NOT RUN. Keep experimental outcomes and diagnostics in `.artifacts/`
when useful; summarize crucial findings in the response.

For design-only requests, deliver the design and stop before training, benchmark
execution, or downloading additional resources.

## Failure patterns

- Pseudoreplication: counting steps/episodes as independent benchmark tasks.
- Confounded ablation: changing modality and compute budget with the belief policy.
- Task and data leakage: tuning on a benchmark treated as held-out evaluation.
- Best-seed selection, cherry-picked successes, undocumented exclusions.
- Repeated significance testing without valid sequential error control.
- Missing overhead measurements or comparisons between mismatched OSWorld versions.
- Treating a seed alone as complete reproducibility.

## Bundled resources

- `scripts/randomization.py`: simple, block, stratified, cluster allocation;
  factorial run ordering; arm counts. Polars output.
- `scripts/doe_designs.py`: full/fractional factorial, Plackett-Burman-type
  screening, CCD, Box-Behnken, Latin hypercube. NumPy + Polars only.
- `references/randomization_and_blocking.md`: paired tasks, strata, controls.
- `references/design_types.md`: experimental units and nested outcomes.
- `references/factorial_and_doe.md`: factorial aliasing and numerical designs.
- `references/sequential_and_adaptive.md`: interim looks and stopping rules.
- `tests/test_scripts.py`: executable regression tests for the bundled helpers.

## Attribution

Adapted and specialized from the MIT-licensed experimental-design skill in
[K-Dense Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills).
Retains principles of Fisher-style randomization, replication, blocking,
statistical design validity, and pre-specified analysis. The source author's
attribution and MIT notice should remain with redistributed adaptations.

For scientific manuscripts citing the skill library, verify publication
metadata of *Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents* (arXiv:2609.00065) before citation.
