# Randomization, Blocking, Stratification, and Controls

## First decide what should be randomized

In matched agent evaluations, **do not split the baseline and candidate across
different task sets** if both can run on the same fixed tasks. Instead keep
the tasks paired, restore the environment, and randomize or counterbalance the
order in which conditions are evaluated. Randomly assigning different task
sets would usually lose the power of pairing and risk unequal difficulty.

When conditions must be assigned to genuinely independent units, use simple,
blocked, stratified, or cluster randomization. The unit of assignment determines
which observations count as independent.

## Allocation methods

| Method                                | Use                                                 | Caveat                                        |
| ------------------------------------- | --------------------------------------------------- | --------------------------------------------- |
| Simple allocation                     | Independent units; balance not essential            | Small samples can be imbalanced               |
| Permuted blocks                       | Maintain arm ratios in complete blocks              | Incomplete blocks need not balance            |
| Stratified blocks                     | Balance arms within task family or application      | Sparse strata may make blocks impractical     |
| Cluster assignment                    | Conditions assigned at whole machine/app/site level | Cluster, not episode, is the independent unit |
| Paired tasks with order randomization | Every condition runs on every task                  | Reset state and handle order effects          |

Use `scripts/randomization.py` only when an explicit allocation/order schedule
helps. It returns `pl.DataFrame` and uses four-digit seeds by default.

```python
from pathlib import Path

from randomization import arm_balance, block_randomization

Path(".artifacts/experiments/ablation").mkdir(parents=True, exist_ok=True)
schedule = block_randomization(
    n=24, arms=["baseline", "active_belief"], block_size=4, seed=2026
)
summary = arm_balance(schedule, by="block")
schedule.write_csv(".artifacts/experiments/ablation/allocation.csv")
```

`arm_balance` returns a **long-form Polars table** with columns such as `block`,
`arm`, and `count`. Existing schedules can be inspected without this helper by
using Polars `group_by(...).agg(pl.len())`.

### Stratified allocation

```python
from randomization import stratified_block_randomization

schedule = stratified_block_randomization(
    ["browser", "editor", "browser", "settings", "editor"], seed=2026
)
```

For a sequence input, `unit_id` preserves the original one-based position even
when strata are interleaved. A dictionary input maps stratum labels to future
slot counts and allocates consecutively. Preserve the mapping from unit IDs to
actual benchmark task IDs.

### Cluster assignment

Use `cluster_randomization` when conditions must apply to whole task families,
machines, or other clusters. If a condition is applied to every task and the
unit of comparison is the task itself, prefer paired evaluation rather than
cluster randomization.

### Run order

`scripts/randomization.py::assign_factorial_runs` assigns a global shuffled
`run_order` to a Polars design table. **Do not use global shuffling for
restricted/split-plot designs**. In that case randomize within the explicitly
permitted blocks and retain `block` or `batch` columns.

## Controls that matter for computer-use agents

- Benchmark release, fixed task IDs, task exclusions, and success checker.
- VM image, operating system, software versions, UI viewport and resolution.
- Starting state, reset procedure, account/session/file artifacts.
- Agent backbone, prompt, tool/action API, temperature, model endpoint.
- Number of actions, retries, model calls, inference-time compute and timeouts.
- Evaluation batch, GPU contention, API throttling, cache warm-up, network drift.
- Training/validation/test separation; no tuning on the held-out test set.

Where possible, run controls contemporaneously instead of comparing a fresh
candidate against an old baseline produced under a different environment.

## Documentation

Record assignment method, seed, task IDs, block variables, run order, control
protocol, any deviations, and analysis plan. Save intermediate schedules in
`.artifacts/` and keep the directory gitignored. Promote important confirmed
findings to versioned documentation only when explicitly requested.

## Safety and approvals

Before downloading benchmark images, datasets, model weights, or installing a
package, ask the user with source, size if known, destination, and command.
Reading existing local files and using installed NumPy/Polars requires no
additional download approval. Large evaluations require separate budget approval.
