# Experimental Units and Design Types for ML Agents

Choose the design based on what is assigned independently, what is repeated,
and what population the comparison should generalize to. Do not confuse a UI
transition, action step, episode, benchmark task, task family, or training seed.

## Paired task design

For an ablation on OSWorld, evaluate baseline and candidate on the **same task
IDs** and use task-level paired differences. Within each task, reset the VM
and controls to the same initial state. Randomize or alternate condition order
where execution order, cache, server load, or time could matter. This is not
literal simultaneous random assignment: both variants are tested on the same
task under controlled resets.

An example result table:

| task_id  | application | seed | baseline_success | active_success |
| -------- | ----------- | ---- | ---------------- | -------------- |
| task_001 | editor      | 2026 | 0                | 1              |
| task_002 | browser     | 2026 | 1                | 1              |

These rows are **illustrative**, not observed OSWorld results. Report the number
of distinct tasks and aggregation over repeated evaluations explicitly.

## Completely randomized design

Use when assigning independent units to different conditions without known
nuisance strata, such as a genuinely independent collection of evaluation jobs
that cannot all run under both conditions. Randomization only balances
confounders in expectation and can still be imbalanced at small sample sizes.
Use `scripts/randomization.py::simple_randomization` if necessary.

## Randomized block and stratified designs

Group tasks by application, difficulty, UI type, environment image, or execution
batch, then randomize treatment order or assignment **within** each block.
Blocking aims to reduce noise from known nuisance variation. A randomized
complete block has each treatment represented within every block. Preserve
block membership in the results and analysis.

For an experiment evaluating all conditions on all tasks, task identity itself
is a block, enabling paired analysis.

## Full and fractional factorial designs

For factors such as predicted modality, active policy, rollout horizon, and
belief uncertainty threshold, a full design examines combinations and
interactions. Fractional designs reduce runs but alias some effects. The
numerical helpers are documented in `factorial_and_doe.md`.

## Repeated measurements and crossovers

Multiple episodes on one benchmark task or repeated measurements under model
checkpoints are correlated. A repeated-task paired design can reduce variation,
but initial states, interaction history, and temporal drift must be controlled.
Unlike clinical crossovers, a resettable virtual machine can often be restored
to the same snapshot before each condition. Verify that the reset genuinely
restores the relevant state, including files, session, caches, and accounts.

Analyze within-task correlation with paired aggregation or a hierarchical
model; do not treat every action as an independent sample.

## Split-plot designs

Some experimental factors are expensive to change, such as a model checkpoint,
GPU machine configuration, or VM image. Other factors change cheaply per task,
such as an inference threshold. If entire sessions share the costly setting,
randomization occurs at different levels. Account for separate variability
from checkpoint/VM batch and task-level factors. A naive fully randomized
analysis would overstate effective replication.

## Cluster-randomized design

When a treatment is genuinely assigned to whole applications, machines, or task
families, the **cluster** is the randomization unit. More episodes inside one
cluster do not create more independent assignments. Cluster count and
intra-cluster correlation limit precision. Use cluster-level or suitable
hierarchical analyses and plan replication at the cluster level.

## Latin squares

Latin squares can balance two nuisance dimensions, for example model order and
evaluation day, when conditions can be arranged in a square with each condition
once per row/column. They assume absent or controlled factor interactions and
are often unnecessary for small agent ablations. Do not use without a clear
reason and a documented valid layout.

## Nested design and pseudoreplication

Possible dependency hierarchy:

```text
benchmark version
  application / task family
    task ID
      model seed / evaluation episode
        UI step / observation
```

The hierarchy is conceptual; model seed may cross task IDs rather than nest
within them. Define the actual crossing or nesting in each experiment.

Pseudoreplication occurs when correlated lower-level observations are counted
as independent treatment replications. A task run with 60 UI actions is not
60 independent successes; ten runs on the same task are not ten independent
task IDs. Use per-task aggregation, paired tests, cluster resampling, or an
appropriate hierarchical model.

**Implementation rule:** report task IDs, unique counts, runs per task, model
seeds, run order, excluded episodes, and reset policy. Do not infer statistical
independence just because results are stored in separate CSV rows.
