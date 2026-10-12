# Sequential and Adaptive ML Experiments

Sequential methods permit predefined interim evaluations. Adaptive methods
change the future evaluation schedule based on interim information. Both can
save resources, but naive repeated testing or winner selection can invalidate
statistical conclusions. Most small agent ablations should use a simpler fixed
protocol unless the compute budget justifies adaptivity.

## Fixed protocol first

For a primary comparison, predeclare:

- Benchmark version, fixed task IDs, evaluation protocol, and test isolation.
- Baseline and candidate checkpoint identifiers.
- Number of independent tasks, seeds, and repeated episodes per task.
- Primary effect measure, uncertainty method, and allowed exclusions.
- Wall-clock/GPU/API limit and how failed or timed-out runs count.

Running experiments in batches for operational reasons does not make the
inference sequential **unless interim results are used to decide whether or
how to continue**. Avoid peeking at test outcomes and stopping when success
looks favorable.

## Why naive interim peeking is problematic

Repeatedly testing with a nominal significance level of 0.05 and stopping on
any favorable look increases the overall false-positive probability. The
amount of inflation depends on number and dependence of looks, effect-size
distributions, and the stopping rule. Do not attach unadjusted significance
claims to a selectively stopped result.

## Group-sequential approaches

If confirmatory inference genuinely requires early stopping, decide in advance
how many looks will occur and at which fractions of statistical information.
Common approaches include:

- **Pocock:** similar nominal boundaries at each planned look.
- **O'Brien-Fleming:** very stringent early boundaries, closer to ordinary
  significance thresholds near the final look.
- **Alpha-spending:** allocates total error budget across interim analyses.

The appropriate boundary depends on the outcome estimator, variance structure,
paired task design, correlated repeats, and planned information fractions.
These calculations are **not** implemented in the bundled scripts. Do not
invent boundaries or infer that a fixed number of evaluated tasks always equals
a fixed information fraction.

## Adaptive development versus held-out evaluation

A useful ML research distinction:

1. **Development experiments** may use adaptive model selection, hyperparameter
   sweeps, or early stopping on a designated validation set. Document search
   space, seeds, runs evaluated, selection criterion, and compute budget.
2. **Final benchmark claims** require an evaluation set and protocol insulated
   from that iterative tuning. Repeatedly viewing held-out OSWorld results and
   adapting the method makes that benchmark part of development.

Use a new untouched holdout or transparent exploratory qualification if prior
benchmark feedback has already influenced decisions. Never call a repeatedly
tuned test set a clean unbiased final test.

## Compute-aware adaptation

For very expensive agent evaluations, a staged execution plan can be useful:

- Validate code and environment on local fixtures first.
- Run an explicitly approved small smoke sample for execution integrity, not
  for cherry-picking statistical significance.
- Execute the precommitted main evaluation plan within approved budget.
- If early termination is necessary, classify the outcome as incomplete or
  apply an independently justified preplanned sequential rule.

A resource-based stop caused by a machine outage or exhausted budget is not
statistical evidence that the candidate is worse. Report completed tasks,
missing outcomes, failures, and the exact interruption point.

## Reporting requirements

- List all interim looks, selection decisions, variants examined, and changes
  to the design, not only the final chosen run.
- Report point estimates, task-level uncertainty, number of independent units,
  compute cost, and effects of early stopping.
- Distinguish exploratory model development from confirmatory evaluation.
- Document data leakage risk and correlated repeated observations.

Keep actual run logs, command lines, metric CSV files, and decisions in
`.artifacts/` when they aid debugging; preserve validated conclusions in the
tracked research report when requested.

## Resource and permission gate

Before fetching assets, external code, packages, or model checkpoints, ask for
approval with source and expected footprint. Before running an expensive or
paid sequential experiment, request approval of the compute budget and stop
conditions. No automated execution is implied by a sequential plan.
