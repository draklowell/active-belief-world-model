# AGENTS.md

## Development Workflow

1. Inspect relevant code and configuration before making changes. Follow existing project conventions.
2. Make the smallest coherent change that solves the task. Avoid unrelated refactoring and unnecessary abstractions.
3. Use `uv` exclusively for Python environments and dependency management. Run project commands through `uv run`.
4. Update `pyproject.toml` and `uv.lock` together when changing dependencies.
5. Run relevant tests and quality checks after modifications. Never claim a check passed unless it was executed.
6. Report changes, verification results, and remaining issues concisely.
7. Never stage, commit, push, or create branches unless explicitly requested.

## Python Style

1. Target Python 3.14. Write clean, idiomatic, readable Python.
2. Add parameter and return type annotations to every function and method. Prefer built-in types such as `list`, `dict`, `tuple`, and `str`, and modern unions such as `str | None`.
3. Do not use `from __future__ import annotations`. Avoid importing `typing` when built-in syntax is sufficient.
4. Prefer comprehensions and standard Python constructs when they improve readability.
5. Do not add code comments. Prefer expressive naming and straightforward implementations.
6. Avoid leading underscores in variable and function names, except when required by Python protocols or external APIs.
7. Use `pathlib` for filesystem operations.
8. Keep functions focused and avoid unnecessary classes, wrappers, and helper functions.

## Simplicity and Error Handling

1. Prefer simple code over defensive boilerplate.
2. Assume internal inputs satisfy documented contracts. Do not repeatedly validate values already guaranteed by the calling code.
3. Avoid speculative `if` checks, redundant `raise` statements, and excessive `try/except` blocks.
4. Validate inputs at meaningful system boundaries, especially when reading external data, configuration, or API responses.
5. Catch exceptions only when recovery or useful additional context is possible. Never silently swallow errors.
6. Do not use `assert` for production input validation. Use normal pytest assertions in tests.
7. Do not introduce fallback behavior unless it is explicitly required.

## Tooling and Verification

| Purpose | Tool |
|---|---|
| Dependencies and environment | uv |
| Formatting | Black |
| Import sorting | isort |
| Linting | flake8 |
| Static type checking | ty |
| Tests | pytest |
| Git hooks | pre-commit |
| Interactive notebooks | marimo |
| Tabular data | Polars |

Use the existing configuration instead of introducing alternative tooling.

After relevant Python changes, run:

```bash
uv run isort --check-only .
uv run black --check .
uv run flake8 src/ scripts/ tests/ app/
uv run ty check
uv run pytest
```

Use `uv run pre-commit run --all-files` for complete repository validation.

Report failures rather than weakening checks to make them pass.

## Data and Experiments

1. Prefer Polars for tabular data processing. Use native expressions, `with_columns`, and lazy operations where appropriate. Avoid Python loops for operations that can be expressed directly in Polars.
2. Preserve existing experiment seeds and splits unless explicitly changing the experiment.
3. Keep training, validation, and testing boundaries clear. Prevent data leakage.
4. Record experiment configurations, seeds, metrics, and relevant artifacts.
5. Compare experiments against a consistent baseline. Do not report unmeasured improvements or invent results.

## Notebooks and Visualizations

1. Prefer marimo for new notebooks. Store them as `.py` files in `notebooks/`.
2. Use dedicated Markdown cells for explanatory text. Avoid generating narrative paragraphs through `print()`, `display(Markdown(...))`, or similar wrappers.
3. Let notebook cells display their results directly when possible. Avoid verbose console output.
4. Prefer named plot colors such as `"red"` or `"steelblue"` instead of hexadecimal values.
5. Use consistent mathematical notation across code, plots, and explanations.
6. Keep notebooks focused on exploration and analysis. Move reusable logic into regular Python modules.
7. Preserve existing `.ipynb` notebooks unless migration is explicitly requested.

## Writing

1. Write concise, technical explanations without unnecessary repetition.
2. Avoid dash punctuation in prose.
3. Keep mathematical symbols and terminology consistent.
4. Prefer clear descriptions of assumptions, decisions, and results over generic commentary.

## Git Conventions

Use Conventional Commits with these prefixes:

`feat:`, `fix:`, `docs:`, `exp:`, `refactor:`, `test:`, `chore:`.

Commit subjects should describe the change. Commit bodies should explain **why the change was made**, including important reasoning and tradeoffs, rather than repeating the diff.

For `exp:` commits, include the measured result, evaluation setup, and location of detailed experiment records. Never invent metrics.

Do not combine unrelated changes into one commit. Do not create commits unless explicitly instructed.

## Artifacts and Research Notes

1. Use `.artifacts/` as the workspace for temporary research outputs, experiment results, debugging notes, and agent-generated reports.
2. Store useful intermediate findings when they help explain decisions, reproduce experiments, or continue work in a later session.
3. Allowed artifacts include Markdown notes, logs, CSV files, metrics, experiment configurations, profiling results, and debugging reports.
4. Organize artifacts by task or experiment. Use descriptive filenames and avoid unnecessary files.
5. Prefer updating an existing artifact over creating multiple overlapping reports.
6. Do not create documentation or artifacts for trivial changes.
7. Record relevant commands, configurations, seeds, metrics, and conclusions when reporting experimental results.
8. Clearly distinguish measured results, hypotheses, assumptions, and unverified observations.
9. Keep `.artifacts/` excluded from Git. Do not stage or commit its contents.
10. Do not store credentials, tokens, private information, or unnecessary copies of large datasets in artifacts.
11. Important validated findings intended for the project history should be promoted to tracked documentation when explicitly requested.
12. Summarize significant findings and reference relevant artifact paths in the final response.

## Code Changes and Cleanup

1. All substantial code changes must pass the relevant test suite, formatting, linting, and static type checks.
2. Preserve existing functionality unless the task explicitly requires changing it.
3. When replacing an implementation, remove obsolete functions, classes, imports, configuration, and references that are no longer used.
4. Do not leave commented-out implementations, redundant wrappers, or compatibility code that serves no purpose.
5. Update affected tests when behavior or interfaces change. Add focused tests for new functionality and important regressions.
6. Prefer modifying existing code over introducing parallel implementations of the same behavior.
7. Keep changes minimal, readable, idiomatic, and consistent with project conventions.
8. Do not perform unrelated cleanup outside the task scope.
9. Before removing code, verify that it is no longer referenced by relevant callers, tests, or supported interfaces.
10. Do not suppress type errors, disable tests, or weaken quality checks to make changes pass.
11. If verification fails, investigate and fix issues introduced by the change. Report pre-existing failures separately.
12. Do not describe a task as complete when required checks are failing or have not been executed. State the remaining blockers explicitly.
13. For substantial changes, optionally record implementation decisions, removed legacy code, verification results, and unresolved issues in `.artifacts/`.
