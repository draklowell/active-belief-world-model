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
6. Avoid leading underscores in variable and function names, except when required by Python protocols, Marimo conventions, or external APIs.
7. Use `pathlib` for filesystem operations.
8. Keep functions focused and avoid unnecessary classes, wrappers, and helper functions.
9. For imports do not use `import *`. You should import `polars` as `pl`, `polars.selectors` as `cs`, `plotly.express` as `px`, `plotly.graph_objects` as `go`, `plotly.subplots` as `sp`, `numpy` as `np`, `marimo` as `mo`, `streamlit` as `st`. For all other libraries, use the default import name or use `from <module> import <class/function>` for specific classes or functions. Avoid aliasing unless necessary to avoid name conflicts or improve clarity.

## Naming Conventions

1. Use snake_case for variables, functions, and methods. Use PascalCase for classes. Use UPPER_SNAKE_CASE for constants. Prefix internal/private attributes and methods with a single underscore (e.g., \_internal_state).
2. For boolean variables, use names that clearly indicate their meaning. Always use prefixes such as `is_`, `has_`, `can_`, or `should_`. For example, use `is_valid`, `has_permission`, `can_execute`, or `should_retry`.
3. Use strong verbs that describe the exact action being performed. Avoid ambiguous names like `process_data()`, `handle_event()`, or `do_work()`. Instead, use `parse_market_data()`, `route_incoming_event()`, or `calculate_volatility()`.
4. Specific Class Nouns: Use descriptive nouns that define the entity's exact role. Avoid vague suffix crutches like `Manager`, `Handler`, or `Processor`. Instead of `DataProcessor`, use `DataNormalizer` or `MetricsAggregator`.
5. Avoid arbitrary abbreviations, but accept domain-standard acronyms (e.g., `http`, `uuid`). While descriptive loop variables (e.g., for record in records) are strongly preferred, standard mathematical or index variables (`i`, `j`, `x`, `y`) are acceptable for simple iterations or coordinate math. Always use _ for throwaway variables in unpacks or loops. Do not use `df` acronyms for DataFrames. Use descriptive names like `user_data`, `transaction_records`, or `sensor_readings` instead.

## Docstrings

1. Add docstrings to all public functions, methods, and classes.
2. Do not add docstrings to private functions, private methods, private classes, special methods, any code in Marimo notebooks, or in the header of a file.
3. Use imperative mood for function and method docstrings. Use descriptive mood for class docstrings.
4. For parameters, attributes, and return values, provide 1-sentence desciprtion, unless the type is nested or complex, in which case provide a brief description of the structure and contents (e.g. key and value descriptions for a dictionary).
5. For large functions that have more than 60 lines of code, include a brief description of the algorithm between the summary and the `Args` section. Use blank lines to separate the summary, algorithm description, and `Args` section.
6. Use modified Google-style docstrings with a 1-to-3 sentence summary. Do not include types in the docstring, as they are already specified in the function signature.

Example of docstrings:

```python
def function(param1: int, param2: str) -> bool:
    """{1-3 sentence summary of the function}

    {brief description of the algorithm if the function is large}

    Args:
        param1: {description of `param1`}.
        param2: {description of `param2`}.

    Returns:
        {description of the return value}.
    """

def generator(param1: int) -> Generator[int, None, None]:
    """{1-3 sentence summary of the generator function}

    {brief description of the algorithm if the function is large}

    Args:
        param1: {description of `param1`}.

    Yields:
        {description of the yielded values}.
    """

class Class(object):
    """{1-3 sentence summary of the class}

    Attributes:
        attr1: {description of `attr1`}.
        attr2: {description of `attr2`}.

    Args:
        param1: {description of `param1`}.
        param2: {description of `param2`}.
        param3: {description of `param3`}.
    """
    attr1: str
    attr2: int

    def __init__(self, param1: str, param2: int = 0, param3: list[str] = []) -> None:
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3

    def method(self, param1: int) -> bool:
        """{1-3 sentence summary of the method}

        {brief description of the algorithm if the method is large}

        Args:
            param1: {description of `param1`}.

        Returns:
            {description of the return value}.
        """

    @property
    def readwrite_property(self) -> list[str]:
        """{description of `readwrite_property`}"""

    @readwrite_property.setter
    def readwrite_property(self, value: list[str]) -> None:
        ...
```

## Simplicity and Error Handling

1. Prefer simple code over defensive boilerplate.
2. Assume internal inputs satisfy documented contracts. Do not repeatedly validate values already guaranteed by the calling code.
3. Avoid speculative `if` checks, redundant `raise` statements, and excessive `try/except` blocks.
4. Validate inputs at meaningful system boundaries, especially when reading external data, configuration, or API responses.
5. Catch exceptions only when recovery or useful additional context is possible. Never silently swallow errors.
6. Do not use catch-all and reraise patterns for cleaning. Instead, use context manager where possible and if not, then `try/finally` pattern.
7. Do not use `assert` for production input validation. Use normal pytest assertions in tests.
8. Do not introduce fallback behavior unless it is explicitly required.
9. Never throw errors for operations where the semantic intent can be gracefully satisfied, for example:
   - Enforce mathematical identities: `mean([])` or `sum([])` should return `0`, not raise `ValueError`.
   - Make deletions idempotent: `delete(missing_record)` or `rm(missing_file)` must return success. The intent ("ensure it's gone") is already met.
   - Forgive boundaries: array slicing outside existing bounds should yield an empty array, not an `IndexOutOfBounds` exception.

## Tooling and Verification

| Purpose                      | Tool                   |
| ---------------------------- | ---------------------- |
| Dependencies and environment | uv                     |
| Formatting                   | Black                  |
| Import sorting               | isort                  |
| Linting                      | flake8                 |
| Static type checking         | ty                     |
| Tests                        | pytest                 |
| Git hooks                    | pre-commit             |
| Interactive notebooks        | marimo                 |
| Tabular data                 | polars                 |
| Plotting                     | plotly, plotly-express |

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

1. Prefer Polars for tabular data processing. Use native expressions, `with_columns`. Avoid Python loops for operations that can be expressed directly in Polars.
2. Use lazy evaluation only for multi-step pipelines that benefit from query optimization or streaming processing. Avoid lazy evaluation for single-step operations.
3. Preserve existing experiment seeds and splits unless explicitly changing the experiment. Use seed 2026 for new experiments. Avoid using random seeds unless explicitly required. Keep all experiments reproducible and deterministics.
4. Keep training, validation, and testing boundaries clear. Prevent data leakage.
5. Record experiment configurations, seeds, metrics, and relevant artifacts. # How to record?
6. Compare experiments against a consistent baseline. Do not report unmeasured improvements or invent results. # How to compare?

## Notebooks

1. Prefer marimo for new notebooks. Store them as `.py` files in `notebooks/`.
2. Use dedicated Markdown cells for explanatory text. Avoid generating narrative paragraphs through `print()`, `display(Markdown(...))`, or similar wrappers.
3. If narrative text has to be generated programmatically (e.g., for reporting variables), use native Marimo's functions such as `mo.md()`, `mo.image()` and others.
4. Do not use `print()` for any output in notebooks, instead use Marimo's native functions or return values from cells.
5. For slow iterative computations, use `mo.status.progress_bar(iterator)` to display progress bars. Avoid verbose console output.
6. Let notebook cells display their results directly when possible. Avoid verbose console output.
7. Use consistent mathematical notation across code, plots, and explanations.
8. Keep notebooks focused on exploration and analysis. Move reusable logic into regular Python modules.
9. Preserve existing `.ipynb` notebooks unless migration is explicitly requested.

## Plotting

1. Use Plotly Express for quick visualizations where possible. Only if Plotly Express does not support the required functionality, use Plotly's `graph_objects` API.
2. Do not use any other plotting libraries, wrappers or custom styles. Stick to built-in style and color palettes.
3. Always label axes, legends, and titles. Use clear and descriptive labels. Avoid generic labels such as "x", "y", or "value". Axes, legends and titles should be capitalized and use sentence case. Avoid all caps, camel case, or snake case.

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

01. Use `.artifacts/` as the workspace for temporary research outputs, experiment results, debugging notes, and agent-generated reports.
02. Store useful intermediate findings when they help explain decisions, reproduce experiments, or continue work in a later session.
03. Allowed artifacts include Markdown notes, logs, CSV files, metrics, experiment configurations, profiling results, and debugging reports.
04. Organize artifacts by task or experiment. Use descriptive filenames and avoid unnecessary files.
05. Prefer updating an existing artifact over creating multiple overlapping reports.
06. Do not create documentation or artifacts for trivial changes.
07. Record relevant commands, configurations, seeds, metrics, and conclusions when reporting experimental results.
08. Clearly distinguish measured results, hypotheses, assumptions, and unverified observations.
09. Keep `.artifacts/` excluded from Git. Do not stage or commit its contents.
10. Do not store credentials, tokens, private information, or unnecessary copies of large datasets in artifacts.
11. Important validated findings intended for the project history should be promoted to tracked documentation when explicitly requested.
12. Summarize significant findings and reference relevant artifact paths in the final response.

## Code Changes and Cleanup

01. All substantial code changes must pass the relevant test suite, formatting, linting, and static type checks.
02. Preserve existing functionality unless the task explicitly requires changing it.
03. When replacing an implementation, remove obsolete functions, classes, imports, configuration, and references that are no longer used.
04. Do not leave commented-out implementations, redundant wrappers, or compatibility code that serves no purpose.
05. Update affected tests when behavior or interfaces change. Add focused tests for new functionality and important regressions.
06. Prefer modifying existing code over introducing parallel implementations of the same behavior.
07. Keep changes minimal, readable, idiomatic, and consistent with project conventions.
08. Do not perform unrelated cleanup outside the task scope.
09. Before removing code, verify that it is no longer referenced by relevant callers, tests, or supported interfaces.
10. Do not suppress type errors, disable tests, or weaken quality checks to make changes pass.
11. If verification fails, investigate and fix issues introduced by the change. Report pre-existing failures separately.
12. Do not describe a task as complete when required checks are failing or have not been executed. State the remaining blockers explicitly.
13. For substantial changes, optionally record implementation decisions, removed legacy code, verification results, and unresolved issues in `.artifacts/`.
