# Active Belief World Models for Computer-Use Agents (ABWM)

![lint](https://github.com/draklowell/active-belief-world-model/actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

Research project on active belief world models for computer-use agents, evaluated on
[OSWorld](https://github.com/xlang-ai/OSWorld) and trained on
[AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) trajectories.

## Repository structure

```
.
├── src/                  # agentnet.py and agents/ (agent implementations)
├── scripts/              # benchmark, inference server, and AgentNet download scripts
├── app/                  # streamlit_app.py, placeholder for the results viewer
├── notebooks/            # research notebooks (see notebooks/README.md)
├── tests/                # pytest suite
├── config/               # agent and inference server configurations
├── data/                 # raw data NOT committed; data/README.md (data card) IS
├── results/              # experiment results
├── evaluation_examples/  # OSWorld task definitions
├── docs/                 # proposal, poster, slides, and supporting docs
└── .github/workflows/    # lint.yml
```

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.0 or newer,
then install the system dependencies for your OS before setting up the project.
These libraries support source builds of Pillow and pygame, which are pulled in by
`desktop-env`.

### Ubuntu

The following commands are for Ubuntu, which is also used by the lint workflow:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y \
  build-essential \
  pkg-config \
  libjpeg-dev \
  zlib1g-dev \
  libsdl2-dev \
  libsdl2-image-dev \
  libsdl2-mixer-dev \
  libsdl2-ttf-dev \
  libfreetype6-dev \
  libportmidi-dev
```

### macOS

Use an Apple silicon Mac running macOS 14 or newer.

Install the Xcode Command Line Tools if needed, and wait for installation to finish:

```bash
xcode-select --install
```

Install [Homebrew](https://brew.sh/), then install the native build dependencies:

```bash
brew install pkg-config jpeg zlib sdl2 sdl2_image sdl2_mixer sdl2_ttf freetype portmidi
```

### Project setup

After completing the Ubuntu or macOS prerequisites, run:

```bash
git clone git@github.com:draklowell/active-belief-world-model.git
cd active-belief-world-model
uv python install 3.14
uv sync --locked                    # creates .venv and installs runtime and dev dependencies
uv run pre-commit install            # installs the existing Git hooks
uv run streamlit run app/streamlit_app.py  # opens the demo at localhost:8501
```

> [!important]
> Python 3.14 is required by `.python-version` and `pyproject.toml`. After installing the system prerequisites, `make setup` installs Python, syncs dependencies, and installs the Git hooks. Do not substitute an older Python version if a dependency fails to install.
>
> The `Makefile` wraps project setup and the local MLflow server. Every other command in this README (the app, linting, tests) is run directly; see "Development" below.

### Dependency management

`pyproject.toml` is the source of truth for runtime dependencies and the `dev` dependency group. `uv.lock` records the resolved versions for reproducible installs; commit both files when dependencies change. `uv sync` includes development dependencies and installs the project in editable mode. Use `uv run` for commands in the project environment. PyTorch uses the CPU index on Linux and Windows and PyPI on macOS.

```bash
uv add <package>                     # add a runtime dependency
uv add --dev <package>               # add a development dependency
uv lock --check                      # check that the lock matches pyproject.toml
uv sync --locked                     # install without changing the lock (also used in CI)
```

The existing flake8, Black, isort, and pre-commit checks remain in use. After changing dependencies, rerun the tests and hooks.

Project tables use Polars. Streamlit still requires pandas as a transitive dependency; project code and notebooks do not import it.

## Development

```bash
uv run flake8 src/ scripts/ tests/ app/ && uv run black --check src/ scripts/ tests/ app/ && uv run isort --check src/ scripts/ tests/ app/  # what CI runs
uv run black src/ scripts/ tests/ app/ && uv run isort src/ scripts/ tests/ app/  # auto-fix formatting
uv run pytest                         # full test suite
uv run ty check                       # type-check src/, scripts/, tests/, and app/
uv run pre-commit run --all-files      # existing hooks
uv run jupyter lab notebooks/          # open notebooks with the project kernel
```

Pre-commit hooks (installed by `make setup`) catch most of this before you even open a PR — see `.pre-commit-config.yaml`.

The ty hook checks all first-party Python code on every commit using the project's uv environment. Run `uv sync --locked` after dependency changes; the hook uses `--locked --no-sync` so it does not install dependencies or update `uv.lock`. CI also runs `uv run ty check`.

### Research notebooks

Use JupyterLab for research notebooks. Save them as `.ipynb` files in `notebooks/`
and use a Python 3.14 kernel with the dependencies from `pyproject.toml`.
See `notebooks/00_jupyter_example.ipynb` for an interactive example and
`notebooks/README.md` for notebook conventions.

Before rerunning a notebook, inspect its code and metadata for compute and environment
requirements. Prefer an existing configured remote kernel with the required environment
when available. Use a local kernel only for verified lightweight CPU notebooks, such as
the interactive example. Do not automatically rerun GPU, model training or fine-tuning,
or other expensive notebooks, locally or remotely; these runs require explicit user
authorization and suitable compute.

When execution is appropriate, restart the selected kernel and run all cells in order.
If suitable compute is unavailable, requirements are unclear, or execution is not
authorized, validate notebook schema and code syntax without executing cells. Preserve
saved outputs and report that execution was not run, the reason, and any potentially
stale outputs.

```bash
uv run jupyter lab notebooks/
```

> [!note] Why this matters: CI enforces it instead of "we agreed to"
> "The team agreed to run black before committing" survives about one deadline-week skip before someone's code silently stops matching. A red `lint` check in CI is unambiguous, and — importantly — **a red CI check blocks merging**: reviewing a PR you can't tell has passed its own tests is a bad use of a reviewer's time, so don't merge over red CI, fix it first.

## License and citation

Code is released under the MIT license, see [`LICENSE`](LICENSE). Datasets keep their own terms,
see [`data/README.md`](data/README.md). Citation metadata is in [`CITATION.cff`](CITATION.cff).
