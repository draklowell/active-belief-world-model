"""Loading, splitting, and persisting the wine dataset."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"


def materialize_raw_csv(output_path: str | Path) -> pl.DataFrame:
    """Load ``sklearn``'s built-in wine dataset and write it to ``output_path`` as CSV.

    This is the one place the dataset is fetched from ``sklearn`` — everything else in
    the project reads the CSV, so the pipeline works the same way it would for a
    dataset that came from a file instead of a Python package.

    Parameters
    ----------
    output_path : str or Path
        Where to write the CSV, e.g. ``data/raw/wine.csv``.

    Returns
    -------
    polars.DataFrame
        The 13 feature columns plus a ``target`` column (0/1/2, one per cultivar).
    """
    bunch = load_wine()
    df = pl.DataFrame(bunch.data, schema=bunch.feature_names, orient="row").with_columns(
        pl.Series(TARGET_COLUMN, bunch.target)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    return df


def load_raw_csv(path: str | Path) -> pl.DataFrame:
    """Read the materialized dataset back from disk.

    Raises
    ------
    FileNotFoundError
        With a hint to run `python scripts/download_data.py` — this is the error
        message a student sees immediately after cloning the repo, so it should say
        what to do next.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. Run `python scripts/download_data.py` first."
        )
    return pl.read_csv(path)


def split_features_target(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Split a dataframe into feature columns ``X`` and the ``target`` series ``y``."""
    X = df.drop(TARGET_COLUMN)
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_stratified(
    df: pl.DataFrame, test_size: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stratified train/test split so all three classes stay proportionally represented.

    Why this matters: with an unstratified split, a small or imbalanced dataset (this
    one has 178 samples across 3 classes) can easily leave the test set with too few
    examples of the smallest class to measure anything meaningful about it — or, in
    the worst case, none at all.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    return train_df, test_df
