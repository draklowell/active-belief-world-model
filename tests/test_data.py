"""Catches the most common silent mistakes in an ML data pipeline: wrong shape,
missing values that sneak through, and train/test contamination.
"""

import polars as pl
from polars.testing import assert_frame_equal
from sklearn.datasets import load_wine

from wine_origin.data import (
    TARGET_COLUMN,
    load_raw_csv,
    materialize_raw_csv,
    split_features_target,
    train_test_split_stratified,
)


def test_materialized_csv_has_expected_shape(tmp_path):
    df = materialize_raw_csv(tmp_path / "wine.csv")
    assert df.shape == (178, 14)  # 13 features + target
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()) == {0, 1, 2}


def test_csv_round_trip_preserves_feature_order_and_values(tmp_path):
    path = tmp_path / "wine.csv"
    materialized = materialize_raw_csv(path)
    restored = load_raw_csv(path)
    assert_frame_equal(restored, materialized)

    X, y = split_features_target(restored)
    wine = load_wine()
    assert X.columns == wine.feature_names
    assert X.to_numpy().tolist() == wine.data.tolist()
    assert y.to_list() == wine.target.tolist()


def test_no_missing_values(tmp_path):
    df = materialize_raw_csv(tmp_path / "wine.csv")
    assert df.select((pl.all().is_null() | pl.all().is_nan()).sum()).sum_horizontal().item() == 0


def test_split_is_stratified(tmp_path):
    df = materialize_raw_csv(tmp_path / "wine.csv")
    train_df, test_df = train_test_split_stratified(df, test_size=0.2, seed=42)

    train_proportions = (
        train_df[TARGET_COLUMN].value_counts(normalize=True).sort(TARGET_COLUMN)["proportion"]
    )
    test_proportions = (
        test_df[TARGET_COLUMN].value_counts(normalize=True).sort(TARGET_COLUMN)["proportion"]
    )

    # Stratified splitting should keep class proportions close between train and test,
    # not just "some of each class made it in".
    max_deviation = (train_proportions - test_proportions).abs().max()
    assert isinstance(max_deviation, float)
    assert max_deviation < 0.05


def test_no_overlap_between_train_and_test(tmp_path):
    df = materialize_raw_csv(tmp_path / "wine.csv")
    train_df, test_df = train_test_split_stratified(df, test_size=0.2, seed=42)

    train_rows = set(train_df.iter_rows())
    test_rows = set(test_df.iter_rows())
    assert train_rows.isdisjoint(test_rows)

    assert len(train_df) + len(test_df) == len(df)


def test_split_features_target_shapes(tmp_path):
    df = materialize_raw_csv(tmp_path / "wine.csv")
    X, y = split_features_target(df)
    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert TARGET_COLUMN not in X.columns
