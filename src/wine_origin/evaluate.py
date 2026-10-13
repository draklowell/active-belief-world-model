"""Metrics, cross-validation, and the confusion-matrix figure."""

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def compute_metrics(y_true: pl.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute the metrics this project reports everywhere: accuracy, macro-F1, and
    the confusion matrix.

    Why macro-F1 alongside accuracy: accuracy on a 3-class dataset can look fine while
    the model is quietly bad at the smallest class. Macro-F1 averages the F1 score of
    each class equally, so a model that ignores a minority class gets penalized for it
    even if overall accuracy stays high.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cross_validate_accuracy(
    model: BaseEstimator, X: pl.DataFrame, y: pl.Series, n_splits: int, seed: int
) -> dict[str, float]:
    """Report mean +/- std accuracy across ``n_splits`` stratified folds.

    Why this matters: a single train/test split gives you one number, and one number
    can't tell you whether a model is reliably good or just got a lucky split. The
    standard deviation across folds is what tells you how much to trust the mean.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(model), X, y, cv=cv, scoring="accuracy")
    return {"cv_mean_accuracy": float(scores.mean()), "cv_std_accuracy": float(scores.std())}


def plot_confusion_matrix(
    cm: list[list[int]], class_names: list[str], output_path: str | Path
) -> None:
    """Render and save a confusion-matrix heatmap."""
    cm_array = np.array(cm)
    text = [
        [str(cm_array[i, j]) for j in range(cm_array.shape[1])] for i in range(cm_array.shape[0])
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm_array,
            x=class_names,
            y=class_names,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis_autorange="reversed",
        width=500,
        height=400,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
