from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from cbsi.churn.dataset import CHURN_FEATURE_COLUMNS, CHURN_TARGET_COLUMN, ChurnDataset

CHURN_MODEL_NAMES: Final[tuple[str, ...]] = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
)
DEFAULT_CLASSIFICATION_THRESHOLD: Final[float] = 0.50
RANDOM_STATE: Final[int] = 42
COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "threshold",
    "positive_prevalence",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "selection_rank",
)


@dataclass(frozen=True)
class ChurnMetrics:
    positive_prevalence: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class FittedChurnModel:
    model_name: str
    estimator: Pipeline = field(repr=False, compare=False)
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD
    feature_names: tuple[str, ...] = CHURN_FEATURE_COLUMNS


@dataclass(frozen=True)
class ChurnModelResult:
    model: FittedChurnModel = field(repr=False, compare=False)
    metrics: ChurnMetrics

    @property
    def model_name(self) -> str:
        return self.model.model_name


@dataclass(frozen=True)
class ChurnModelSelection:
    selected: ChurnModelResult
    comparison: pd.DataFrame = field(repr=False, compare=False)
    rationale: tuple[str, ...]


def evaluate_churn_models(
    train: ChurnDataset,
    validation: ChurnDataset,
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> tuple[ChurnModelResult, ...]:
    """Fit each benchmark on training data and evaluate on the later validation snapshot."""
    _validate_threshold(threshold)
    if train.prediction_end > validation.observation_end:
        raise ValueError("Training labels must not extend beyond validation observation features")

    results = []
    for model_name in CHURN_MODEL_NAMES:
        model = fit_churn_model(model_name, train, threshold=threshold)
        results.append(
            ChurnModelResult(model=model, metrics=evaluate_churn_model(model, validation))
        )
    return tuple(results)


def select_best_churn_model(
    results: Iterable[ChurnModelResult],
) -> ChurnModelSelection:
    """Select by PR-AUC, then ROC-AUC, F1, and model name."""
    candidates = tuple(results)
    if not candidates:
        raise ValueError("At least one churn model result is required")
    names = [candidate.model_name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Churn model result names must be unique")
    for candidate in candidates:
        values = (
            candidate.metrics.pr_auc,
            candidate.metrics.roc_auc,
            candidate.metrics.f1,
        )
        if not np.isfinite(values).all():
            raise ValueError("Churn model selection metrics must be finite")

    ranked = sorted(candidates, key=_selection_key)
    comparison = pd.DataFrame(
        [_comparison_record(candidate, rank) for rank, candidate in enumerate(ranked, start=1)],
        columns=COMPARISON_COLUMNS,
    )
    selected = ranked[0]
    rationale = (
        "Models were fitted on the training snapshot and ranked on the later validation snapshot.",
        "Average precision (PR-AUC) is primary; ROC-AUC and F1 are deterministic tie-breakers.",
        f"Selected {selected.model_name} at the fixed threshold {selected.model.threshold:.2f}.",
    )
    return ChurnModelSelection(selected=selected, comparison=comparison, rationale=rationale)


def fit_churn_model(
    model_name: str,
    datasets: ChurnDataset | Iterable[ChurnDataset],
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> FittedChurnModel:
    """Fit one deterministic supervised pipeline on one or more historical snapshots."""
    _validate_threshold(threshold)
    if model_name not in CHURN_MODEL_NAMES:
        raise ValueError(f"Unsupported churn model: {model_name}")
    dataset_list = _dataset_tuple(datasets)
    features = pd.concat(
        [dataset.data.loc[:, list(CHURN_FEATURE_COLUMNS)] for dataset in dataset_list],
        ignore_index=True,
    )
    target = pd.concat(
        [dataset.data[CHURN_TARGET_COLUMN] for dataset in dataset_list],
        ignore_index=True,
    )
    if set(target.unique()) != {0, 1}:
        raise ValueError("Churn model training requires both target classes")

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", _classifier(model_name)),
        ]
    )
    if model_name == "gradient_boosting":
        sample_weights = compute_sample_weight(class_weight="balanced", y=target)
        estimator.fit(features, target, classifier__sample_weight=sample_weights)
    else:
        estimator.fit(features, target)
    return FittedChurnModel(
        model_name=model_name,
        estimator=estimator,
        threshold=threshold,
    )


def evaluate_churn_model(model: FittedChurnModel, dataset: ChurnDataset) -> ChurnMetrics:
    """Evaluate one fitted model at its documented operating threshold."""
    target = dataset.data[CHURN_TARGET_COLUMN].to_numpy(dtype=int)
    probabilities = _predict_probabilities(model, dataset)
    predictions = (probabilities >= model.threshold).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    metrics = ChurnMetrics(
        positive_prevalence=float(target.mean()),
        roc_auc=float(roc_auc_score(target, probabilities)),
        pr_auc=float(average_precision_score(target, probabilities)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
        confusion_matrix=(
            (true_negative, false_positive),
            (false_negative, true_positive),
        ),
    )
    if not np.isfinite(
        (
            metrics.positive_prevalence,
            metrics.roc_auc,
            metrics.pr_auc,
            metrics.precision,
            metrics.recall,
            metrics.f1,
        )
    ).all():
        raise ValueError("Churn evaluation metrics must be finite")
    return metrics


def predict_churn_probabilities(
    model: FittedChurnModel,
    dataset: ChurnDataset,
) -> np.ndarray:
    """Return positive-class probabilities in dataset customer order."""
    return _predict_probabilities(model, dataset).copy()


def _classifier(
    model_name: str,
) -> LogisticRegression | RandomForestClassifier | GradientBoostingClassifier:
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )


def _predict_probabilities(model: FittedChurnModel, dataset: ChurnDataset) -> np.ndarray:
    features = dataset.data.loc[:, list(model.feature_names)]
    classes = np.asarray(model.estimator.classes_)
    positive_indexes = np.flatnonzero(classes == 1)
    if len(positive_indexes) != 1:
        raise ValueError("Fitted churn model must contain the positive class")
    probabilities = model.estimator.predict_proba(features)[:, int(positive_indexes[0])]
    if len(probabilities) != dataset.customer_count:
        raise ValueError("Churn probability output must contain one value per customer")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Churn probabilities must be finite and in [0, 1]")
    return np.asarray(probabilities, dtype=float)


def _dataset_tuple(
    datasets: ChurnDataset | Iterable[ChurnDataset],
) -> tuple[ChurnDataset, ...]:
    result: tuple[ChurnDataset, ...]
    if isinstance(datasets, ChurnDataset):
        result = (datasets,)
    else:
        result = tuple(datasets)
    if not result:
        raise ValueError("At least one churn dataset is required for model fitting")
    if any(not isinstance(dataset, ChurnDataset) for dataset in result):
        raise ValueError("Model fitting requires ChurnDataset inputs")
    ordered = sorted(result, key=lambda dataset: dataset.observation_end)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if earlier.prediction_end > later.observation_end:
            raise ValueError("Training snapshot labels must not extend beyond later features")
    return tuple(ordered)


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Classification threshold must be finite and in (0, 1)")


def _selection_key(result: ChurnModelResult) -> tuple[float, float, float, str]:
    return (
        -result.metrics.pr_auc,
        -result.metrics.roc_auc,
        -result.metrics.f1,
        result.model_name,
    )


def _comparison_record(result: ChurnModelResult, rank: int) -> dict[str, object]:
    matrix = result.metrics.confusion_matrix
    return {
        "model": result.model_name,
        "threshold": result.model.threshold,
        "positive_prevalence": result.metrics.positive_prevalence,
        "roc_auc": result.metrics.roc_auc,
        "pr_auc": result.metrics.pr_auc,
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "f1": result.metrics.f1,
        "true_negative": matrix[0][0],
        "false_positive": matrix[0][1],
        "false_negative": matrix[1][0],
        "true_positive": matrix[1][1],
        "selection_rank": rank,
    }
