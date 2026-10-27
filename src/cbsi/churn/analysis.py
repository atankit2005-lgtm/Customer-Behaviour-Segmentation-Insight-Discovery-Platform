from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from cbsi.churn.dataset import CHURN_TARGET_COLUMN, ChurnDataset
from cbsi.churn.models import FittedChurnModel, predict_churn_probabilities

RISK_BANDS: Final[tuple[str, ...]] = ("High", "Medium", "Low")
HIGH_RISK_THRESHOLD: Final[float] = 0.70
MEDIUM_RISK_THRESHOLD: Final[float] = 0.40
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "actual_churn",
    "churn_probability",
    "churn_prediction",
    "risk_band",
)
FEATURE_IMPORTANCE_COLUMNS: Final[tuple[str, ...]] = (
    "feature",
    "importance",
    "direction",
    "rank",
)
SEGMENT_CHURN_COLUMNS: Final[tuple[str, ...]] = (
    "segment_label",
    "customer_count",
    "average_churn_probability",
    "churn_rate",
    "high_risk_customer_count",
    "high_risk_share",
)


def predict_churn_risk(
    model: FittedChurnModel,
    dataset: ChurnDataset,
) -> pd.DataFrame:
    """Build deterministic customer-level probabilities, predictions, and risk bands."""
    probabilities = predict_churn_probabilities(model, dataset)
    predictions = (probabilities >= model.threshold).astype(int)
    risk_bands = np.select(
        [probabilities >= HIGH_RISK_THRESHOLD, probabilities >= MEDIUM_RISK_THRESHOLD],
        ["High", "Medium"],
        default="Low",
    )
    result = pd.DataFrame(
        {
            "customer_id": dataset.data["customer_id"].to_numpy(copy=True),
            "actual_churn": dataset.data[CHURN_TARGET_COLUMN].to_numpy(dtype=int, copy=True),
            "churn_probability": probabilities,
            "churn_prediction": predictions,
            "risk_band": risk_bands,
        },
        columns=PREDICTION_COLUMNS,
    )
    _validate_predictions(result, dataset.customer_count)
    return result


def build_churn_feature_importance(model: FittedChurnModel) -> pd.DataFrame:
    """Rank native model importance values against the supervised feature schema."""
    classifier = model.estimator.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        signed = np.asarray(classifier.coef_, dtype=float)[0]
        importance = np.abs(signed)
        direction = np.where(signed >= 0, "increases_risk", "decreases_risk")
    elif hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_, dtype=float)
        direction = np.repeat("not_available", len(importance))
    else:
        raise ValueError("Selected churn model does not expose native feature importance")
    if len(importance) != len(model.feature_names):
        raise ValueError("Feature importance values must align with churn feature names")
    if not np.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("Feature importance values must be finite and non-negative")

    result = pd.DataFrame(
        {
            "feature": model.feature_names,
            "importance": importance,
            "direction": direction,
        }
    )
    result = result.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result.loc[:, list(FEATURE_IMPORTANCE_COLUMNS)]


def build_segment_churn_summary(
    predictions: pd.DataFrame,
    profiled_customers: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate scored customers by behavior-derived Stage 7 segment labels."""
    _validate_segment_inputs(predictions, profiled_customers, segment_summary)
    assignments = profiled_customers.loc[:, ["customer_id", "cluster"]].copy(deep=True)
    labels = segment_summary.loc[:, ["cluster", "segment_label"]].copy(deep=True)
    joined = predictions.merge(
        assignments,
        on="customer_id",
        how="left",
        sort=False,
        validate="one_to_one",
    ).merge(
        labels,
        on="cluster",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if joined["segment_label"].isna().any() or len(joined) != len(predictions):
        raise ValueError("Every churn prediction must map to one segment label")

    joined["is_high_risk"] = joined["risk_band"].eq("High")
    result = (
        joined.groupby("segment_label", sort=True)
        .agg(
            customer_count=("customer_id", "size"),
            average_churn_probability=("churn_probability", "mean"),
            churn_rate=("actual_churn", "mean"),
            high_risk_customer_count=("is_high_risk", "sum"),
        )
        .reset_index()
    )
    result["high_risk_customer_count"] = result["high_risk_customer_count"].astype(int)
    result["high_risk_share"] = result["high_risk_customer_count"] / result["customer_count"]
    result = result.loc[:, list(SEGMENT_CHURN_COLUMNS)]
    _validate_segment_output(result, len(predictions))
    return result


def _validate_predictions(predictions: pd.DataFrame, expected_customers: int) -> None:
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("Churn predictions do not match the required schema")
    if len(predictions) != expected_customers:
        raise ValueError("Churn predictions must contain one row per evaluated customer")
    if predictions["customer_id"].isna().any() or predictions["customer_id"].duplicated().any():
        raise ValueError("Churn predictions require unique, non-missing customer IDs")
    probabilities = predictions["churn_probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Churn probabilities must be finite and in [0, 1]")
    if not predictions["churn_prediction"].isin([0, 1]).all():
        raise ValueError("Churn predictions must be binary")
    if not predictions["actual_churn"].isin([0, 1]).all():
        raise ValueError("Actual churn values must be binary")
    if not predictions["risk_band"].isin(RISK_BANDS).all():
        raise ValueError("Churn risk bands must be valid")


def _validate_segment_inputs(
    predictions: pd.DataFrame,
    profiled_customers: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> None:
    missing_prediction = [
        column for column in PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing_prediction:
        raise ValueError(f"Churn predictions are missing required column(s): {missing_prediction}")
    if predictions.empty:
        raise ValueError("Churn predictions must not be empty")
    _validate_predictions(predictions.loc[:, list(PREDICTION_COLUMNS)], len(predictions))

    for frame, name, required in (
        (profiled_customers, "Profiled customers", ("customer_id", "cluster")),
        (segment_summary, "Segment summary", ("cluster", "segment_label")),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required column(s): {missing}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
    if (
        profiled_customers["customer_id"].isna().any()
        or profiled_customers["customer_id"].duplicated().any()
    ):
        raise ValueError("Profiled customer IDs must be unique and non-missing")
    if segment_summary["cluster"].duplicated().any():
        raise ValueError("Segment summary clusters must be unique")
    if set(predictions["customer_id"]) != set(profiled_customers["customer_id"]):
        raise ValueError("Segment/churn join requires exact customer coverage")


def _validate_segment_output(summary: pd.DataFrame, expected_customers: int) -> None:
    if summary["customer_count"].sum() != expected_customers:
        raise ValueError("Segment churn summary must preserve exact customer coverage")
    numeric = summary.loc[:, list(SEGMENT_CHURN_COLUMNS[1:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Segment churn summary values must be finite")
    for column in ("average_churn_probability", "churn_rate", "high_risk_share"):
        if not summary[column].between(0, 1).all():
            raise ValueError(f"Segment churn summary {column} must be in [0, 1]")
