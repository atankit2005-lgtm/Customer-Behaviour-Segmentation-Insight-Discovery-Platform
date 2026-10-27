from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from cbsi.data.schema import TRANSACTION_COLUMNS
from cbsi.features.customer import NUMERIC_FEATURE_COLUMNS, build_customer_features

PREDICTION_WINDOW_DAYS: Final[int] = 90
CHURN_TARGET_COLUMN: Final[str] = "churned"
CHURN_FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_FEATURE_COLUMNS
CHURN_DATA_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    *CHURN_FEATURE_COLUMNS,
    CHURN_TARGET_COLUMN,
)
TARGET_DEFINITION: Final[str] = (
    "1 = no qualifying purchase in (observation_end, prediction_end]; "
    "0 = at least one qualifying purchase in that window"
)


@dataclass(frozen=True)
class ChurnDataset:
    data: pd.DataFrame
    observation_end: pd.Timestamp
    prediction_end: pd.Timestamp
    observation_transaction_count: int
    prediction_transaction_count: int

    @property
    def customer_count(self) -> int:
        return len(self.data)

    @property
    def prevalence(self) -> float:
        return float(self.data[CHURN_TARGET_COLUMN].mean())


@dataclass(frozen=True)
class TemporalChurnSplit:
    train: ChurnDataset
    validation: ChurnDataset
    test: ChurnDataset


def build_churn_dataset(
    transactions: pd.DataFrame,
    observation_end: str | pd.Timestamp,
    *,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
) -> ChurnDataset:
    """Build one leakage-safe customer snapshot and its future inactivity target."""
    working = _validated_transactions(transactions)
    cutoff = _validated_cutoff(observation_end)
    horizon = _validated_horizon(prediction_window_days)
    return _build_churn_dataset(working, cutoff, horizon)


def build_temporal_churn_split(
    transactions: pd.DataFrame,
    *,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
) -> TemporalChurnSplit:
    """Build consecutive train, validation, and final test snapshots."""
    working = _validated_transactions(transactions)
    horizon = _validated_horizon(prediction_window_days)
    maximum_date = working["invoice_date"].max()

    test_cutoff = maximum_date - horizon
    validation_cutoff = test_cutoff - horizon
    train_cutoff = validation_cutoff - horizon

    split = TemporalChurnSplit(
        train=_build_churn_dataset(working, train_cutoff, horizon),
        validation=_build_churn_dataset(working, validation_cutoff, horizon),
        test=_build_churn_dataset(working, test_cutoff, horizon),
    )
    _validate_temporal_split(split)
    return split


def _build_churn_dataset(
    transactions: pd.DataFrame,
    observation_end: pd.Timestamp,
    horizon: pd.Timedelta,
) -> ChurnDataset:
    prediction_end = observation_end + horizon
    minimum_date = transactions["invoice_date"].min()
    maximum_date = transactions["invoice_date"].max()
    if observation_end < minimum_date:
        raise ValueError("Observation end must not precede the first eligible transaction")
    if prediction_end > maximum_date:
        raise ValueError("Prediction window must not extend beyond the available transaction data")

    observation_mask = transactions["invoice_date"].le(observation_end)
    prediction_mask = transactions["invoice_date"].gt(observation_end) & transactions[
        "invoice_date"
    ].le(prediction_end)
    observation = transactions.loc[observation_mask, list(TRANSACTION_COLUMNS)].copy(deep=True)
    future = transactions.loc[prediction_mask, list(TRANSACTION_COLUMNS)].copy(deep=True)
    if observation.empty:
        raise ValueError("Observation period must contain eligible transactions")
    if future.empty:
        raise ValueError("Prediction window must contain eligible transactions")

    customer_features = build_customer_features(observation)
    latest_purchases = observation.groupby("customer_id", sort=True)["invoice_date"].max()
    reference_date = observation_end + pd.Timedelta(days=1)
    recency = (reference_date - latest_purchases).dt.days
    customer_features["recency_days"] = customer_features["customer_id"].map(recency)

    future_customer_ids = set(future["customer_id"])
    customer_features[CHURN_TARGET_COLUMN] = (
        ~customer_features["customer_id"].isin(future_customer_ids)
    ).astype(int)
    data = customer_features.loc[:, list(CHURN_DATA_COLUMNS)].copy(deep=True)
    _validate_churn_data(data)

    return ChurnDataset(
        data=data,
        observation_end=observation_end,
        prediction_end=prediction_end,
        observation_transaction_count=len(observation),
        prediction_transaction_count=len(future),
    )


def _validated_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TRANSACTION_COLUMNS if column not in transactions.columns]
    if missing:
        raise ValueError(f"Transaction data is missing required column(s): {missing}")
    if transactions.empty:
        raise ValueError("Transaction data must not be empty")

    working = transactions.loc[:, list(TRANSACTION_COLUMNS)].copy(deep=True)
    working["invoice_date"] = pd.to_datetime(working["invoice_date"], errors="coerce")
    working["quantity"] = pd.to_numeric(working["quantity"], errors="coerce")
    working["unit_price"] = pd.to_numeric(working["unit_price"], errors="coerce")
    if working["invoice_date"].isna().any():
        raise ValueError("Stage 3 transactions must contain valid invoice dates")
    if working["customer_id"].isna().any():
        raise ValueError("Stage 3 transactions must not contain missing customer IDs")
    if working.duplicated().any():
        raise ValueError("Stage 3 transactions must not contain exact duplicates")
    if working["invoice_no"].astype("string").str.startswith("C", na=False).any():
        raise ValueError("Stage 3 transactions must not contain cancellations")
    for column in ("quantity", "unit_price"):
        values = working[column]
        if (~np.isfinite(values) | values.le(0)).any():
            raise ValueError(f"Stage 3 transactions must contain positive finite {column} values")
    return working


def _validated_cutoff(observation_end: str | pd.Timestamp) -> pd.Timestamp:
    try:
        cutoff = pd.Timestamp(observation_end)
    except (TypeError, ValueError) as error:
        raise ValueError("Observation end must be a valid timestamp") from error
    if pd.isna(cutoff):
        raise ValueError("Observation end must be a valid timestamp")
    if cutoff.tz is not None:
        raise ValueError("Observation end must be timezone-naive")
    return cutoff


def _validated_horizon(prediction_window_days: int) -> pd.Timedelta:
    if not isinstance(prediction_window_days, int) or prediction_window_days <= 0:
        raise ValueError("Prediction window days must be a positive integer")
    return pd.Timedelta(days=prediction_window_days)


def _validate_churn_data(data: pd.DataFrame) -> None:
    if data.columns.tolist() != list(CHURN_DATA_COLUMNS):
        raise ValueError("Churn data does not match the required schema")
    if data["customer_id"].isna().any() or data["customer_id"].duplicated().any():
        raise ValueError("Churn data requires unique, non-missing customer IDs")
    numeric = data.loc[:, list(CHURN_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Churn features must be finite")
    targets = set(data[CHURN_TARGET_COLUMN].unique())
    if targets != {0, 1}:
        raise ValueError("Churn target must contain both binary classes")


def _validate_temporal_split(split: TemporalChurnSplit) -> None:
    if split.train.prediction_end != split.validation.observation_end:
        raise ValueError("Training prediction window must end at validation observation end")
    if split.validation.prediction_end != split.test.observation_end:
        raise ValueError("Validation prediction window must end at test observation end")
    if not (
        split.train.observation_end < split.validation.observation_end < split.test.observation_end
    ):
        raise ValueError("Temporal churn snapshots must be strictly ordered")
