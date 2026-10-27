from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cbsi.features.customer import CUSTOMER_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS

CLUSTERING_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURE_COLUMNS
STRICTLY_POSITIVE_FEATURES: Final[tuple[str, ...]] = tuple(
    feature for feature in CLUSTERING_FEATURES if feature != "purchase_span_days"
)


@dataclass(frozen=True)
class ClusteringDataset:
    customer_ids: pd.Series
    feature_names: tuple[str, ...]
    values: np.ndarray


def prepare_clustering_features(customer_features: pd.DataFrame) -> ClusteringDataset:
    """Apply log1p and standard scaling to ordered numeric customer features."""
    missing = [column for column in CUSTOMER_FEATURE_COLUMNS if column not in customer_features]
    if missing:
        raise ValueError(f"Customer features are missing required column(s): {missing}")
    if len(customer_features) < 2:
        raise ValueError("Clustering requires at least two customers")
    if customer_features["customer_id"].isna().any():
        raise ValueError("Customer IDs must not be missing")
    if customer_features["customer_id"].duplicated().any():
        raise ValueError("Customer IDs must be unique")

    selected = customer_features.loc[:, list(CLUSTERING_FEATURES)].copy(deep=True)
    selected = selected.apply(pd.to_numeric, errors="coerce")
    raw_values = selected.to_numpy(dtype=float)

    if not np.isfinite(raw_values).all():
        raise ValueError("Clustering features must contain only finite numeric values")
    if selected.loc[:, list(STRICTLY_POSITIVE_FEATURES)].le(0).any().any():
        raise ValueError("Clustering features must be positive except purchase_span_days")
    if selected["purchase_span_days"].lt(0).any():
        raise ValueError("purchase_span_days must be non-negative")

    logged = np.log1p(raw_values)
    if np.any(np.ptp(logged, axis=0) == 0):
        raise ValueError("Clustering features must vary across customers")

    values = StandardScaler().fit_transform(logged)
    if not np.isfinite(values).all():
        raise ValueError("Prepared clustering matrix contains non-finite values")

    return ClusteringDataset(
        customer_ids=customer_features["customer_id"].copy(deep=True).reset_index(drop=True),
        feature_names=CLUSTERING_FEATURES,
        values=values,
    )
