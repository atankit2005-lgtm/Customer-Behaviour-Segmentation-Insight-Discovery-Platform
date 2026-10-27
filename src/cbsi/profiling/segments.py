from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cbsi.features.customer import CUSTOMER_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS

PROFILED_CUSTOMER_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "cluster",
    *CUSTOMER_FEATURE_COLUMNS[1:],
)
INDEX_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURE_COLUMNS
LABEL_FEATURES: Final[tuple[str, ...]] = (
    "monetary",
    "frequency",
    "recency_days",
    "active_days",
    "unique_products",
    "avg_order_value",
)
LABEL_MIN_RELATIVE_DEVIATION: Final[float] = 0.10

_LABEL_PHRASES: Final[dict[str, tuple[str, str]]] = {
    "recency_days": ("Recent", "Less-Recent"),
    "frequency": ("Infrequent", "Frequent"),
    "monetary": ("Lower-Value", "Higher-Value"),
    "total_quantity": ("Lower-Quantity", "Higher-Quantity"),
    "active_days": ("Less-Active", "Highly-Active"),
    "avg_order_value": ("Lower-Order-Value", "High-Order-Value"),
    "avg_items_per_invoice": ("Small-Basket", "Large-Basket"),
    "unique_products": ("Focused-Variety", "Broad-Variety"),
    "purchase_span_days": ("Short-History", "Long-History"),
}
_DESCRIPTION_PHRASES: Final[dict[str, tuple[str, str]]] = {
    "recency_days": ("more recent purchasing", "less recent purchasing"),
    "frequency": ("lower purchase frequency", "higher purchase frequency"),
    "monetary": ("lower monetary value", "higher monetary value"),
    "total_quantity": ("lower purchased quantity", "higher purchased quantity"),
    "active_days": ("fewer active purchase days", "more active purchase days"),
    "avg_order_value": ("lower average order value", "higher average order value"),
    "avg_items_per_invoice": ("fewer items per invoice", "more items per invoice"),
    "unique_products": ("narrower product variety", "broader product variety"),
    "purchase_span_days": ("shorter purchase history", "longer purchase history"),
}

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "cluster",
    "segment_label",
    "description",
    "characteristics",
    "customer_count",
    "customer_share",
    "dominant_country",
    "dominant_country_share",
    "monetary_total",
    "total_quantity_total",
    *(
        column
        for feature in NUMERIC_FEATURE_COLUMNS
        for column in (f"{feature}_mean", f"{feature}_median", f"{feature}_index")
    ),
)


@dataclass(frozen=True)
class SegmentProfiles:
    """Customer-level assignments and their segment-level analytical summary."""

    customers: pd.DataFrame
    summary: pd.DataFrame


def build_profiled_customers(
    customer_features: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one cluster assignment to every Stage 4 customer without changing order."""
    _validate_customer_features(customer_features)
    _validate_assignments(assignments)

    feature_ids = set(customer_features["customer_id"])
    assignment_ids = set(assignments["customer_id"])
    unknown_ids = assignment_ids - feature_ids
    if unknown_ids:
        raise ValueError(f"Assignments contain unknown customer IDs: {list(unknown_ids)[:5]}")
    unassigned_ids = feature_ids - assignment_ids
    if unassigned_ids:
        raise ValueError(f"Customer features are missing assignments: {list(unassigned_ids)[:5]}")

    features = customer_features.loc[:, list(CUSTOMER_FEATURE_COLUMNS)].copy(deep=True)
    ordered_assignments = assignments.loc[:, ["customer_id", "cluster"]].copy(deep=True)
    ordered_assignments["cluster"] = pd.to_numeric(
        ordered_assignments["cluster"], errors="coerce"
    ).astype(int)
    profiled = ordered_assignments.merge(
        features,
        on="customer_id",
        how="left",
        sort=False,
        validate="one_to_one",
    ).loc[:, list(PROFILED_CUSTOMER_COLUMNS)]

    _validate_profiled_customers(profiled)
    if len(profiled) != len(assignments):
        raise ValueError("Profiling join changed the number of assigned customers")
    return profiled


def build_segment_summary(profiled_customers: pd.DataFrame) -> pd.DataFrame:
    """Summarize profiles; a recency index below 1 indicates more recent purchasing."""
    _validate_profiled_customers(profiled_customers)
    working = profiled_customers.loc[:, list(PROFILED_CUSTOMER_COLUMNS)].copy(deep=True)
    total_customers = len(working)
    overall_means = working.loc[:, list(NUMERIC_FEATURE_COLUMNS)].mean()

    records = []
    for cluster, segment in working.groupby("cluster", sort=True):
        customer_count = len(segment)
        country_counts = (
            segment.groupby("country", sort=False)
            .size()
            .rename("customer_count")
            .reset_index()
            .sort_values(
                ["customer_count", "country"],
                ascending=[False, True],
                kind="stable",
            )
        )
        dominant_country = country_counts.iloc[0]
        record: dict[str, object] = {
            "cluster": int(cluster),
            "customer_count": customer_count,
            "customer_share": customer_count / total_customers,
            "dominant_country": dominant_country["country"],
            "dominant_country_share": dominant_country["customer_count"] / customer_count,
            "monetary_total": float(segment["monetary"].sum()),
            "total_quantity_total": float(segment["total_quantity"].sum()),
        }
        indices: dict[str, float] = {}
        for feature in NUMERIC_FEATURE_COLUMNS:
            segment_mean = float(segment[feature].mean())
            segment_median = float(segment[feature].median())
            index = _relative_index(segment_mean, float(overall_means[feature]))
            record[f"{feature}_mean"] = segment_mean
            record[f"{feature}_median"] = segment_median
            record[f"{feature}_index"] = index
            indices[feature] = index

        characteristics = _segment_characteristics(indices)
        record["segment_label"] = _segment_label(indices)
        record["characteristics"] = characteristics
        record["description"] = _segment_description(characteristics)
        records.append(record)

    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    _validate_summary(summary, total_customers)
    return summary


def build_segment_profiles(
    customer_features: pd.DataFrame,
    assignments: pd.DataFrame,
) -> SegmentProfiles:
    """Build aligned customer profiles and their deterministic segment summary."""
    customers = build_profiled_customers(customer_features, assignments)
    summary = build_segment_summary(customers)
    return SegmentProfiles(customers=customers, summary=summary)


def _validate_customer_features(customer_features: pd.DataFrame) -> None:
    missing = [
        column for column in CUSTOMER_FEATURE_COLUMNS if column not in customer_features.columns
    ]
    if missing:
        raise ValueError(f"Customer features are missing required column(s): {missing}")
    if customer_features.empty:
        raise ValueError("Customer features must not be empty")
    if customer_features["customer_id"].isna().any():
        raise ValueError("Customer feature IDs must not be missing")
    if customer_features["customer_id"].duplicated().any():
        raise ValueError("Customer feature IDs must be unique")

    for feature in NUMERIC_FEATURE_COLUMNS:
        if not is_numeric_dtype(customer_features[feature]):
            raise ValueError(f"Customer feature {feature} must be numeric")
    values = customer_features.loc[:, list(NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Customer numeric features must be finite")
    if customer_features.loc[:, list(NUMERIC_FEATURE_COLUMNS[:-1])].le(0).any().any():
        raise ValueError("Customer numeric features must be positive")
    if customer_features["purchase_span_days"].lt(0).any():
        raise ValueError("purchase_span_days must be non-negative")
    if customer_features["country"].isna().any():
        raise ValueError("Customer countries must not be missing")
    if (
        not customer_features["country"]
        .map(lambda country: isinstance(country, str) and bool(country.strip()))
        .all()
    ):
        raise ValueError("Customer countries must be non-empty strings")


def _validate_assignments(assignments: pd.DataFrame) -> None:
    missing = [column for column in ("customer_id", "cluster") if column not in assignments.columns]
    if missing:
        raise ValueError(f"Assignments are missing required column(s): {missing}")
    if assignments.empty:
        raise ValueError("Assignments must not be empty")
    if assignments["customer_id"].isna().any():
        raise ValueError("Assignment customer IDs must not be missing")
    if assignments["customer_id"].duplicated().any():
        raise ValueError("Assignment customer IDs must be unique")

    clusters = pd.to_numeric(assignments["cluster"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(clusters).all():
        raise ValueError("Cluster assignments must be finite and not missing")
    if not np.equal(clusters, np.floor(clusters)).all():
        raise ValueError("Cluster assignments must be integer labels")


def _validate_profiled_customers(profiled_customers: pd.DataFrame) -> None:
    missing = [
        column for column in PROFILED_CUSTOMER_COLUMNS if column not in profiled_customers.columns
    ]
    if missing:
        raise ValueError(f"Profiled customers are missing required column(s): {missing}")
    _validate_customer_features(profiled_customers.loc[:, list(CUSTOMER_FEATURE_COLUMNS)])
    _validate_assignments(profiled_customers.loc[:, ["customer_id", "cluster"]])


def _relative_index(segment_mean: float, overall_mean: float) -> float:
    if overall_mean == 0:
        if segment_mean == 0:
            return 1.0
        raise ValueError("Relative indices require a positive overall mean")
    index = segment_mean / overall_mean
    if not np.isfinite(index):
        raise ValueError("Calculated relative indices must be finite")
    return index


def _segment_label(indices: dict[str, float]) -> str:
    ranked = sorted(
        LABEL_FEATURES,
        key=lambda feature: (-abs(indices[feature] - 1.0), LABEL_FEATURES.index(feature)),
    )
    distinctive = [
        feature for feature in ranked if abs(indices[feature] - 1.0) >= LABEL_MIN_RELATIVE_DEVIATION
    ][:2]
    if not distinctive:
        return "Typical Customers"
    phrases = [
        _directional_phrase(feature, indices[feature], _LABEL_PHRASES) for feature in distinctive
    ]
    return f"{', '.join(phrases)} Customers"


def _segment_characteristics(indices: dict[str, float]) -> tuple[str, ...]:
    ranked = sorted(
        INDEX_FEATURES,
        key=lambda feature: (-abs(indices[feature] - 1.0), INDEX_FEATURES.index(feature)),
    )
    distinctive = [feature for feature in ranked if not np.isclose(indices[feature], 1.0)][:3]
    if not distinctive:
        return ("typical levels across measured features",)
    return tuple(
        f"{_directional_phrase(feature, indices[feature], _DESCRIPTION_PHRASES)} "
        f"({indices[feature]:.2f}x overall)"
        for feature in distinctive
    )


def _directional_phrase(
    feature: str,
    index: float,
    phrases: dict[str, tuple[str, str]],
) -> str:
    lower, higher = phrases[feature]
    return lower if index < 1 else higher


def _segment_description(characteristics: tuple[str, ...]) -> str:
    return f"{'; '.join(characteristics).capitalize()}."


def _validate_summary(summary: pd.DataFrame, total_customers: int) -> None:
    if summary.empty:
        raise ValueError("Segment summary must not be empty")
    if summary["cluster"].duplicated().any():
        raise ValueError("Segment summary requires one row per cluster")
    if summary["customer_count"].sum() != total_customers:
        raise ValueError("Segment counts must sum to the profiled customer population")
    if not np.isclose(summary["customer_share"].sum(), 1.0):
        raise ValueError("Segment customer shares must sum to 1")
    if summary["dominant_country"].isna().any():
        raise ValueError("Every segment must have a dominant country")

    numeric_columns = [
        column
        for column in summary.columns
        if column not in {"segment_label", "description", "characteristics", "dominant_country"}
    ]
    numeric = summary.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Segment summary statistics must be finite")
    if summary["cluster"].tolist() != sorted(summary["cluster"].tolist()):
        raise ValueError("Segment summary clusters must use deterministic ordering")
