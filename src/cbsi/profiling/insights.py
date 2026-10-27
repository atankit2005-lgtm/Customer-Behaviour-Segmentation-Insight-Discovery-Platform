from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cbsi.features.customer import NUMERIC_FEATURE_COLUMNS
from cbsi.profiling.segments import SUMMARY_COLUMNS

STRATEGY_CATEGORIES: Final[tuple[str, ...]] = (
    "RETENTION",
    "REACTIVATION",
    "VALUE_DEVELOPMENT",
    "LOYALTY",
    "CROSS_SELL",
    "ENGAGEMENT",
)
PRIORITIES: Final[tuple[str, ...]] = ("High", "Medium", "Low")
HIGH_INDEX_THRESHOLD: Final[float] = 1.20
LOW_INDEX_THRESHOLD: Final[float] = 0.80

INSIGHT_COLUMNS: Final[tuple[str, ...]] = (
    "cluster",
    "segment_label",
    "customer_count",
    "customer_share",
    "strategy_category",
    "priority",
    "primary_objective",
    "recommended_actions",
    "key_risks",
    "key_opportunities",
    "rationale",
    "recommended_kpis",
)

_ACTIONS: Final[dict[str, tuple[str, ...]]] = {
    "RETENTION": (
        "Prioritize personalized re-engagement for previously valuable customers.",
        "Monitor recency changes and adjust contact cadence accordingly.",
        "Test proportionate win-back incentives with a measured control group.",
    ),
    "REACTIVATION": (
        "Use low-cost reactivation messages focused on the next purchase.",
        "Test modest return offers with a measured control group.",
        "Monitor recency and repeat activity before increasing treatment cost.",
    ),
    "VALUE_DEVELOPMENT": (
        "Encourage the next purchase through relevant product discovery.",
        "Recommend complementary products based on observed purchase history.",
        "Test frequency-building offers while controlling incentive cost.",
    ),
    "LOYALTY": (
        "Provide loyalty recognition proportional to established purchase activity.",
        "Personalize recommendations using observed product history.",
        "Test complementary-product cross-sell while monitoring customer value.",
    ),
    "CROSS_SELL": (
        "Recommend complementary products from categories not yet purchased.",
        "Use product-discovery communications to broaden considered items.",
        "Track product breadth before and after measured cross-sell tests.",
    ),
    "ENGAGEMENT": (
        "Maintain relevant communications based on observed purchase behavior.",
        "Test product-discovery content without intensive incentives.",
        "Monitor frequency, active days, and recency for meaningful changes.",
    ),
}

_OBJECTIVE_KPIS: Final[dict[str, tuple[str, ...]]] = {
    "Protect existing value": (
        "monetary",
        "frequency",
        "recency_days",
        "avg_order_value",
    ),
    "Reactivate inactive customers": (
        "recency_days",
        "frequency",
        "active_days",
        "repeat_purchase_rate (future measurement)",
    ),
    "Increase purchase frequency": (
        "frequency",
        "active_days",
        "recency_days",
        "repeat_purchase_rate (future measurement)",
    ),
    "Increase customer value": (
        "monetary",
        "avg_order_value",
        "frequency",
    ),
    "Increase product breadth": (
        "unique_products",
        "avg_items_per_invoice",
        "monetary",
    ),
    "Maintain customer engagement": (
        "frequency",
        "active_days",
        "recency_days",
    ),
}

_NON_NUMERIC_SUMMARY_COLUMNS: Final[set[str]] = {
    "segment_label",
    "description",
    "characteristics",
    "dominant_country",
}


def build_segment_insights(segment_summary: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic, descriptive business strategies from Stage 7 profiles."""
    _validate_segment_summary(segment_summary)
    working = segment_summary.loc[:, list(SUMMARY_COLUMNS)].copy(deep=True)
    working = working.sort_values("cluster", kind="stable").reset_index(drop=True)

    records = []
    for _, segment in working.iterrows():
        indices = {
            feature: float(segment[f"{feature}_index"]) for feature in NUMERIC_FEATURE_COLUMNS
        }
        category = _strategy_category(indices)
        objective = _primary_objective(category, indices)
        record = {
            "cluster": int(segment["cluster"]),
            "segment_label": segment["segment_label"],
            "customer_count": int(segment["customer_count"]),
            "customer_share": float(segment["customer_share"]),
            "strategy_category": category,
            "priority": _priority(category, indices),
            "primary_objective": objective,
            "recommended_actions": _ACTIONS[category],
            "key_risks": _key_risks(indices),
            "key_opportunities": _key_opportunities(indices),
            "rationale": _rationale(category, objective, indices),
            "recommended_kpis": _OBJECTIVE_KPIS[objective],
        }
        records.append(record)

    insights = pd.DataFrame(records, columns=INSIGHT_COLUMNS)
    _validate_insights(insights, len(segment_summary))
    return insights


def _strategy_category(indices: dict[str, float]) -> str:
    high_value = indices["monetary"] >= HIGH_INDEX_THRESHOLD
    low_value = indices["monetary"] <= LOW_INDEX_THRESHOLD
    high_frequency = indices["frequency"] >= HIGH_INDEX_THRESHOLD
    low_frequency = indices["frequency"] <= LOW_INDEX_THRESHOLD
    less_recent = indices["recency_days"] >= HIGH_INDEX_THRESHOLD
    low_breadth = indices["unique_products"] <= LOW_INDEX_THRESHOLD

    if high_value and less_recent:
        return "RETENTION"
    if high_value and high_frequency:
        return "LOYALTY"
    if low_frequency and less_recent:
        return "REACTIVATION"
    if low_value or low_frequency:
        return "VALUE_DEVELOPMENT"
    if high_value:
        return "LOYALTY"
    if low_breadth:
        return "CROSS_SELL"
    return "ENGAGEMENT"


def _priority(category: str, indices: dict[str, float]) -> str:
    high_value = indices["monetary"] >= HIGH_INDEX_THRESHOLD
    high_frequency = indices["frequency"] >= HIGH_INDEX_THRESHOLD
    less_recent = indices["recency_days"] >= HIGH_INDEX_THRESHOLD
    if high_value and (high_frequency or less_recent):
        return "High"
    if category in {"REACTIVATION", "VALUE_DEVELOPMENT", "CROSS_SELL", "LOYALTY"}:
        return "Medium"
    return "Low"


def _primary_objective(category: str, indices: dict[str, float]) -> str:
    if category in {"RETENTION", "LOYALTY"}:
        return "Protect existing value"
    if category == "REACTIVATION":
        return "Reactivate inactive customers"
    if category == "VALUE_DEVELOPMENT":
        if indices["frequency"] <= LOW_INDEX_THRESHOLD:
            return "Increase purchase frequency"
        return "Increase customer value"
    if category == "CROSS_SELL":
        return "Increase product breadth"
    return "Maintain customer engagement"


def _key_opportunities(indices: dict[str, float]) -> tuple[str, ...]:
    opportunities = []
    if indices["recency_days"] <= LOW_INDEX_THRESHOLD:
        opportunities.append("Purchasing is more recent than the overall population.")
    if indices["frequency"] >= HIGH_INDEX_THRESHOLD:
        opportunities.append("Purchase frequency is above the overall population.")
    if indices["monetary"] >= HIGH_INDEX_THRESHOLD:
        opportunities.append("Monetary value is above the overall population.")
    if indices["avg_order_value"] >= HIGH_INDEX_THRESHOLD:
        opportunities.append("Average order value is above the overall population.")
    if indices["unique_products"] >= HIGH_INDEX_THRESHOLD:
        opportunities.append("Product breadth is above the overall population.")
    if indices["frequency"] <= LOW_INDEX_THRESHOLD:
        opportunities.append("Purchase frequency has measurable room to develop.")
    if indices["unique_products"] <= LOW_INDEX_THRESHOLD:
        opportunities.append("Product breadth has measurable room to develop.")
    if indices["monetary"] <= LOW_INDEX_THRESHOLD:
        opportunities.append("Customer value has measurable room to develop.")
    if not opportunities:
        opportunities.append(
            "No strong relative opportunity signal; maintain baseline measurement."
        )
    return tuple(opportunities[:3])


def _key_risks(indices: dict[str, float]) -> tuple[str, ...]:
    risks = []
    if indices["recency_days"] >= HIGH_INDEX_THRESHOLD:
        risks.append("Purchasing is less recent than the overall population.")
    if indices["frequency"] <= LOW_INDEX_THRESHOLD:
        risks.append("Purchase frequency is below the overall population.")
    if indices["monetary"] <= LOW_INDEX_THRESHOLD:
        risks.append("Monetary value is below the overall population.")
    if indices["active_days"] <= LOW_INDEX_THRESHOLD:
        risks.append("Active purchase days are below the overall population.")
    if indices["unique_products"] <= LOW_INDEX_THRESHOLD:
        risks.append("Product breadth is below the overall population.")
    if indices["avg_order_value"] <= LOW_INDEX_THRESHOLD:
        risks.append("Average order value is below the overall population.")
    if not risks:
        risks.append("No strong relative risk signal; continue monitoring recency and frequency.")
    return tuple(risks[:3])


def _rationale(
    category: str,
    objective: str,
    indices: dict[str, float],
) -> str:
    value_signal = _level(indices["monetary"], "monetary value")
    frequency_signal = _level(indices["frequency"], "purchase frequency")
    recency_signal = _recency_level(indices["recency_days"])
    return (
        f"Monetary index {indices['monetary']:.2f} indicates {value_signal}; "
        f"frequency index {indices['frequency']:.2f} indicates {frequency_signal}; "
        f"recency index {indices['recency_days']:.2f} indicates {recency_signal}. "
        f"These measured signals support the {category.lower().replace('_', ' ')} strategy "
        f"and the objective to {objective.lower()}."
    )


def _level(index: float, measure: str) -> str:
    if index >= HIGH_INDEX_THRESHOLD:
        return f"above-average {measure}"
    if index <= LOW_INDEX_THRESHOLD:
        return f"below-average {measure}"
    return f"near-average {measure}"


def _recency_level(index: float) -> str:
    if index >= HIGH_INDEX_THRESHOLD:
        return "less recent purchasing"
    if index <= LOW_INDEX_THRESHOLD:
        return "more recent purchasing"
    return "near-average purchasing recency"


def _validate_segment_summary(segment_summary: pd.DataFrame) -> None:
    missing = [column for column in SUMMARY_COLUMNS if column not in segment_summary.columns]
    if missing:
        raise ValueError(f"Segment summary is missing required column(s): {missing}")
    if segment_summary.empty:
        raise ValueError("Segment summary must not be empty")
    if segment_summary["cluster"].duplicated().any():
        raise ValueError("Segment summary clusters must be unique")
    if (
        segment_summary["segment_label"].isna().any()
        or not segment_summary["segment_label"]
        .map(lambda label: isinstance(label, str) and bool(label.strip()))
        .all()
    ):
        raise ValueError("Segment labels must be non-empty strings")

    numeric_columns = [
        column for column in SUMMARY_COLUMNS if column not in _NON_NUMERIC_SUMMARY_COLUMNS
    ]
    for column in numeric_columns:
        if not is_numeric_dtype(segment_summary[column]):
            raise ValueError(f"Segment summary column {column} must be numeric")
    numeric = segment_summary.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Segment summary numeric fields must be finite")

    clusters = segment_summary["cluster"].to_numpy(dtype=float)
    if not np.equal(clusters, np.floor(clusters)).all():
        raise ValueError("Segment cluster IDs must be integer labels")
    counts = segment_summary["customer_count"].to_numpy(dtype=float)
    if (counts <= 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Segment customer counts must be positive integers")
    shares = segment_summary["customer_share"].to_numpy(dtype=float)
    if (shares <= 0).any() or (shares > 1).any() or not np.isclose(shares.sum(), 1.0):
        raise ValueError("Segment customer shares must be in (0, 1] and sum to 1")
    dominant_shares = segment_summary["dominant_country_share"].to_numpy(dtype=float)
    if (dominant_shares <= 0).any() or (dominant_shares > 1).any():
        raise ValueError("Dominant-country shares must be in (0, 1]")
    index_columns = [f"{feature}_index" for feature in NUMERIC_FEATURE_COLUMNS]
    if segment_summary.loc[:, index_columns].lt(0).any().any():
        raise ValueError("Segment relative indices must be non-negative")


def _validate_insights(insights: pd.DataFrame, expected_segments: int) -> None:
    if len(insights) != expected_segments:
        raise ValueError("Insight output must contain one record per segment")
    if insights["cluster"].duplicated().any():
        raise ValueError("Insight output clusters must be unique")
    if insights["cluster"].tolist() != sorted(insights["cluster"].tolist()):
        raise ValueError("Insight output must use deterministic cluster ordering")
    if not insights["strategy_category"].isin(STRATEGY_CATEGORIES).all():
        raise ValueError("Insight output contains an invalid strategy category")
    if not insights["priority"].isin(PRIORITIES).all():
        raise ValueError("Insight output contains an invalid priority")

    for column in ("primary_objective", "rationale"):
        if (
            insights[column].isna().any()
            or not insights[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"Insight output {column} values must be non-empty")
    if (
        not insights["recommended_actions"]
        .map(
            lambda actions: (
                isinstance(actions, tuple)
                and 2 <= len(actions) <= 4
                and all(isinstance(action, str) and action for action in actions)
            )
        )
        .all()
    ):
        raise ValueError("Insight output must contain two to four recommended actions")
    for column in ("key_risks", "key_opportunities", "recommended_kpis"):
        if (
            not insights[column]
            .map(
                lambda values: (
                    isinstance(values, tuple)
                    and bool(values)
                    and all(isinstance(value, str) and value for value in values)
                )
            )
            .all()
        ):
            raise ValueError(f"Insight output {column} values must be non-empty tuples")
