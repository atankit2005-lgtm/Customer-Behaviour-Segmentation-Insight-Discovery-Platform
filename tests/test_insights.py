from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cbsi.features.customer import NUMERIC_FEATURE_COLUMNS
from cbsi.profiling import INSIGHT_COLUMNS, SUMMARY_COLUMNS, build_segment_insights


def _segment_record(
    cluster: int,
    label: str,
    *,
    customer_count: int = 100,
    customer_share: float = 1.0,
    **indices: float,
) -> dict[str, object]:
    record: dict[str, object] = {
        "cluster": cluster,
        "segment_label": label,
        "description": f"Measured profile for {label}.",
        "characteristics": ("measured characteristic",),
        "customer_count": customer_count,
        "customer_share": customer_share,
        "dominant_country": "United Kingdom",
        "dominant_country_share": 0.75,
        "monetary_total": 1000.0,
        "total_quantity_total": 500.0,
    }
    for feature in NUMERIC_FEATURE_COLUMNS:
        record[f"{feature}_mean"] = 10.0
        record[f"{feature}_median"] = 9.0
        record[f"{feature}_index"] = indices.get(feature, 1.0)
    return record


def _summary(*records: dict[str, object]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS)


def _single_insight(**indices: float) -> pd.Series:
    summary = _summary(_segment_record(7, "Test Segment", **indices))
    return build_segment_insights(summary).iloc[0]


def test_high_value_frequent_recent_segment_uses_loyalty_strategy() -> None:
    insight = _single_insight(
        monetary=1.6,
        frequency=1.4,
        recency_days=0.6,
        unique_products=1.3,
    )

    assert insight["strategy_category"] == "LOYALTY"
    assert insight["priority"] == "High"
    assert insight["primary_objective"] == "Protect existing value"
    assert any(
        "Monetary value is above" in opportunity for opportunity in insight["key_opportunities"]
    )
    assert insight["recommended_kpis"] == (
        "monetary",
        "frequency",
        "recency_days",
        "avg_order_value",
    )


def test_high_value_less_recent_segment_uses_retention_strategy() -> None:
    insight = _single_insight(monetary=1.7, frequency=1.1, recency_days=1.5)

    assert insight["strategy_category"] == "RETENTION"
    assert insight["priority"] == "High"
    assert insight["primary_objective"] == "Protect existing value"
    assert any("Purchasing is less recent" in risk for risk in insight["key_risks"])
    assert "monetary index 1.70" in insight["rationale"].lower()
    assert "recency index 1.50" in insight["rationale"].lower()


def test_low_value_infrequent_recent_segment_develops_frequency() -> None:
    insight = _single_insight(monetary=0.6, frequency=0.7, recency_days=0.6)

    assert insight["strategy_category"] == "VALUE_DEVELOPMENT"
    assert insight["priority"] == "Medium"
    assert insight["primary_objective"] == "Increase purchase frequency"
    assert any(
        "Purchasing is more recent" in opportunity for opportunity in insight["key_opportunities"]
    )
    assert any("Purchase frequency is below" in risk for risk in insight["key_risks"])
    assert "repeat_purchase_rate (future measurement)" in insight["recommended_kpis"]


def test_low_value_infrequent_inactive_segment_uses_low_cost_reactivation() -> None:
    insight = _single_insight(monetary=0.5, frequency=0.6, recency_days=1.6)

    assert insight["strategy_category"] == "REACTIVATION"
    assert insight["priority"] == "Medium"
    assert insight["primary_objective"] == "Reactivate inactive customers"
    assert any("low-cost reactivation" in action for action in insight["recommended_actions"])
    assert any("Purchasing is less recent" in risk for risk in insight["key_risks"])


def test_neutral_segment_uses_low_priority_engagement_strategy() -> None:
    insight = _single_insight()

    assert insight["strategy_category"] == "ENGAGEMENT"
    assert insight["priority"] == "Low"
    assert insight["primary_objective"] == "Maintain customer engagement"
    assert insight["key_opportunities"] == (
        "No strong relative opportunity signal; maintain baseline measurement.",
    )
    assert insight["key_risks"] == (
        "No strong relative risk signal; continue monitoring recency and frequency.",
    )


def test_low_product_breadth_uses_cross_sell_strategy() -> None:
    insight = _single_insight(unique_products=0.7)

    assert insight["strategy_category"] == "CROSS_SELL"
    assert insight["priority"] == "Medium"
    assert insight["primary_objective"] == "Increase product breadth"
    assert "Product breadth has measurable room to develop." in insight["key_opportunities"]
    assert insight["recommended_kpis"] == (
        "unique_products",
        "avg_items_per_invoice",
        "monetary",
    )


def test_actions_are_concrete_non_empty_and_limited() -> None:
    scenarios = (
        {"monetary": 1.6, "frequency": 1.4, "recency_days": 0.6},
        {"monetary": 1.6, "recency_days": 1.4},
        {"frequency": 0.6, "recency_days": 1.4},
        {"monetary": 0.6, "frequency": 0.7, "recency_days": 0.6},
        {"unique_products": 0.7},
        {},
    )

    for scenario in scenarios:
        actions = _single_insight(**scenario)["recommended_actions"]
        assert 2 <= len(actions) <= 4
        assert all(isinstance(action, str) and action.strip() for action in actions)


def test_cluster_ids_do_not_determine_strategy() -> None:
    high = _segment_record(
        10,
        "High Behavior",
        customer_share=0.5,
        monetary=1.5,
        frequency=1.4,
        recency_days=0.6,
    )
    low = _segment_record(
        20,
        "Low Behavior",
        customer_share=0.5,
        monetary=0.5,
        frequency=0.6,
        recency_days=1.5,
    )
    original = build_segment_insights(_summary(high, low))

    high["cluster"] = 20
    low["cluster"] = 10
    swapped = build_segment_insights(_summary(high, low))

    behavior_columns = [column for column in INSIGHT_COLUMNS if column != "cluster"]
    original = original.sort_values("segment_label").loc[:, behavior_columns].reset_index(drop=True)
    swapped = swapped.sort_values("segment_label").loc[:, behavior_columns].reset_index(drop=True)
    assert_frame_equal(original, swapped)


def test_output_is_deterministic_and_sorted_without_mutating_input() -> None:
    summary = _summary(
        _segment_record(20, "Inactive", customer_share=0.5, frequency=0.6, recency_days=1.5),
        _segment_record(10, "Loyal", customer_share=0.5, monetary=1.5, frequency=1.4),
    )
    original = summary.copy(deep=True)

    first = build_segment_insights(summary)
    second = build_segment_insights(summary)

    assert first.columns.tolist() == list(INSIGHT_COLUMNS)
    assert first["cluster"].tolist() == [10, 20]
    assert_frame_equal(first, second)
    assert_frame_equal(summary, original)


def test_missing_required_column_is_rejected() -> None:
    summary = _summary(_segment_record(1, "Segment")).drop(columns=["monetary_index"])

    with pytest.raises(ValueError, match="missing required column.*monetary_index"):
        build_segment_insights(summary)


def test_empty_summary_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_segment_insights(_summary())


def test_non_finite_profile_metric_is_rejected() -> None:
    summary = _summary(_segment_record(1, "Segment"))
    summary.loc[0, "frequency_index"] = np.inf

    with pytest.raises(ValueError, match="numeric fields must be finite"):
        build_segment_insights(summary)


def test_invalid_customer_shares_are_rejected() -> None:
    summary = _summary(
        _segment_record(1, "One", customer_share=0.7),
        _segment_record(2, "Two", customer_share=0.4),
    )

    with pytest.raises(ValueError, match="shares must be in.*sum to 1"):
        build_segment_insights(summary)


def test_duplicate_clusters_are_rejected() -> None:
    summary = _summary(
        _segment_record(1, "One", customer_share=0.5),
        _segment_record(1, "Two", customer_share=0.5),
    )

    with pytest.raises(ValueError, match="clusters must be unique"):
        build_segment_insights(summary)


def test_invalid_cluster_and_customer_count_are_rejected() -> None:
    fractional_cluster = _summary(_segment_record(1, "Segment")).astype({"cluster": float})
    fractional_cluster.loc[0, "cluster"] = 1.5
    with pytest.raises(ValueError, match="cluster IDs must be integer"):
        build_segment_insights(fractional_cluster)

    invalid_count = _summary(_segment_record(1, "Segment", customer_count=0))
    with pytest.raises(ValueError, match="counts must be positive integers"):
        build_segment_insights(invalid_count)
