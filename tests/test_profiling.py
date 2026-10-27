from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cbsi.features.customer import CUSTOMER_FEATURE_COLUMNS
from cbsi.profiling import (
    PROFILED_CUSTOMER_COLUMNS,
    SUMMARY_COLUMNS,
    build_profiled_customers,
    build_segment_profiles,
    build_segment_summary,
)


def _customer_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "A",
                "recency_days": 10,
                "frequency": 10,
                "monetary": 1000.0,
                "total_quantity": 100,
                "active_days": 8,
                "avg_order_value": 100.0,
                "avg_items_per_invoice": 10.0,
                "unique_products": 20,
                "purchase_span_days": 100,
                "country": "United Kingdom",
            },
            {
                "customer_id": "B",
                "recency_days": 20,
                "frequency": 8,
                "monetary": 800.0,
                "total_quantity": 80,
                "active_days": 6,
                "avg_order_value": 100.0,
                "avg_items_per_invoice": 10.0,
                "unique_products": 15,
                "purchase_span_days": 80,
                "country": "France",
            },
            {
                "customer_id": "C",
                "recency_days": 100,
                "frequency": 2,
                "monetary": 100.0,
                "total_quantity": 10,
                "active_days": 2,
                "avg_order_value": 50.0,
                "avg_items_per_invoice": 5.0,
                "unique_products": 5,
                "purchase_span_days": 10,
                "country": "Germany",
            },
            {
                "customer_id": "D",
                "recency_days": 120,
                "frequency": 1,
                "monetary": 50.0,
                "total_quantity": 5,
                "active_days": 1,
                "avg_order_value": 50.0,
                "avg_items_per_invoice": 5.0,
                "unique_products": 3,
                "purchase_span_days": 0,
                "country": "Germany",
            },
        ],
        columns=CUSTOMER_FEATURE_COLUMNS,
    )


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D"],
            "cluster": [10, 10, 20, 20],
        }
    )


def test_profiled_customers_join_without_loss_and_preserve_assignment_order() -> None:
    features = _customer_features()
    assignments = _assignments().iloc[[3, 0, 2, 1]].reset_index(drop=True)

    profiled = build_profiled_customers(features, assignments)

    assert profiled.columns.tolist() == list(PROFILED_CUSTOMER_COLUMNS)
    assert profiled["customer_id"].tolist() == assignments["customer_id"].tolist()
    assert profiled["cluster"].tolist() == assignments["cluster"].tolist()
    assert len(profiled) == len(features)
    assert profiled["customer_id"].is_unique
    assert profiled.set_index("customer_id").loc["A", "monetary"] == 1000.0


def test_duplicate_assignment_customer_ids_are_rejected() -> None:
    assignments = _assignments()
    assignments.loc[1, "customer_id"] = "A"

    with pytest.raises(ValueError, match="Assignment customer IDs must be unique"):
        build_profiled_customers(_customer_features(), assignments)


def test_unknown_and_unassigned_customer_ids_are_rejected() -> None:
    unknown = _assignments()
    unknown.loc[3, "customer_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown customer IDs"):
        build_profiled_customers(_customer_features(), unknown)

    incomplete = _assignments().iloc[:-1]
    with pytest.raises(ValueError, match="missing assignments"):
        build_profiled_customers(_customer_features(), incomplete)


def test_missing_customer_ids_are_rejected() -> None:
    features = _customer_features()
    features.loc[0, "customer_id"] = None
    with pytest.raises(ValueError, match="Customer feature IDs must not be missing"):
        build_profiled_customers(features, _assignments())

    assignments = _assignments()
    assignments.loc[0, "customer_id"] = None
    with pytest.raises(ValueError, match="Assignment customer IDs must not be missing"):
        build_profiled_customers(_customer_features(), assignments)


def test_missing_or_non_integer_cluster_assignments_are_rejected() -> None:
    missing = _assignments()
    missing.loc[0, "cluster"] = np.nan
    with pytest.raises(ValueError, match="finite and not missing"):
        build_profiled_customers(_customer_features(), missing)

    fractional = _assignments().astype({"cluster": float})
    fractional.loc[0, "cluster"] = 1.5
    with pytest.raises(ValueError, match="integer labels"):
        build_profiled_customers(_customer_features(), fractional)


def test_segment_summary_contains_counts_shares_means_and_medians() -> None:
    profiles = build_segment_profiles(_customer_features(), _assignments())
    summary = profiles.summary.set_index("cluster")

    assert profiles.summary.columns.tolist() == list(SUMMARY_COLUMNS)
    assert summary.loc[10, "customer_count"] == 2
    assert summary.loc[20, "customer_count"] == 2
    assert summary.loc[10, "customer_share"] == pytest.approx(0.5)
    assert summary.loc[20, "customer_share"] == pytest.approx(0.5)
    assert summary.loc[10, "recency_days_mean"] == pytest.approx(15.0)
    assert summary.loc[10, "recency_days_median"] == pytest.approx(15.0)
    assert summary.loc[10, "frequency_mean"] == pytest.approx(9.0)
    assert summary.loc[10, "frequency_median"] == pytest.approx(9.0)
    assert summary.loc[10, "monetary_mean"] == pytest.approx(900.0)
    assert summary.loc[10, "monetary_median"] == pytest.approx(900.0)
    assert summary.loc[10, "active_days_mean"] == pytest.approx(7.0)
    assert summary.loc[10, "unique_products_mean"] == pytest.approx(17.5)


def test_dominant_country_uses_customer_count_and_alphabetical_tie_break() -> None:
    summary = build_segment_profiles(_customer_features(), _assignments()).summary.set_index(
        "cluster"
    )

    assert summary.loc[10, "dominant_country"] == "France"
    assert summary.loc[10, "dominant_country_share"] == pytest.approx(0.5)
    assert summary.loc[20, "dominant_country"] == "Germany"
    assert summary.loc[20, "dominant_country_share"] == pytest.approx(1.0)


def test_relative_indices_compare_segment_means_with_overall_means() -> None:
    summary = build_segment_profiles(_customer_features(), _assignments()).summary.set_index(
        "cluster"
    )

    assert summary.loc[10, "monetary_index"] == pytest.approx(900.0 / 487.5)
    assert summary.loc[20, "frequency_index"] == pytest.approx(1.5 / 5.25)
    assert summary.loc[10, "recency_days_index"] == pytest.approx(15.0 / 62.5)
    assert summary.loc[20, "active_days_index"] == pytest.approx(1.5 / 4.25)


def test_zero_overall_purchase_span_has_neutral_finite_index() -> None:
    features = _customer_features()
    features["purchase_span_days"] = 0

    summary = build_segment_profiles(features, _assignments()).summary

    assert summary["purchase_span_days_index"].eq(1.0).all()
    numeric = summary.select_dtypes(include="number").to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_segment_order_and_labels_are_deterministic() -> None:
    profiled = build_profiled_customers(_customer_features(), _assignments())

    first = build_segment_summary(profiled)
    second = build_segment_summary(profiled.sample(frac=1.0, random_state=42))

    assert first["cluster"].tolist() == [10, 20]
    assert second["cluster"].tolist() == [10, 20]
    assert first["segment_label"].tolist() == second["segment_label"].tolist()
    assert first["description"].tolist() == second["description"].tolist()


def test_labels_depend_on_behavior_not_cluster_numbers() -> None:
    features = _customer_features()
    original = build_segment_profiles(features, _assignments()).summary
    swapped_assignments = _assignments().replace({"cluster": {10: 20, 20: 10}})
    swapped = build_segment_profiles(features, swapped_assignments).summary

    original_by_value = original.sort_values("monetary_mean")["segment_label"].tolist()
    swapped_by_value = swapped.sort_values("monetary_mean")["segment_label"].tolist()

    assert original_by_value == swapped_by_value
    assert "Higher-Value" in original.loc[original["monetary_mean"].idxmax(), "segment_label"]
    assert "Lower-Value" in original.loc[original["monetary_mean"].idxmin(), "segment_label"]


def test_descriptions_use_measured_characteristics_without_predictive_claims() -> None:
    summary = build_segment_profiles(_customer_features(), _assignments()).summary

    assert summary["characteristics"].map(len).eq(3).all()
    assert summary["description"].str.contains("overall").all()
    prohibited = r"churn|guaranteed|will respond|will increase"
    assert not summary["description"].str.contains(prohibited, case=False, regex=True).any()


def test_required_columns_and_unique_feature_ids_are_validated() -> None:
    features = _customer_features().drop(columns=["country"])
    with pytest.raises(ValueError, match="missing required column.*country"):
        build_profiled_customers(features, _assignments())

    duplicate_ids = _customer_features()
    duplicate_ids.loc[1, "customer_id"] = "A"
    with pytest.raises(ValueError, match="Customer feature IDs must be unique"):
        build_profiled_customers(duplicate_ids, _assignments())


def test_non_finite_customer_features_are_rejected() -> None:
    features = _customer_features()
    features.loc[0, "monetary"] = np.inf

    with pytest.raises(ValueError, match="numeric features must be finite"):
        build_profiled_customers(features, _assignments())


def test_inputs_are_not_mutated() -> None:
    features = _customer_features()
    assignments = _assignments()
    original_features = features.copy(deep=True)
    original_assignments = assignments.copy(deep=True)

    build_segment_profiles(features, assignments)

    assert_frame_equal(features, original_features)
    assert_frame_equal(assignments, original_assignments)


def test_segment_totals_reconcile_with_customer_features() -> None:
    profiles = build_segment_profiles(_customer_features(), _assignments())

    assert profiles.summary["customer_count"].sum() == len(profiles.customers)
    assert profiles.summary["customer_share"].sum() == pytest.approx(1.0)
    assert profiles.summary["monetary_total"].sum() == pytest.approx(
        profiles.customers["monetary"].sum()
    )
    assert profiles.summary["total_quantity_total"].sum() == pytest.approx(
        profiles.customers["total_quantity"].sum()
    )
