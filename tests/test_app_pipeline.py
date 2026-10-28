from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cbsi.app.pipeline import (
    AnalysisBundle,
    AnalysisConfig,
    analyze_transactions,
    build_observation_features,
    filter_scored_customers,
    run_analysis,
)
from cbsi.churn import (
    CHURN_FEATURE_COLUMNS,
    CHURN_MODEL_NAMES,
    FEATURE_IMPORTANCE_COLUMNS,
    PREDICTION_COLUMNS,
    SEGMENT_CHURN_COLUMNS,
    build_temporal_churn_split,
)
from cbsi.data.ingestion import SOURCE_TO_CANONICAL
from cbsi.features.customer import CUSTOMER_FEATURE_COLUMNS
from cbsi.profiling import INSIGHT_COLUMNS, PROFILED_CUSTOMER_COLUMNS, SUMMARY_COLUMNS


def test_pipeline_preserves_backend_schemas(analysis_bundle: AnalysisBundle) -> None:
    bundle = analysis_bundle
    for analysis in (bundle.current, bundle.historical):
        assert analysis.profiles.customers.columns.tolist() == list(PROFILED_CUSTOMER_COLUMNS)
        assert analysis.profiles.summary.columns.tolist() == list(SUMMARY_COLUMNS)
        assert analysis.insights.columns.tolist() == list(INSIGHT_COLUMNS)
        assert set(analysis.selection.comparison["model"]) == {"kmeans", "agglomerative", "dbscan"}
        assert len(analysis.selection.comparison) == 5
    assert set(bundle.churn_selection.comparison["model"]) == set(CHURN_MODEL_NAMES)
    assert bundle.predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert bundle.feature_importance.columns.tolist() == list(FEATURE_IMPORTANCE_COLUMNS)
    assert bundle.segment_churn.columns.tolist() == list(SEGMENT_CHURN_COLUMNS)


def test_cohorts_and_exact_historical_join(analysis_bundle: AnalysisBundle) -> None:
    bundle = analysis_bundle
    assert bundle.metadata.customer_count == 41
    assert bundle.churn_split.test.customer_count == 40
    assert "FUTURE" not in set(bundle.scored_customers["customer_id"])
    assert bundle.scored_customers["customer_id"].is_unique
    assert not bundle.scored_customers.isna().any().any()
    assert set(bundle.scored_customers["customer_id"]) == set(bundle.predictions["customer_id"])
    assert bundle.segment_churn["customer_count"].sum() == 40
    assert bundle.risk_distribution["count"].sum() == 40
    assert bundle.risk_distribution["share"].sum() == pytest.approx(1)
    assert bundle.risk_distribution["risk_band"].tolist() == ["High", "Medium", "Low"]


def test_historical_features_equal_cutoff_aligned_churn_features(
    app_transactions: pd.DataFrame, analysis_bundle: AnalysisBundle
) -> None:
    snapshot = analysis_bundle.churn_split.test
    historical = analysis_bundle.historical.profiles.customers
    columns = ["customer_id", *CHURN_FEATURE_COLUMNS]
    assert_frame_equal(historical[columns], snapshot.data[columns])
    assert snapshot.observation_end > app_transactions.loc[
        app_transactions["invoice_date"].le(snapshot.observation_end), "invoice_date"
    ].max() + pd.Timedelta(days=1)
    assert "churned" not in historical.columns


def test_future_perturbations_do_not_change_historical_features(
    app_transactions: pd.DataFrame,
) -> None:
    before = build_temporal_churn_split(app_transactions).test
    changed = app_transactions.copy(deep=True)
    future = changed["invoice_date"].gt(before.observation_end)
    changed.loc[future, "quantity"] *= 10_000
    changed.loc[future, "unit_price"] *= 10_000
    changed.loc[future, "country"] = "Future country"
    after = build_temporal_churn_split(changed).test
    assert_frame_equal(
        build_observation_features(app_transactions, before),
        build_observation_features(changed, after),
    )


def test_test_target_changes_do_not_tune_models_or_historical_segments(
    app_transactions: pd.DataFrame, app_config: AnalysisConfig, analysis_bundle: AnalysisBundle
) -> None:
    cutoff = analysis_bundle.churn_split.test.observation_end
    removed_purchase = app_transactions["customer_id"].eq("U-000") & app_transactions[
        "invoice_date"
    ].gt(cutoff)
    changed = analyze_transactions(app_transactions.loc[~removed_purchase], config=app_config)
    assert_frame_equal(
        changed.churn_selection.comparison, analysis_bundle.churn_selection.comparison
    )
    assert_frame_equal(
        changed.historical.profiles.customers, analysis_bundle.historical.profiles.customers
    )
    assert_frame_equal(
        changed.predictions.drop(columns="actual_churn"),
        analysis_bundle.predictions.drop(columns="actual_churn"),
    )
    assert changed.test_metrics.confusion_matrix != analysis_bundle.test_metrics.confusion_matrix


def test_snapshot_alignment_rejects_missing_customer(
    app_transactions: pd.DataFrame, analysis_bundle: AnalysisBundle
) -> None:
    snapshot = analysis_bundle.churn_split.test
    incomplete = replace(snapshot, data=snapshot.data.iloc[1:].copy(deep=True))
    with pytest.raises(ValueError, match="exact churn snapshot customer coverage"):
        build_observation_features(app_transactions, incomplete)


def test_pipeline_count_value_and_quantity_reconciliation(
    app_transactions: pd.DataFrame, analysis_bundle: AnalysisBundle
) -> None:
    metadata = analysis_bundle.metadata
    summary = analysis_bundle.current.profiles.summary
    assert metadata.raw_transaction_count == len(app_transactions)
    assert metadata.eligible_transaction_count == len(app_transactions)
    assert metadata.excluded_transaction_count == 0
    assert summary["customer_count"].sum() == metadata.customer_count
    assert summary["customer_share"].sum() == pytest.approx(1)
    assert summary["monetary_total"].sum() == pytest.approx(metadata.monetary_total)
    assert metadata.monetary_total == pytest.approx(
        (app_transactions["quantity"] * app_transactions["unit_price"]).sum()
    )
    assert summary["total_quantity_total"].sum() == pytest.approx(metadata.total_quantity)
    assert metadata.total_quantity == pytest.approx(app_transactions["quantity"].sum())
    assert metadata.average_order_value == pytest.approx(
        metadata.monetary_total / metadata.invoice_count
    )
    assert metadata.invoice_count == len(
        app_transactions[["customer_id", "invoice_no"]].drop_duplicates()
    )


def test_pipeline_is_deterministic_and_does_not_mutate_input(
    app_transactions: pd.DataFrame, app_config: AnalysisConfig, analysis_bundle: AnalysisBundle
) -> None:
    original = app_transactions.copy(deep=True)
    rerun = analyze_transactions(app_transactions, config=app_config)
    assert_frame_equal(app_transactions, original)
    assert rerun.metadata == analysis_bundle.metadata
    assert rerun.test_metrics == analysis_bundle.test_metrics
    for first, second in (
        (rerun.current.selection.comparison, analysis_bundle.current.selection.comparison),
        (rerun.historical.profiles.summary, analysis_bundle.historical.profiles.summary),
        (rerun.predictions, analysis_bundle.predictions),
        (rerun.churn_selection.comparison, analysis_bundle.churn_selection.comparison),
    ):
        assert_frame_equal(first, second)


def test_temporal_summary_and_final_metrics_are_finite(analysis_bundle: AnalysisBundle) -> None:
    bundle = analysis_bundle
    assert bundle.temporal_summary["snapshot"].tolist() == ["Train", "Validation", "Test"]
    for row in bundle.temporal_summary.itertuples():
        assert row.active_customers + row.inactive_customers == row.customers
        assert 0 < row.prevalence < 1
    metrics = bundle.test_metrics
    assert np.isfinite((metrics.pr_auc, metrics.roc_auc, metrics.f1)).all()
    assert sum(sum(row) for row in metrics.confusion_matrix) == len(bundle.predictions)
    assert bundle.predictions["churn_probability"].between(0, 1).all()
    assert bundle.churn_selection.selected.model.threshold == 0.5


def test_workbook_entry_point_and_cleaning_funnel(
    app_transactions: pd.DataFrame,
    app_config: AnalysisConfig,
    tmp_path: Path,
) -> None:
    raw = pd.concat([app_transactions, app_transactions.iloc[:1]], ignore_index=True)
    raw.loc[len(raw)] = raw.iloc[0]
    raw.loc[len(raw) - 1, "invoice_no"] = "C-CANCEL"
    source = raw.rename(columns={value: key for key, value in SOURCE_TO_CANONICAL.items()})
    workbook = tmp_path / "retail.xlsx"
    midpoint = len(source) // 2
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        source.iloc[:midpoint].to_excel(writer, sheet_name="Year 2009-2010", index=False)
        source.iloc[midpoint:].to_excel(writer, sheet_name="Year 2010-2011", index=False)
    bundle = run_analysis(workbook, config=app_config)
    assert bundle.metadata.raw_transaction_count == len(raw)
    assert bundle.metadata.excluded_transaction_count == 2
    assert bundle.metadata.customer_count == 41


def test_invalid_or_insufficient_inputs_raise_backend_errors(
    app_transactions: pd.DataFrame, app_config: AnalysisConfig
) -> None:
    with pytest.raises(ValueError, match="missing required column"):
        analyze_transactions(app_transactions.drop(columns="country"), config=app_config)
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_transactions(app_transactions.iloc[:0], config=app_config)
    with pytest.raises(ValueError, match="Observation end must not precede"):
        analyze_transactions(app_transactions.iloc[:2], config=app_config)
    active_only = app_transactions.loc[app_transactions["customer_id"].eq("U-000")]
    with pytest.raises(ValueError, match="both binary classes"):
        analyze_transactions(active_only, config=app_config)
    with pytest.raises(ValueError, match="smaller than the sample count"):
        analyze_transactions(app_transactions, config=replace(app_config, k_values=(100,)))


def test_observation_feature_schema(analysis_bundle: AnalysisBundle) -> None:
    snapshot = analysis_bundle.churn_split.test
    historic = analysis_bundle.historical.profiles.customers.drop(columns="cluster")
    assert historic.columns.tolist() == list(CUSTOMER_FEATURE_COLUMNS)
    assert set(historic["customer_id"]) == set(snapshot.data["customer_id"])


def test_filters_are_combined_inclusive_literal_and_nonmutating(
    analysis_bundle: AnalysisBundle,
) -> None:
    customers = analysis_bundle.scored_customers.copy(deep=True)
    original = customers.copy(deep=True)
    customer = customers.iloc[0]
    probability = float(customer["churn_probability"])
    filtered = filter_scored_customers(
        customers,
        segment_labels=(customer["segment_label"],),
        countries=(customer["country"],),
        risk_bands=(customer["risk_band"],),
        probability_range=(probability, probability),
        customer_id=customer["customer_id"],
    )
    assert filtered["customer_id"].tolist() == [customer["customer_id"]]
    assert filter_scored_customers(customers, risk_bands=()).empty
    assert filter_scored_customers(customers, customer_id=".*").empty
    assert len(filter_scored_customers(customers)) == len(customers)
    filtered.loc[:, "monetary"] = 0
    assert_frame_equal(customers, original)
    with pytest.raises(ValueError, match="Probability range"):
        filter_scored_customers(customers, probability_range=(0.9, 0.1))
