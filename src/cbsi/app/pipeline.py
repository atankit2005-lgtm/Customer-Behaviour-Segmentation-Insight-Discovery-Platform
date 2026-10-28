from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cbsi.churn import (
    CHURN_FEATURE_COLUMNS,
    CHURN_TARGET_COLUMN,
    PREDICTION_WINDOW_DAYS,
    ChurnDataset,
    ChurnMetrics,
    ChurnModelSelection,
    TemporalChurnSplit,
    build_churn_feature_importance,
    build_segment_churn_summary,
    build_temporal_churn_split,
    evaluate_churn_model,
    evaluate_churn_models,
    fit_churn_model,
    predict_churn_risk,
    select_best_churn_model,
)
from cbsi.churn.analysis import RISK_BANDS
from cbsi.clustering.models import (
    DBSCAN_EPS_VALUES,
    DBSCAN_MIN_SAMPLES_VALUES,
    K_VALUES,
    evaluate_agglomerative,
    evaluate_dbscan,
    evaluate_kmeans,
)
from cbsi.clustering.preprocessing import prepare_clustering_features
from cbsi.clustering.selection import (
    DEFAULT_SELECTION_CRITERIA,
    ClusteringSelection,
    SelectionCriteria,
    fit_selected_cluster_assignments,
    select_best_clustering_result,
)
from cbsi.data.ingestion import load_transactions
from cbsi.features.customer import CUSTOMER_FEATURE_COLUMNS, build_customer_features
from cbsi.preprocessing.transactions import preprocess_transactions
from cbsi.profiling import SegmentProfiles, build_segment_insights, build_segment_profiles


@dataclass(frozen=True)
class AnalysisConfig:
    k_values: tuple[int, ...] = K_VALUES
    dbscan_eps_values: tuple[float, ...] = DBSCAN_EPS_VALUES
    dbscan_min_samples_values: tuple[int, ...] = DBSCAN_MIN_SAMPLES_VALUES
    silhouette_sample_size: int | None = 2_000
    selection_criteria: SelectionCriteria = DEFAULT_SELECTION_CRITERIA
    prediction_window_days: int = PREDICTION_WINDOW_DAYS


DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()


@dataclass(frozen=True)
class AnalysisMetadata:
    raw_transaction_count: int
    eligible_transaction_count: int
    raw_customer_count: int
    customer_count: int
    invoice_count: int
    monetary_total: float
    total_quantity: float
    average_order_value: float
    first_transaction: pd.Timestamp
    last_transaction: pd.Timestamp

    @property
    def excluded_transaction_count(self) -> int:
        return self.raw_transaction_count - self.eligible_transaction_count


@dataclass(frozen=True)
class SegmentationAnalysis:
    selection: ClusteringSelection
    profiles: SegmentProfiles
    insights: pd.DataFrame = field(repr=False, compare=False)


@dataclass(frozen=True)
class AnalysisBundle:
    config: AnalysisConfig
    metadata: AnalysisMetadata
    current: SegmentationAnalysis
    historical: SegmentationAnalysis
    churn_split: TemporalChurnSplit
    churn_selection: ChurnModelSelection
    test_metrics: ChurnMetrics
    predictions: pd.DataFrame = field(repr=False, compare=False)
    scored_customers: pd.DataFrame = field(repr=False, compare=False)
    feature_importance: pd.DataFrame = field(repr=False, compare=False)
    segment_churn: pd.DataFrame = field(repr=False, compare=False)
    risk_distribution: pd.DataFrame = field(repr=False, compare=False)
    temporal_summary: pd.DataFrame = field(repr=False, compare=False)


def run_analysis(
    workbook_path: str | Path,
    *,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG,
) -> AnalysisBundle:
    return analyze_transactions(load_transactions(workbook_path), config=config)


def analyze_transactions(
    transactions: pd.DataFrame,
    *,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG,
) -> AnalysisBundle:
    eligible = preprocess_transactions(transactions)
    split = build_temporal_churn_split(
        eligible, prediction_window_days=config.prediction_window_days
    )
    current_features = build_customer_features(eligible)
    historical_features = build_observation_features(eligible, split.test)
    current = _analyze_segments(current_features, config)
    historical = _analyze_segments(historical_features, config)
    churn_selection = select_best_churn_model(evaluate_churn_models(split.train, split.validation))
    model = fit_churn_model(
        churn_selection.selected.model_name,
        (split.train, split.validation),
        threshold=churn_selection.selected.model.threshold,
    )
    metrics = evaluate_churn_model(model, split.test)
    predictions = predict_churn_risk(model, split.test)
    segment_churn = build_segment_churn_summary(
        predictions, historical.profiles.customers, historical.profiles.summary
    )
    scored = predictions.merge(
        historical.profiles.customers,
        on="customer_id",
        how="left",
        sort=False,
        validate="one_to_one",
    ).merge(
        historical.profiles.summary.loc[:, ["cluster", "segment_label"]],
        on="cluster",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    risk_distribution = (
        (predictions["risk_band"].value_counts().reindex(RISK_BANDS, fill_value=0).rename("count"))
        .rename_axis("risk_band")
        .reset_index()
    )
    risk_distribution["share"] = risk_distribution["count"] / split.test.customer_count
    invoice_count = int(current_features["frequency"].sum())
    monetary_total = float(current_features["monetary"].sum())
    metadata = AnalysisMetadata(
        raw_transaction_count=len(transactions),
        eligible_transaction_count=len(eligible),
        raw_customer_count=int(transactions["customer_id"].nunique()),
        customer_count=len(current_features),
        invoice_count=invoice_count,
        monetary_total=monetary_total,
        total_quantity=float(current_features["total_quantity"].sum()),
        average_order_value=monetary_total / invoice_count,
        first_transaction=eligible["invoice_date"].min(),
        last_transaction=eligible["invoice_date"].max(),
    )
    temporal_summary = pd.DataFrame(
        [
            {
                "snapshot": name,
                "observation_end": dataset.observation_end,
                "prediction_end": dataset.prediction_end,
                "observation_transactions": dataset.observation_transaction_count,
                "prediction_transactions": dataset.prediction_transaction_count,
                "customers": dataset.customer_count,
                "active_customers": int(dataset.data[CHURN_TARGET_COLUMN].eq(0).sum()),
                "inactive_customers": int(dataset.data[CHURN_TARGET_COLUMN].eq(1).sum()),
                "prevalence": dataset.prevalence,
            }
            for name, dataset in (
                ("Train", split.train),
                ("Validation", split.validation),
                ("Test", split.test),
            )
        ]
    )
    return AnalysisBundle(
        config=config,
        metadata=metadata,
        current=current,
        historical=historical,
        churn_split=split,
        churn_selection=churn_selection,
        test_metrics=metrics,
        predictions=predictions,
        scored_customers=scored,
        feature_importance=build_churn_feature_importance(model),
        segment_churn=segment_churn,
        risk_distribution=risk_distribution,
        temporal_summary=temporal_summary,
    )


def build_observation_features(
    transactions: pd.DataFrame,
    snapshot: ChurnDataset,
) -> pd.DataFrame:
    observation = transactions.loc[transactions["invoice_date"].le(snapshot.observation_end)]
    features = build_customer_features(observation)
    if set(features["customer_id"]) != set(snapshot.data["customer_id"]):
        raise ValueError("Historical segmentation requires exact churn snapshot customer coverage")
    # Churn features use the explicit cutoff, not the latest observed purchase.
    aligned = snapshot.data.loc[:, ["customer_id", *CHURN_FEATURE_COLUMNS]].merge(
        features.loc[:, ["customer_id", "country"]],
        on="customer_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return aligned.loc[:, list(CUSTOMER_FEATURE_COLUMNS)].copy(deep=True)


def filter_scored_customers(
    customers: pd.DataFrame,
    *,
    segment_labels: tuple[str, ...] | None = None,
    countries: tuple[str, ...] | None = None,
    risk_bands: tuple[str, ...] | None = None,
    probability_range: tuple[float, float] = (0.0, 1.0),
    customer_id: str = "",
) -> pd.DataFrame:
    minimum, maximum = probability_range
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("Probability range must satisfy 0 <= minimum <= maximum <= 1")
    mask = customers["churn_probability"].between(minimum, maximum)
    for column, values in (
        ("segment_label", segment_labels),
        ("country", countries),
        ("risk_band", risk_bands),
    ):
        if values is not None:
            mask &= customers[column].isin(values)
    if customer_id.strip():
        mask &= customers["customer_id"].astype(str).str.contains(customer_id.strip(), regex=False)
    return customers.loc[mask].copy(deep=True)


def _analyze_segments(features: pd.DataFrame, config: AnalysisConfig) -> SegmentationAnalysis:
    dataset = prepare_clustering_features(features)
    results = [
        *evaluate_kmeans(
            dataset,
            k_values=config.k_values,
            silhouette_sample_size=config.silhouette_sample_size,
        ),
        *evaluate_agglomerative(
            dataset,
            k_values=config.k_values,
            silhouette_sample_size=config.silhouette_sample_size,
        ),
        *evaluate_dbscan(
            dataset,
            eps_values=config.dbscan_eps_values,
            min_samples_values=config.dbscan_min_samples_values,
            silhouette_sample_size=config.silhouette_sample_size,
        ),
    ]
    selection = select_best_clustering_result(results, criteria=config.selection_criteria)
    assignments = fit_selected_cluster_assignments(dataset, selection)
    profiles = build_segment_profiles(features, assignments)
    return SegmentationAnalysis(selection, profiles, build_segment_insights(profiles.summary))
