from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cbsi.clustering.models import evaluate_kmeans
from cbsi.clustering.preprocessing import ClusteringDataset
from cbsi.clustering.selection import (
    COMPARISON_COLUMNS,
    SelectionCriteria,
    compare_clustering_results,
    fit_selected_cluster_assignments,
    select_best_clustering_result,
)
from cbsi.clustering.validation import ClusteringResult


def _result(
    model: str,
    cluster_sizes: dict[int, int],
    *,
    silhouette: float = 0.4,
    calinski_harabasz: float = 400.0,
    davies_bouldin: float = 0.8,
    parameters: dict[str, int | float | str] | None = None,
) -> ClusteringResult:
    labels = np.concatenate(
        [np.full(size, label, dtype=int) for label, size in cluster_sizes.items()]
    )
    noise_points = cluster_sizes.get(-1, 0)
    n_clusters = sum(label != -1 for label in cluster_sizes)
    metric_points = len(labels) - noise_points if model == "dbscan" else len(labels)
    return ClusteringResult(
        model=model,
        parameters=parameters or {"n_clusters": n_clusters},
        n_clusters=n_clusters,
        noise_points=noise_points,
        noise_fraction=noise_points / len(labels),
        silhouette_score=silhouette,
        calinski_harabasz_score=calinski_harabasz,
        davies_bouldin_score=davies_bouldin,
        metrics_include_noise=model != "dbscan",
        metric_points=metric_points,
        silhouette_points=metric_points,
        cluster_sizes=dict(cluster_sizes),
        labels=labels,
    )


def test_comparison_uses_canonical_schema_and_aggregates_metrics() -> None:
    result = _result("kmeans", {0: 60, 1: 40})

    comparison = compare_clustering_results([result])
    candidate = comparison.iloc[0]

    assert comparison.columns.tolist() == list(COMPARISON_COLUMNS)
    assert candidate["n_clusters"] == 2
    assert candidate["total_customers"] == 100
    assert candidate["assigned_customers"] == 100
    assert candidate["silhouette_score"] == 0.4
    assert candidate["calinski_harabasz_score"] == 400.0
    assert candidate["davies_bouldin_score"] == 0.8


def test_cluster_sizes_and_balance_are_extracted_from_non_noise_clusters() -> None:
    result = _result(
        "dbscan",
        {-1: 10, 0: 50, 1: 40},
        parameters={"eps": 0.8, "min_samples": 10, "metric": "euclidean"},
    )

    candidate = compare_clustering_results([result]).iloc[0]

    assert candidate["noise_points"] == 10
    assert candidate["noise_fraction"] == pytest.approx(0.10)
    assert candidate["coverage_fraction"] == pytest.approx(0.90)
    assert candidate["min_cluster_size"] == 40
    assert candidate["max_cluster_size"] == 50
    assert candidate["cluster_size_ratio"] == pytest.approx(1.25)
    assert candidate["balance_score"] == pytest.approx(0.8)
    assert bool(candidate["eligible"])


def test_single_cluster_with_invalid_metrics_is_rejected() -> None:
    result = _result(
        "dbscan",
        {-1: 20, 0: 80},
        silhouette=np.nan,
        calinski_harabasz=np.nan,
        davies_bouldin=np.nan,
        parameters={"eps": 1.0, "min_samples": 20, "metric": "euclidean"},
    )

    candidate = compare_clustering_results([result]).iloc[0]

    assert not bool(candidate["metrics_valid"])
    assert not bool(candidate["eligible"])
    assert "insufficient_clusters" in candidate["rejection_reasons"]
    assert "invalid_metrics" in candidate["rejection_reasons"]


def test_extreme_cluster_imbalance_is_rejected() -> None:
    result = _result("kmeans", {0: 990, 1: 10}, silhouette=0.9)

    candidate = compare_clustering_results([result]).iloc[0]

    assert candidate["min_cluster_fraction"] == pytest.approx(0.01)
    assert candidate["cluster_size_ratio"] == pytest.approx(99.0)
    assert not bool(candidate["eligible"])
    assert "undersized_cluster" in candidate["rejection_reasons"]
    assert "extreme_imbalance" in candidate["rejection_reasons"]


def test_tiny_dbscan_cluster_does_not_win_on_silhouette_alone() -> None:
    kmeans = _result(
        "kmeans",
        {0: 50, 1: 50},
        silhouette=0.35,
        calinski_harabasz=500.0,
        davies_bouldin=1.0,
        parameters={"n_clusters": 2, "random_state": 42, "n_init": 20},
    )
    dbscan = _result(
        "dbscan",
        {-1: 5, 0: 94, 1: 1},
        silhouette=0.80,
        calinski_harabasz=900.0,
        davies_bouldin=0.2,
        parameters={"eps": 1.0, "min_samples": 10, "metric": "euclidean"},
    )

    selection = select_best_clustering_result([dbscan, kmeans])

    assert selection.selected.model == "kmeans"
    dbscan_row = selection.comparison.loc[selection.comparison["model"] == "dbscan"].iloc[0]
    assert not bool(dbscan_row["eligible"])
    assert "undersized_cluster" in dbscan_row["rejection_reasons"]


def test_metric_rank_sum_considers_all_three_validation_metrics() -> None:
    silhouette_leader = _result(
        "kmeans",
        {0: 50, 1: 50},
        silhouette=0.8,
        calinski_harabasz=100.0,
        davies_bouldin=2.0,
        parameters={"n_clusters": 2},
    )
    consensus_leader = _result(
        "agglomerative",
        {0: 50, 1: 50},
        silhouette=0.6,
        calinski_harabasz=300.0,
        davies_bouldin=0.5,
        parameters={"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
    )
    third = _result(
        "dbscan",
        {0: 50, 1: 50},
        silhouette=0.4,
        calinski_harabasz=200.0,
        davies_bouldin=1.0,
        parameters={"eps": 0.8, "min_samples": 10, "metric": "euclidean"},
    )

    selection = select_best_clustering_result([silhouette_leader, consensus_leader, third])

    assert selection.selected.model == "agglomerative"
    assert selection.selected.metric_rank_sum == 4


def test_selection_is_deterministic_across_input_order() -> None:
    results = [
        _result("kmeans", {0: 55, 1: 45}, parameters={"n_clusters": 2}),
        _result(
            "agglomerative",
            {0: 60, 1: 40},
            silhouette=0.3,
            calinski_harabasz=350.0,
            davies_bouldin=0.9,
            parameters={"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
        ),
    ]

    forward = select_best_clustering_result(results)
    reverse = select_best_clustering_result(reversed(results))

    assert forward.selected.model == reverse.selected.model
    assert forward.selected.parameters == reverse.selected.parameters
    assert forward.selected.metric_rank_sum == reverse.selected.metric_rank_sum


def test_identical_quality_uses_explicit_stable_model_tie_break() -> None:
    kmeans = _result("kmeans", {0: 50, 1: 50}, parameters={"n_clusters": 2})
    agglomerative = _result(
        "agglomerative",
        {0: 50, 1: 50},
        parameters={"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
    )

    selection = select_best_clustering_result([agglomerative, kmeans])

    assert selection.selected.model == "kmeans"
    assert selection.selected.selection_rank == 1


def test_selected_result_contains_reproducible_metadata_and_rationale() -> None:
    result = _result(
        "kmeans",
        {0: 52, 1: 48},
        parameters={"n_clusters": 2, "random_state": 42, "n_init": 20},
    )

    selection = select_best_clustering_result([result])

    assert selection.selected.parameters == {
        "n_clusters": 2,
        "random_state": 42,
        "n_init": 20,
    }
    assert selection.selected.selection_rank == 1
    assert selection.rationale


def test_selection_does_not_mutate_results_or_comparison_output() -> None:
    result = _result("kmeans", {0: 60, 1: 40})
    original_parameters = copy.deepcopy(result.parameters)
    original_sizes = copy.deepcopy(result.cluster_sizes)
    original_labels = result.labels.copy()
    first = compare_clustering_results([result])

    selection = select_best_clustering_result([result])
    second = compare_clustering_results([result])

    assert result.parameters == original_parameters
    assert result.cluster_sizes == original_sizes
    np.testing.assert_array_equal(result.labels, original_labels)
    assert_frame_equal(first, second)
    assert selection.comparison is not first


def test_selected_assignments_preserve_customer_alignment() -> None:
    dataset = ClusteringDataset(
        customer_ids=pd.Series([f"customer-{index}" for index in range(6)]),
        feature_names=("x", "y"),
        values=np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [3.0, 3.0],
                [3.1, 3.0],
                [3.0, 3.1],
            ]
        ),
    )
    results = evaluate_kmeans(dataset, k_values=(2,), silhouette_sample_size=None)
    selection = select_best_clustering_result(results)

    assignments = fit_selected_cluster_assignments(dataset, selection)

    assert assignments.columns.tolist() == ["customer_id", "cluster"]
    assert assignments["customer_id"].tolist() == dataset.customer_ids.tolist()
    assert len(assignments) == len(dataset.customer_ids)
    assert assignments["cluster"].nunique() == 2


def test_noise_threshold_is_configurable() -> None:
    result = _result(
        "dbscan",
        {-1: 15, 0: 45, 1: 40},
        parameters={"eps": 0.8, "min_samples": 15, "metric": "euclidean"},
    )

    default_candidate = compare_clustering_results([result]).iloc[0]
    permissive_candidate = compare_clustering_results(
        [result], criteria=SelectionCriteria(max_noise_fraction=0.20)
    ).iloc[0]

    assert not bool(default_candidate["eligible"])
    assert "excessive_noise" in default_candidate["rejection_reasons"]
    assert bool(permissive_candidate["eligible"])


def test_selection_rejects_empty_or_entirely_ineligible_candidates() -> None:
    with pytest.raises(ValueError, match="At least one"):
        select_best_clustering_result([])

    invalid = _result(
        "dbscan",
        {0: 100},
        silhouette=np.nan,
        calinski_harabasz=np.nan,
        davies_bouldin=np.nan,
    )
    with pytest.raises(ValueError, match="No clustering candidates"):
        select_best_clustering_result([invalid])
