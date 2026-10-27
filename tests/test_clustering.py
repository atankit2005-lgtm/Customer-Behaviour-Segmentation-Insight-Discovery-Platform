from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from cbsi.clustering.models import (
    DBSCAN_EPS_VALUES,
    DBSCAN_MIN_SAMPLES_VALUES,
    K_VALUES,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    evaluate_agglomerative,
    evaluate_dbscan,
    evaluate_kmeans,
    fit_cluster_assignments,
)
from cbsi.clustering.preprocessing import (
    CLUSTERING_FEATURES,
    ClusteringDataset,
    prepare_clustering_features,
)
from cbsi.clustering.validation import METRIC_DIRECTIONS, ClusteringResult, evaluate_labels
from cbsi.features.customer import CUSTOMER_FEATURE_COLUMNS


def _customer_features() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in range(3):
        for offset in range(8):
            rows.append(
                {
                    "customer_id": f"customer-{group}-{offset}",
                    "recency_days": (5, 80, 300)[group] + offset,
                    "frequency": (50, 12, 2)[group] + offset,
                    "monetary": (5000, 900, 100)[group] + 20 * offset,
                    "total_quantity": (1000, 240, 30)[group] + 5 * offset,
                    "active_days": (40, 10, 2)[group] + offset,
                    "avg_order_value": (100, 75, 50)[group] + offset,
                    "avg_items_per_invoice": (20, 15, 10)[group] + offset,
                    "unique_products": (100, 30, 5)[group] + offset,
                    "purchase_span_days": (300, 100, 0)[group] + offset,
                    "country": ("United Kingdom", "France", "Germany")[group],
                }
            )
    return pd.DataFrame(rows, columns=CUSTOMER_FEATURE_COLUMNS)


@pytest.fixture
def customer_features() -> pd.DataFrame:
    return _customer_features()


@pytest.fixture
def dataset(customer_features: pd.DataFrame) -> ClusteringDataset:
    return prepare_clustering_features(customer_features)


def _dbscan_dataset() -> ClusteringDataset:
    values = np.array(
        [
            [0.00, 0.00],
            [0.04, 0.02],
            [-0.04, 0.01],
            [2.00, 2.00],
            [2.04, 2.02],
            [1.96, 2.01],
            [6.00, 6.00],
        ]
    )
    return ClusteringDataset(
        customer_ids=pd.Series([f"customer-{index}" for index in range(len(values))]),
        feature_names=("x", "y"),
        values=values,
    )


def test_preparation_validates_required_columns(customer_features: pd.DataFrame) -> None:
    incomplete = customer_features.drop(columns=["monetary", "country"])

    with pytest.raises(ValueError, match=r"monetary.*country"):
        prepare_clustering_features(incomplete)


def test_preparation_preserves_customer_id_order(customer_features: pd.DataFrame) -> None:
    row_order = [5, 1, 20, *range(3, 5), *range(6, 20), 0, 2, *range(21, 24)]
    reordered = customer_features.iloc[row_order]

    result = prepare_clustering_features(reordered)

    assert result.customer_ids.tolist() == reordered["customer_id"].tolist()


def test_preparation_uses_exact_ordered_numeric_features(dataset: ClusteringDataset) -> None:
    assert dataset.feature_names == CLUSTERING_FEATURES
    assert len(dataset.feature_names) == 9
    assert "customer_id" not in dataset.feature_names
    assert "country" not in dataset.feature_names


def test_preparation_applies_log1p_then_standard_scaler(
    customer_features: pd.DataFrame,
) -> None:
    raw = customer_features.loc[:, list(CLUSTERING_FEATURES)].to_numpy(dtype=float)
    expected = StandardScaler().fit_transform(np.log1p(raw))

    result = prepare_clustering_features(customer_features)

    np.testing.assert_allclose(result.values, expected)


def test_prepared_features_have_zero_mean_and_unit_variance(dataset: ClusteringDataset) -> None:
    np.testing.assert_allclose(dataset.values.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(dataset.values.std(axis=0), 1.0, atol=1e-12)


def test_prepared_matrix_shape_and_values_are_finite(dataset: ClusteringDataset) -> None:
    assert dataset.values.shape == (24, 9)
    assert np.isfinite(dataset.values).all()


def test_preparation_does_not_mutate_input(customer_features: pd.DataFrame) -> None:
    original = customer_features.copy(deep=True)

    prepare_clustering_features(customer_features)

    assert_frame_equal(customer_features, original)


@pytest.mark.parametrize(("column", "value"), [("monetary", np.nan), ("frequency", np.inf)])
def test_preparation_rejects_non_finite_values(
    customer_features: pd.DataFrame,
    column: str,
    value: float,
) -> None:
    customer_features[column] = customer_features[column].astype(float)
    customer_features.loc[0, column] = value

    with pytest.raises(ValueError, match="finite numeric"):
        prepare_clustering_features(customer_features)


@pytest.mark.parametrize(
    ("column", "value"), [("frequency", 0), ("monetary", -1), ("purchase_span_days", -1)]
)
def test_preparation_rejects_invalid_non_positive_values(
    customer_features: pd.DataFrame,
    column: str,
    value: int,
) -> None:
    customer_features.loc[0, column] = value

    with pytest.raises(ValueError, match="positive|non-negative"):
        prepare_clustering_features(customer_features)


def test_preparation_accepts_zero_purchase_span(customer_features: pd.DataFrame) -> None:
    assert customer_features["purchase_span_days"].eq(0).any()

    result = prepare_clustering_features(customer_features)

    assert result.values.shape == (24, 9)


def test_preparation_rejects_duplicate_customer_ids(customer_features: pd.DataFrame) -> None:
    customer_features.loc[1, "customer_id"] = customer_features.loc[0, "customer_id"]

    with pytest.raises(ValueError, match="unique"):
        prepare_clustering_features(customer_features)


def test_kmeans_records_requested_k_values(dataset: ClusteringDataset) -> None:
    results = evaluate_kmeans(dataset, k_values=(2, 3), silhouette_sample_size=None)

    assert [result.parameters["n_clusters"] for result in results] == [2, 3]
    assert [result.n_clusters for result in results] == [2, 3]


def test_kmeans_uses_explicit_reproducible_configuration(dataset: ClusteringDataset) -> None:
    result = evaluate_kmeans(dataset, k_values=(3,), silhouette_sample_size=None)[0]

    assert result.parameters["random_state"] == KMEANS_RANDOM_STATE == 42
    assert result.parameters["n_init"] == KMEANS_N_INIT == 20
    assert len(result.labels) == len(dataset.customer_ids)


def test_kmeans_calculates_valid_metrics(dataset: ClusteringDataset) -> None:
    result = evaluate_kmeans(dataset, k_values=(3,), silhouette_sample_size=None)[0]

    assert np.isfinite(result.silhouette_score)
    assert np.isfinite(result.calinski_harabasz_score)
    assert np.isfinite(result.davies_bouldin_score)
    assert sum(result.cluster_sizes.values()) == len(dataset.customer_ids)


def test_agglomerative_records_ward_euclidean_configuration(
    dataset: ClusteringDataset,
) -> None:
    results = evaluate_agglomerative(dataset, k_values=(2, 3), silhouette_sample_size=None)

    assert [result.parameters["n_clusters"] for result in results] == [2, 3]
    assert all(result.parameters["linkage"] == "ward" for result in results)
    assert all(result.parameters["metric"] == "euclidean" for result in results)


def test_agglomerative_calculates_valid_metrics(dataset: ClusteringDataset) -> None:
    result = evaluate_agglomerative(dataset, k_values=(3,), silhouette_sample_size=None)[0]

    assert result.n_clusters == 3
    assert np.isfinite(result.silhouette_score)
    assert np.isfinite(result.calinski_harabasz_score)
    assert np.isfinite(result.davies_bouldin_score)


def test_default_search_grids_are_explicit() -> None:
    assert K_VALUES == tuple(range(2, 11))
    assert DBSCAN_EPS_VALUES == (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
    assert DBSCAN_MIN_SAMPLES_VALUES == (5, 10, 15, 20)


def test_dbscan_preserves_noise_and_excludes_it_from_cluster_count() -> None:
    result = evaluate_dbscan(
        _dbscan_dataset(),
        eps_values=(0.10,),
        min_samples_values=(2,),
        silhouette_sample_size=None,
    )[0]

    assert -1 in result.labels
    assert result.n_clusters == 2
    assert result.noise_points == 1
    assert result.noise_fraction == pytest.approx(1 / 7)
    assert result.cluster_sizes[-1] == 1


def test_dbscan_metrics_use_only_non_noise_observations() -> None:
    dataset = _dbscan_dataset()
    result = evaluate_dbscan(
        dataset,
        eps_values=(0.10,),
        min_samples_values=(2,),
        silhouette_sample_size=None,
    )[0]
    non_noise = result.labels != -1
    expected = silhouette_score(dataset.values[non_noise], result.labels[non_noise])

    assert result.metrics_include_noise is False
    assert result.metric_points == 6
    assert result.silhouette_points == 6
    assert result.silhouette_score == pytest.approx(expected)


def test_invalid_dbscan_metrics_are_nan_instead_of_crashing() -> None:
    result = evaluate_dbscan(
        _dbscan_dataset(),
        eps_values=(0.001,),
        min_samples_values=(2,),
        silhouette_sample_size=None,
    )[0]

    assert result.n_clusters == 0
    assert result.noise_points == 7
    assert np.isnan(result.silhouette_score)
    assert np.isnan(result.calinski_harabasz_score)
    assert np.isnan(result.davies_bouldin_score)


def test_dbscan_records_each_explicit_parameter_combination() -> None:
    results = evaluate_dbscan(
        _dbscan_dataset(),
        eps_values=(0.10, 0.20),
        min_samples_values=(2, 3),
        silhouette_sample_size=None,
    )

    combinations = {
        (result.parameters["eps"], result.parameters["min_samples"]) for result in results
    }
    assert combinations == {(0.10, 2), (0.10, 3), (0.20, 2), (0.20, 3)}
    assert all(result.parameters["metric"] == "euclidean" for result in results)


def test_generic_validation_returns_nan_for_single_cluster() -> None:
    values = np.array([[0.0], [1.0], [2.0]])
    labels = np.zeros(3, dtype=int)

    result = evaluate_labels(values, labels, model="test", parameters={})

    assert np.isnan(result.silhouette_score)
    assert np.isnan(result.calinski_harabasz_score)
    assert np.isnan(result.davies_bouldin_score)


def test_metric_directions_are_explicit() -> None:
    assert METRIC_DIRECTIONS == {
        "silhouette_score": "maximize",
        "calinski_harabasz_score": "maximize",
        "davies_bouldin_score": "minimize",
    }


def test_result_record_contains_model_parameters_metrics_and_labels(
    dataset: ClusteringDataset,
) -> None:
    result = evaluate_kmeans(dataset, k_values=(3,), silhouette_sample_size=None)[0]

    assert isinstance(result, ClusteringResult)
    assert result.model == "kmeans"
    assert result.parameters["n_clusters"] == 3
    assert len(result.labels) == len(dataset.customer_ids)


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        ("kmeans", {"n_clusters": 3}),
        ("agglomerative", {"n_clusters": 3}),
        ("dbscan", {"eps": 0.5, "min_samples": 2}),
    ],
)
def test_final_assignments_preserve_customer_order(
    dataset: ClusteringDataset,
    model: str,
    parameters: dict[str, int | float],
) -> None:
    assignments = fit_cluster_assignments(dataset, model, **parameters)

    assert assignments.columns.tolist() == ["customer_id", "cluster"]
    assert assignments["customer_id"].tolist() == dataset.customer_ids.tolist()
    assert len(assignments) == len(dataset.customer_ids)


def test_dbscan_final_assignments_retain_noise_label() -> None:
    dataset = _dbscan_dataset()

    assignments = fit_cluster_assignments(dataset, "dbscan", eps=0.10, min_samples=2)

    assert -1 in assignments["cluster"].to_numpy()
