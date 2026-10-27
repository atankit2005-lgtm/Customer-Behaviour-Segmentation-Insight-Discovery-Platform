from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from cbsi.clustering.preprocessing import ClusteringDataset
from cbsi.clustering.validation import ClusteringResult, ParameterValue, evaluate_labels

K_VALUES: Final[tuple[int, ...]] = tuple(range(2, 11))
DBSCAN_EPS_VALUES: Final[tuple[float, ...]] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
DBSCAN_MIN_SAMPLES_VALUES: Final[tuple[int, ...]] = (5, 10, 15, 20)
KMEANS_RANDOM_STATE: Final[int] = 42
KMEANS_N_INIT: Final[int] = 20


def evaluate_kmeans(
    dataset: ClusteringDataset,
    *,
    k_values: Iterable[int] = K_VALUES,
    random_state: int = KMEANS_RANDOM_STATE,
    n_init: int = KMEANS_N_INIT,
    silhouette_sample_size: int | None = 2_000,
) -> list[ClusteringResult]:
    results = []
    for k in _validate_cluster_counts(k_values, len(dataset.values)):
        parameters: dict[str, ParameterValue] = {
            "n_clusters": k,
            "random_state": random_state,
            "n_init": n_init,
        }
        labels = KMeans(**parameters).fit_predict(dataset.values)
        results.append(
            evaluate_labels(
                dataset.values,
                labels,
                model="kmeans",
                parameters=parameters,
                silhouette_sample_size=silhouette_sample_size,
                random_state=random_state,
            )
        )
    return results


def evaluate_agglomerative(
    dataset: ClusteringDataset,
    *,
    k_values: Iterable[int] = K_VALUES,
    silhouette_sample_size: int | None = 2_000,
) -> list[ClusteringResult]:
    results = []
    for k in _validate_cluster_counts(k_values, len(dataset.values)):
        parameters: dict[str, ParameterValue] = {
            "n_clusters": k,
            "linkage": "ward",
            "metric": "euclidean",
        }
        labels = AgglomerativeClustering(**parameters).fit_predict(dataset.values)
        results.append(
            evaluate_labels(
                dataset.values,
                labels,
                model="agglomerative",
                parameters=parameters,
                silhouette_sample_size=silhouette_sample_size,
            )
        )
    return results


def evaluate_dbscan(
    dataset: ClusteringDataset,
    *,
    eps_values: Iterable[float] = DBSCAN_EPS_VALUES,
    min_samples_values: Iterable[int] = DBSCAN_MIN_SAMPLES_VALUES,
    silhouette_sample_size: int | None = 2_000,
) -> list[ClusteringResult]:
    eps_grid = tuple(eps_values)
    min_samples_grid = tuple(min_samples_values)
    if not eps_grid or not min_samples_grid:
        raise ValueError("DBSCAN parameter grids must not be empty")

    results = []
    for eps in eps_grid:
        if eps <= 0:
            raise ValueError("DBSCAN eps values must be positive")
        for min_samples in min_samples_grid:
            if min_samples < 2:
                raise ValueError("DBSCAN min_samples values must be at least 2")
            parameters: dict[str, ParameterValue] = {
                "eps": eps,
                "min_samples": min_samples,
                "metric": "euclidean",
            }
            labels = DBSCAN(**parameters).fit_predict(dataset.values)
            results.append(
                evaluate_labels(
                    dataset.values,
                    labels,
                    model="dbscan",
                    parameters=parameters,
                    exclude_noise=True,
                    silhouette_sample_size=silhouette_sample_size,
                )
            )
    return results


def fit_cluster_assignments(
    dataset: ClusteringDataset,
    model: str,
    **parameters: ParameterValue,
) -> pd.DataFrame:
    """Fit one configured clustering model and preserve customer assignment order."""
    if model == "kmeans":
        estimator = KMeans(
            n_clusters=int(parameters["n_clusters"]),
            random_state=int(parameters.get("random_state", KMEANS_RANDOM_STATE)),
            n_init=int(parameters.get("n_init", KMEANS_N_INIT)),
        )
    elif model == "agglomerative":
        estimator = AgglomerativeClustering(
            n_clusters=int(parameters["n_clusters"]),
            linkage="ward",
            metric="euclidean",
        )
    elif model == "dbscan":
        estimator = DBSCAN(
            eps=float(parameters["eps"]),
            min_samples=int(parameters["min_samples"]),
            metric="euclidean",
        )
    else:
        raise ValueError(f"Unsupported clustering model: {model}")

    labels = estimator.fit_predict(dataset.values)
    return pd.DataFrame(
        {
            "customer_id": dataset.customer_ids.to_numpy(copy=True),
            "cluster": labels,
        }
    )


def _validate_cluster_counts(k_values: Iterable[int], sample_count: int) -> tuple[int, ...]:
    counts = tuple(k_values)
    if not counts:
        raise ValueError("At least one cluster count is required")
    if any(k < 2 or k >= sample_count for k in counts):
        raise ValueError("Cluster counts must be at least 2 and smaller than the sample count")
    return counts
