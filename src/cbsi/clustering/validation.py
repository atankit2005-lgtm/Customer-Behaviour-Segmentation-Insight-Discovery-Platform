from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ParameterValue = int | float | str

METRIC_DIRECTIONS: Final[dict[str, str]] = {
    "silhouette_score": "maximize",
    "calinski_harabasz_score": "maximize",
    "davies_bouldin_score": "minimize",
}


@dataclass(frozen=True)
class ClusteringResult:
    model: str
    parameters: dict[str, ParameterValue]
    n_clusters: int
    noise_points: int
    noise_fraction: float
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    metrics_include_noise: bool
    metric_points: int
    silhouette_points: int
    cluster_sizes: dict[int, int]
    labels: np.ndarray = field(repr=False, compare=False)


def evaluate_labels(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    model: str,
    parameters: dict[str, ParameterValue],
    exclude_noise: bool = False,
    silhouette_sample_size: int | None = 2_000,
    random_state: int = 42,
) -> ClusteringResult:
    """Calculate clustering metrics, excluding DBSCAN noise when requested."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("Values and labels must contain the same number of observations")
    if not np.isfinite(values).all():
        raise ValueError("Clustering values must be finite")

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {
        int(label): int(count) for label, count in zip(unique_labels, counts, strict=True)
    }
    noise_points = cluster_sizes.get(-1, 0)
    non_noise_labels = unique_labels[unique_labels != -1]
    n_clusters = len(non_noise_labels)

    metric_mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
    metric_values = values[metric_mask]
    metric_labels = labels[metric_mask]
    metric_points = len(metric_labels)
    valid = 2 <= len(np.unique(metric_labels)) < metric_points

    silhouette = np.nan
    calinski_harabasz = np.nan
    davies_bouldin = np.nan
    silhouette_points = 0
    if valid:
        silhouette_points = metric_points
        sample_size = None
        if silhouette_sample_size is not None and silhouette_sample_size < metric_points:
            sample_size = silhouette_sample_size
            silhouette_points = silhouette_sample_size
        silhouette = _safe_metric(
            silhouette_score,
            metric_values,
            metric_labels,
            sample_size=sample_size,
            random_state=random_state,
        )
        calinski_harabasz = _safe_metric(calinski_harabasz_score, metric_values, metric_labels)
        davies_bouldin = _safe_metric(davies_bouldin_score, metric_values, metric_labels)

    return ClusteringResult(
        model=model,
        parameters=dict(parameters),
        n_clusters=n_clusters,
        noise_points=noise_points,
        noise_fraction=noise_points / len(labels),
        silhouette_score=float(silhouette),
        calinski_harabasz_score=float(calinski_harabasz),
        davies_bouldin_score=float(davies_bouldin),
        metrics_include_noise=not exclude_noise,
        metric_points=metric_points,
        silhouette_points=silhouette_points,
        cluster_sizes=cluster_sizes,
        labels=labels.copy(),
    )


def _safe_metric(
    metric: Callable[..., float],
    values: np.ndarray,
    labels: np.ndarray,
    **kwargs: object,
) -> float:
    try:
        return float(metric(values, labels, **kwargs))
    except ValueError:
        return float("nan")
