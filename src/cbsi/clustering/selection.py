from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import pandas as pd

from cbsi.clustering.models import fit_cluster_assignments
from cbsi.clustering.preprocessing import ClusteringDataset
from cbsi.clustering.validation import ClusteringResult, ParameterValue

COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "parameters",
    "n_clusters",
    "total_customers",
    "assigned_customers",
    "noise_points",
    "noise_fraction",
    "coverage_fraction",
    "min_cluster_size",
    "max_cluster_size",
    "min_cluster_fraction",
    "cluster_size_ratio",
    "balance_score",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "metrics_valid",
    "eligible",
    "rejection_reasons",
    "silhouette_rank",
    "calinski_harabasz_rank",
    "davies_bouldin_rank",
    "metric_rank_sum",
    "selection_rank",
)

_MODEL_PREFERENCE: Final[dict[str, int]] = {
    "kmeans": 0,
    "agglomerative": 1,
    "dbscan": 2,
}


@dataclass(frozen=True)
class SelectionCriteria:
    min_clusters: int = 2
    max_noise_fraction: float = 0.10
    min_cluster_fraction: float = 0.02
    max_cluster_size_ratio: float = 10.0

    def __post_init__(self) -> None:
        if self.min_clusters < 2:
            raise ValueError("min_clusters must be at least 2")
        if not 0 <= self.max_noise_fraction < 1:
            raise ValueError("max_noise_fraction must be in [0, 1)")
        if not 0 < self.min_cluster_fraction <= 1:
            raise ValueError("min_cluster_fraction must be in (0, 1]")
        if self.max_cluster_size_ratio < 1:
            raise ValueError("max_cluster_size_ratio must be at least 1")


DEFAULT_SELECTION_CRITERIA: Final[SelectionCriteria] = SelectionCriteria()


@dataclass(frozen=True)
class ClusteringCandidate:
    model: str
    parameters: dict[str, ParameterValue]
    n_clusters: int
    total_customers: int
    assigned_customers: int
    noise_points: int
    noise_fraction: float
    coverage_fraction: float
    min_cluster_size: int
    max_cluster_size: int
    min_cluster_fraction: float
    cluster_size_ratio: float
    balance_score: float
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    metrics_valid: bool
    eligible: bool
    rejection_reasons: tuple[str, ...]
    silhouette_rank: int | None = None
    calinski_harabasz_rank: int | None = None
    davies_bouldin_rank: int | None = None
    metric_rank_sum: int | None = None
    selection_rank: int | None = None


@dataclass(frozen=True)
class ClusteringSelection:
    criteria: SelectionCriteria
    selected: ClusteringCandidate
    comparison: pd.DataFrame = field(repr=False, compare=False)
    rationale: tuple[str, ...]


def compare_clustering_results(
    results: Iterable[ClusteringResult],
    *,
    criteria: SelectionCriteria = DEFAULT_SELECTION_CRITERIA,
) -> pd.DataFrame:
    """Summarize and deterministically rank clustering candidates."""
    candidates = _rank_candidates(results, criteria)
    return pd.DataFrame(
        [_candidate_record(candidate) for candidate in candidates],
        columns=COMPARISON_COLUMNS,
    )


def select_best_clustering_result(
    results: Iterable[ClusteringResult],
    *,
    criteria: SelectionCriteria = DEFAULT_SELECTION_CRITERIA,
) -> ClusteringSelection:
    """Select the best eligible candidate using equal-rank metric consensus."""
    candidates = _rank_candidates(results, criteria)
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise ValueError("No clustering candidates satisfy the selection criteria")

    selected = min(eligible, key=_selection_key)
    comparison = pd.DataFrame(
        [_candidate_record(candidate) for candidate in candidates],
        columns=COMPARISON_COLUMNS,
    )
    rationale = (
        "Candidate passed cluster-count, coverage, minimum-size, and balance criteria.",
        "All three validation metrics contribute equally through ordinal rank sum.",
        f"Selected rank {selected.selection_rank} with metric rank sum "
        f"{selected.metric_rank_sum}, coverage {selected.coverage_fraction:.2%}, and "
        f"balance {selected.balance_score:.3f}.",
    )
    return ClusteringSelection(
        criteria=criteria,
        selected=selected,
        comparison=comparison,
        rationale=rationale,
    )


def fit_selected_cluster_assignments(
    dataset: ClusteringDataset,
    selection: ClusteringSelection,
) -> pd.DataFrame:
    """Fit the selected configuration and preserve customer ordering."""
    return fit_cluster_assignments(
        dataset,
        selection.selected.model,
        **selection.selected.parameters,
    )


def _rank_candidates(
    results: Iterable[ClusteringResult],
    criteria: SelectionCriteria,
) -> tuple[ClusteringCandidate, ...]:
    result_list = tuple(results)
    if not result_list:
        raise ValueError("At least one clustering result is required")

    candidates = tuple(_summarize_result(result, criteria) for result in result_list)
    eligible_indexes = [index for index, candidate in enumerate(candidates) if candidate.eligible]
    if not eligible_indexes:
        return candidates

    silhouette_ranks = _competition_ranks(
        candidates, eligible_indexes, "silhouette_score", maximize=True
    )
    calinski_ranks = _competition_ranks(
        candidates, eligible_indexes, "calinski_harabasz_score", maximize=True
    )
    davies_ranks = _competition_ranks(
        candidates, eligible_indexes, "davies_bouldin_score", maximize=False
    )

    ranked = list(candidates)
    for index in eligible_indexes:
        metric_rank_sum = silhouette_ranks[index] + calinski_ranks[index] + davies_ranks[index]
        ranked[index] = replace(
            ranked[index],
            silhouette_rank=silhouette_ranks[index],
            calinski_harabasz_rank=calinski_ranks[index],
            davies_bouldin_rank=davies_ranks[index],
            metric_rank_sum=metric_rank_sum,
        )

    selection_order = sorted(eligible_indexes, key=lambda index: _selection_key(ranked[index]))
    for selection_rank, index in enumerate(selection_order, start=1):
        ranked[index] = replace(ranked[index], selection_rank=selection_rank)
    return tuple(ranked)


def _summarize_result(
    result: ClusteringResult,
    criteria: SelectionCriteria,
) -> ClusteringCandidate:
    if not result.cluster_sizes:
        raise ValueError("Clustering results must contain cluster sizes")
    if any(size <= 0 for size in result.cluster_sizes.values()):
        raise ValueError("Cluster sizes must be positive")

    total_customers = sum(result.cluster_sizes.values())
    noise_points = result.cluster_sizes.get(-1, 0)
    non_noise_sizes = [size for label, size in result.cluster_sizes.items() if label != -1]
    assigned_customers = sum(non_noise_sizes)
    n_clusters = len(non_noise_sizes)
    if total_customers != len(result.labels):
        raise ValueError("Cluster sizes must sum to the number of labels")
    if result.n_clusters != n_clusters or result.noise_points != noise_points:
        raise ValueError("Clustering result metadata is inconsistent with cluster sizes")

    noise_fraction = noise_points / total_customers
    if not np.isclose(result.noise_fraction, noise_fraction):
        raise ValueError("Clustering noise fraction is inconsistent with cluster sizes")

    min_cluster_size = min(non_noise_sizes, default=0)
    max_cluster_size = max(non_noise_sizes, default=0)
    min_cluster_fraction = min_cluster_size / assigned_customers if assigned_customers else 0.0
    cluster_size_ratio = max_cluster_size / min_cluster_size if min_cluster_size else float("inf")
    balance_score = min_cluster_size / max_cluster_size if max_cluster_size else 0.0
    coverage_fraction = assigned_customers / total_customers
    metrics_valid = all(
        np.isfinite(value)
        for value in (
            result.silhouette_score,
            result.calinski_harabasz_score,
            result.davies_bouldin_score,
        )
    )

    rejection_reasons = []
    if n_clusters < criteria.min_clusters:
        rejection_reasons.append("insufficient_clusters")
    if not metrics_valid:
        rejection_reasons.append("invalid_metrics")
    if noise_fraction > criteria.max_noise_fraction:
        rejection_reasons.append("excessive_noise")
    if min_cluster_fraction < criteria.min_cluster_fraction:
        rejection_reasons.append("undersized_cluster")
    if cluster_size_ratio > criteria.max_cluster_size_ratio:
        rejection_reasons.append("extreme_imbalance")

    return ClusteringCandidate(
        model=result.model,
        parameters=dict(result.parameters),
        n_clusters=n_clusters,
        total_customers=total_customers,
        assigned_customers=assigned_customers,
        noise_points=noise_points,
        noise_fraction=noise_fraction,
        coverage_fraction=coverage_fraction,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_cluster_fraction=min_cluster_fraction,
        cluster_size_ratio=cluster_size_ratio,
        balance_score=balance_score,
        silhouette_score=result.silhouette_score,
        calinski_harabasz_score=result.calinski_harabasz_score,
        davies_bouldin_score=result.davies_bouldin_score,
        metrics_valid=metrics_valid,
        eligible=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
    )


def _competition_ranks(
    candidates: tuple[ClusteringCandidate, ...],
    eligible_indexes: list[int],
    attribute: str,
    *,
    maximize: bool,
) -> dict[int, int]:
    values = {index: float(getattr(candidates[index], attribute)) for index in eligible_indexes}
    ranks = {}
    for index, value in values.items():
        if maximize:
            ranks[index] = 1 + sum(other > value for other in values.values())
        else:
            ranks[index] = 1 + sum(other < value for other in values.values())
    return ranks


def _selection_key(candidate: ClusteringCandidate) -> tuple[object, ...]:
    return (
        candidate.metric_rank_sum,
        candidate.noise_fraction,
        candidate.cluster_size_ratio,
        candidate.n_clusters,
        _MODEL_PREFERENCE.get(candidate.model, len(_MODEL_PREFERENCE)),
        candidate.model,
        tuple(
            (key, type(value).__name__, repr(value))
            for key, value in sorted(candidate.parameters.items())
        ),
    )


def _candidate_record(candidate: ClusteringCandidate) -> dict[str, object]:
    return {
        "model": candidate.model,
        "parameters": dict(candidate.parameters),
        "n_clusters": candidate.n_clusters,
        "total_customers": candidate.total_customers,
        "assigned_customers": candidate.assigned_customers,
        "noise_points": candidate.noise_points,
        "noise_fraction": candidate.noise_fraction,
        "coverage_fraction": candidate.coverage_fraction,
        "min_cluster_size": candidate.min_cluster_size,
        "max_cluster_size": candidate.max_cluster_size,
        "min_cluster_fraction": candidate.min_cluster_fraction,
        "cluster_size_ratio": candidate.cluster_size_ratio,
        "balance_score": candidate.balance_score,
        "silhouette_score": candidate.silhouette_score,
        "calinski_harabasz_score": candidate.calinski_harabasz_score,
        "davies_bouldin_score": candidate.davies_bouldin_score,
        "metrics_valid": candidate.metrics_valid,
        "eligible": candidate.eligible,
        "rejection_reasons": candidate.rejection_reasons,
        "silhouette_rank": candidate.silhouette_rank,
        "calinski_harabasz_rank": candidate.calinski_harabasz_rank,
        "davies_bouldin_rank": candidate.davies_bouldin_rank,
        "metric_rank_sum": candidate.metric_rank_sum,
        "selection_rank": candidate.selection_rank,
    }
