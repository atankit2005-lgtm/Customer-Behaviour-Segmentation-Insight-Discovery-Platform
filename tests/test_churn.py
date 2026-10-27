from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import cbsi.churn.analysis as churn_analysis
from cbsi.churn import (
    CHURN_DATA_COLUMNS,
    CHURN_FEATURE_COLUMNS,
    CHURN_MODEL_NAMES,
    FEATURE_IMPORTANCE_COLUMNS,
    PREDICTION_COLUMNS,
    SEGMENT_CHURN_COLUMNS,
    ChurnDataset,
    ChurnModelResult,
    build_churn_dataset,
    build_churn_feature_importance,
    build_segment_churn_summary,
    build_temporal_churn_split,
    evaluate_churn_model,
    evaluate_churn_models,
    fit_churn_model,
    predict_churn_risk,
    select_best_churn_model,
)
from cbsi.data.schema import TRANSACTION_COLUMNS


def _transaction(
    invoice_no: str,
    customer_id: str,
    invoice_date: str,
    *,
    quantity: int = 1,
    unit_price: float = 10.0,
    stock_code: str | None = None,
) -> dict[str, object]:
    return {
        "invoice_no": invoice_no,
        "stock_code": stock_code or f"SKU-{invoice_no}",
        "description": "Product",
        "quantity": quantity,
        "invoice_date": pd.Timestamp(invoice_date),
        "unit_price": unit_price,
        "customer_id": customer_id,
        "country": "United Kingdom",
    }


def _boundary_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _transaction("A-1", "A", "2021-01-01", quantity=2),
            _transaction("A-2", "A", "2021-03-31", quantity=3),
            _transaction("B-1", "B", "2021-02-01", unit_price=5.0),
            _transaction("I-C1", "C", "2021-04-01"),
            _transaction("A-3", "A", "2021-06-29", quantity=1_000, unit_price=999.0),
        ],
        columns=TRANSACTION_COLUMNS,
    )


def _split_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _transaction("A-1", "A", "2021-01-01"),
            _transaction("B-1", "B", "2021-01-02"),
            _transaction("A-2", "A", "2021-04-20"),
            _transaction("I-C1", "C", "2021-05-01"),
            _transaction("A-3", "A", "2021-07-20"),
            _transaction("D-1", "D", "2021-08-01"),
            _transaction("A-4", "A", "2021-10-20"),
            _transaction("E-1", "E", "2021-12-31"),
        ],
        columns=TRANSACTION_COLUMNS,
    )


def _model_dataset(
    name: str,
    observation_end: str,
    prediction_end: str,
    *,
    offset: float = 0.0,
    rows: int = 60,
) -> ChurnDataset:
    records = []
    for index in range(rows):
        churned = int(index % 3 != 0)
        records.append(
            {
                "customer_id": f"{name}-{index:03d}",
                "recency_days": offset + (80 + index % 10 if churned else 5 + index % 3),
                "frequency": offset + (1 + index % 2 if churned else 8 + index % 4),
                "monetary": offset + (100 + index if churned else 1_000 + 10 * index),
                "total_quantity": offset + (10 + index if churned else 100 + index),
                "active_days": offset + (1 + index % 2 if churned else 6 + index % 3),
                "avg_order_value": offset + (50 + index if churned else 125 + index),
                "avg_items_per_invoice": offset + (2 + index % 3 if churned else 12 + index % 5),
                "unique_products": offset + (3 + index % 3 if churned else 20 + index % 5),
                "purchase_span_days": offset + (5 + index if churned else 100 + index),
                "churned": churned,
            }
        )
    return ChurnDataset(
        data=pd.DataFrame(records, columns=CHURN_DATA_COLUMNS),
        observation_end=pd.Timestamp(observation_end),
        prediction_end=pd.Timestamp(prediction_end),
        observation_transaction_count=rows * 3,
        prediction_transaction_count=rows,
    )


@pytest.fixture(scope="module")
def temporal_model_data() -> tuple[ChurnDataset, ChurnDataset, ChurnDataset]:
    return (
        _model_dataset("train", "2020-03-31", "2020-06-29"),
        _model_dataset("validation", "2020-06-29", "2020-09-27", offset=10.0),
        _model_dataset("test", "2020-09-27", "2020-12-26", offset=20.0),
    )


@pytest.fixture(scope="module")
def benchmark_results(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
) -> tuple[ChurnModelResult, ...]:
    train, validation, _ = temporal_model_data
    return evaluate_churn_models(train, validation)


def test_churn_target_uses_future_purchase_activity() -> None:
    dataset = build_churn_dataset(_boundary_transactions(), "2021-03-31")
    target = dataset.data.set_index("customer_id")["churned"]

    assert target.to_dict() == {"A": 0, "B": 1}
    assert "C" not in target.index


def test_temporal_boundaries_are_left_open_and_right_closed() -> None:
    dataset = build_churn_dataset(_boundary_transactions(), "2021-03-31")

    assert dataset.observation_end == pd.Timestamp("2021-03-31")
    assert dataset.prediction_end == pd.Timestamp("2021-06-29")
    assert dataset.observation_transaction_count == 3
    assert dataset.prediction_transaction_count == 2
    assert dataset.data.set_index("customer_id").loc["A", "frequency"] == 2


def test_future_transactions_do_not_leak_into_features() -> None:
    dataset = build_churn_dataset(_boundary_transactions(), "2021-03-31")
    customer_a = dataset.data.set_index("customer_id").loc["A"]

    assert customer_a["monetary"] == pytest.approx(50.0)
    assert customer_a["total_quantity"] == pytest.approx(5.0)
    assert customer_a["recency_days"] == 1


def test_churn_dataset_has_required_finite_feature_schema() -> None:
    dataset = build_churn_dataset(_boundary_transactions(), "2021-03-31")

    assert dataset.data.columns.tolist() == list(CHURN_DATA_COLUMNS)
    assert tuple(dataset.data.loc[:, list(CHURN_FEATURE_COLUMNS)].columns) == CHURN_FEATURE_COLUMNS
    assert np.isfinite(dataset.data.loc[:, list(CHURN_FEATURE_COLUMNS)].to_numpy()).all()
    assert dataset.data["customer_id"].is_unique


def test_temporal_split_has_three_ordered_non_overlapping_label_windows() -> None:
    split = build_temporal_churn_split(_split_transactions())

    assert split.train.customer_count == 2
    assert split.validation.customer_count == 3
    assert split.test.customer_count == 4
    assert split.train.prediction_end == split.validation.observation_end
    assert split.validation.prediction_end == split.test.observation_end
    assert set(split.train.data["churned"]) == {0, 1}
    assert set(split.validation.data["churned"]) == {0, 1}
    assert set(split.test.data["churned"]) == {0, 1}


def test_transaction_input_is_not_mutated() -> None:
    transactions = _boundary_transactions()
    original = transactions.copy(deep=True)

    build_churn_dataset(transactions, "2021-03-31")

    assert_frame_equal(transactions, original)


def test_missing_columns_and_non_finite_values_are_rejected() -> None:
    missing = _boundary_transactions().drop(columns=["country"])
    with pytest.raises(ValueError, match="missing required column.*country"):
        build_churn_dataset(missing, "2021-03-31")

    non_finite = _boundary_transactions().astype({"unit_price": float})
    non_finite.loc[0, "unit_price"] = np.inf
    with pytest.raises(ValueError, match="positive finite unit_price"):
        build_churn_dataset(non_finite, "2021-03-31")


def test_single_class_churn_target_is_rejected() -> None:
    transactions = pd.concat(
        [
            _boundary_transactions(),
            pd.DataFrame(
                [_transaction("B-2", "B", "2021-05-01")],
                columns=TRANSACTION_COLUMNS,
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="target must contain both binary classes"):
        build_churn_dataset(transactions, "2021-03-31")


def test_all_benchmark_models_train_on_temporal_data(
    benchmark_results: tuple[ChurnModelResult, ...],
) -> None:
    names = tuple(result.model_name for result in benchmark_results)

    assert names == CHURN_MODEL_NAMES


def test_evaluation_metrics_and_confusion_matrices_are_valid(
    benchmark_results: tuple[ChurnModelResult, ...],
) -> None:
    for result in benchmark_results:
        metrics = result.metrics
        values = (
            metrics.positive_prevalence,
            metrics.roc_auc,
            metrics.pr_auc,
            metrics.precision,
            metrics.recall,
            metrics.f1,
        )
        assert np.isfinite(values).all()
        assert all(0 <= value <= 1 for value in values)
        assert sum(sum(row) for row in metrics.confusion_matrix) == 60


def test_preprocessing_is_fitted_only_on_training_features(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
    benchmark_results: tuple[ChurnModelResult, ...],
) -> None:
    train, validation, _ = temporal_model_data
    logistic = next(
        result for result in benchmark_results if result.model_name == "logistic_regression"
    )
    fitted_mean = logistic.model.estimator.named_steps["scaler"].mean_
    train_mean = train.data.loc[:, list(CHURN_FEATURE_COLUMNS)].mean().to_numpy()
    validation_mean = validation.data.loc[:, list(CHURN_FEATURE_COLUMNS)].mean().to_numpy()

    assert np.allclose(fitted_mean, train_mean)
    assert not np.allclose(fitted_mean, validation_mean)


def test_model_selection_is_deterministic(
    benchmark_results: tuple[ChurnModelResult, ...],
) -> None:
    forward = select_best_churn_model(benchmark_results)
    reverse = select_best_churn_model(reversed(benchmark_results))

    assert forward.selected.model_name == reverse.selected.model_name
    assert forward.comparison["model"].tolist() == reverse.comparison["model"].tolist()
    assert forward.comparison.iloc[0]["selection_rank"] == 1


def test_final_model_refit_and_test_evaluation_are_finite(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
    benchmark_results: tuple[ChurnModelResult, ...],
) -> None:
    train, validation, test = temporal_model_data
    selection = select_best_churn_model(benchmark_results)
    model = fit_churn_model(selection.selected.model_name, (train, validation))

    metrics = evaluate_churn_model(model, test)

    assert model.threshold == pytest.approx(0.5)
    assert np.isfinite((metrics.pr_auc, metrics.roc_auc, metrics.f1)).all()


def test_customer_predictions_are_unique_binary_and_probabilistic(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
) -> None:
    train, validation, test = temporal_model_data
    model = fit_churn_model("logistic_regression", (train, validation))

    predictions = predict_churn_risk(model, test)

    assert predictions.columns.tolist() == list(PREDICTION_COLUMNS)
    assert predictions["customer_id"].is_unique
    assert predictions["churn_probability"].between(0, 1).all()
    assert predictions["churn_prediction"].isin([0, 1]).all()
    assert len(predictions) == test.customer_count


def test_risk_band_boundaries_are_deterministic(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, _, _ = temporal_model_data
    dataset = ChurnDataset(
        data=train.data.iloc[:3].copy(deep=True),
        observation_end=train.observation_end,
        prediction_end=train.prediction_end,
        observation_transaction_count=3,
        prediction_transaction_count=3,
    )
    model = fit_churn_model("logistic_regression", train)
    monkeypatch.setattr(
        churn_analysis,
        "predict_churn_probabilities",
        lambda fitted, evaluated: np.array([0.39, 0.40, 0.70]),
    )

    predictions = predict_churn_risk(model, dataset)

    assert predictions["risk_band"].tolist() == ["Low", "Medium", "High"]
    assert predictions["churn_prediction"].tolist() == [0, 0, 1]


def test_feature_importance_is_complete_and_ranked(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
) -> None:
    train, validation, _ = temporal_model_data
    model = fit_churn_model("logistic_regression", (train, validation))

    importance = build_churn_feature_importance(model)

    assert importance.columns.tolist() == list(FEATURE_IMPORTANCE_COLUMNS)
    assert set(importance["feature"]) == set(CHURN_FEATURE_COLUMNS)
    assert importance["rank"].tolist() == list(range(1, len(CHURN_FEATURE_COLUMNS) + 1))
    assert importance["importance"].is_monotonic_decreasing
    assert importance["direction"].isin(["increases_risk", "decreases_risk"]).all()


def _segment_inputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint = len(predictions) // 2
    assignments = pd.DataFrame(
        {
            "customer_id": predictions["customer_id"],
            "cluster": [10] * midpoint + [20] * (len(predictions) - midpoint),
        }
    )
    summary = pd.DataFrame(
        {
            "cluster": [10, 20],
            "segment_label": ["Lower-Activity Customers", "Higher-Activity Customers"],
        }
    )
    return assignments, summary


def test_segment_churn_summary_preserves_exact_customer_coverage(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
) -> None:
    train, validation, test = temporal_model_data
    model = fit_churn_model("logistic_regression", (train, validation))
    predictions = predict_churn_risk(model, test)
    assignments, summary = _segment_inputs(predictions)

    result = build_segment_churn_summary(predictions, assignments, summary)

    assert result.columns.tolist() == list(SEGMENT_CHURN_COLUMNS)
    assert result["customer_count"].sum() == len(predictions)
    assert result["segment_label"].str.contains("Customers").all()
    assert result["average_churn_probability"].between(0, 1).all()
    assert result["churn_rate"].between(0, 1).all()


def test_segment_summary_uses_labels_not_cluster_numbers(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
) -> None:
    train, validation, test = temporal_model_data
    model = fit_churn_model("logistic_regression", (train, validation))
    predictions = predict_churn_risk(model, test)
    assignments, summary = _segment_inputs(predictions)
    original = build_segment_churn_summary(predictions, assignments, summary)

    assignments["cluster"] = assignments["cluster"].replace({10: 20, 20: 10})
    summary["cluster"] = summary["cluster"].replace({10: 20, 20: 10})
    swapped = build_segment_churn_summary(predictions, assignments, summary)

    assert_frame_equal(original, swapped)
    assert "cluster" not in swapped.columns


def test_segment_join_rejects_incomplete_customer_coverage(
    temporal_model_data: tuple[ChurnDataset, ChurnDataset, ChurnDataset],
) -> None:
    train, validation, test = temporal_model_data
    model = fit_churn_model("logistic_regression", (train, validation))
    predictions = predict_churn_risk(model, test)
    assignments, summary = _segment_inputs(predictions)

    with pytest.raises(ValueError, match="exact customer coverage"):
        build_segment_churn_summary(predictions, assignments.iloc[:-1], summary)
