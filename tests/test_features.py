from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cbsi.data.schema import TRANSACTION_COLUMNS
from cbsi.features.customer import (
    CUSTOMER_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_customer_features,
)


def _transaction(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "invoice_no": "1001",
        "stock_code": "A",
        "description": "PRODUCT A",
        "quantity": 2,
        "invoice_date": pd.Timestamp("2021-01-01 10:00:00"),
        "unit_price": 10.0,
        "customer_id": 1,
        "country": "United Kingdom",
    }
    row.update(overrides)
    return row


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)


@pytest.fixture
def transactions() -> pd.DataFrame:
    return _frame(
        _transaction(),
        _transaction(stock_code="B", description="PRODUCT B", quantity=1, unit_price=5.0),
        _transaction(
            invoice_no="1002",
            quantity=3,
            unit_price=4.0,
            invoice_date=pd.Timestamp("2021-01-03 10:00:00"),
            country="France",
        ),
        _transaction(
            invoice_no="2001",
            stock_code="C",
            description="PRODUCT C",
            quantity=4,
            unit_price=3.0,
            invoice_date=pd.Timestamp("2021-01-10 10:00:00"),
            customer_id=2,
            country="Germany",
        ),
        _transaction(
            invoice_no="2002",
            stock_code="D",
            description="PRODUCT D",
            quantity=2,
            unit_price=6.0,
            invoice_date=pd.Timestamp("2021-01-10 10:00:00"),
            customer_id=2,
            country="France",
        ),
    )


def _customer(features: pd.DataFrame, customer_id: int) -> pd.Series:
    return features.set_index("customer_id").loc[customer_id]


def test_one_customer_produces_one_feature_row() -> None:
    features = build_customer_features(_frame(_transaction()))

    assert len(features) == 1
    assert features.loc[0, "customer_id"] == 1


def test_frequency_counts_unique_invoices(transactions: pd.DataFrame) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["frequency"] == 2


def test_monetary_sums_quantity_times_unit_price(transactions: pd.DataFrame) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["monetary"] == pytest.approx(37.0)


def test_total_quantity_is_summed(transactions: pd.DataFrame) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["total_quantity"] == 6


def test_recency_uses_global_max_date_plus_one_day(transactions: pd.DataFrame) -> None:
    features = build_customer_features(transactions)

    assert _customer(features, 1)["recency_days"] == 8
    assert _customer(features, 2)["recency_days"] == 1


def test_active_days_counts_unique_calendar_dates(transactions: pd.DataFrame) -> None:
    features = build_customer_features(transactions)

    assert _customer(features, 1)["active_days"] == 2
    assert _customer(features, 2)["active_days"] == 1


def test_average_order_value_uses_unique_invoice_frequency(
    transactions: pd.DataFrame,
) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["avg_order_value"] == pytest.approx(18.5)


def test_average_items_per_invoice_uses_unique_invoice_frequency(
    transactions: pd.DataFrame,
) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["avg_items_per_invoice"] == pytest.approx(3.0)


def test_unique_products_counts_stock_codes(transactions: pd.DataFrame) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["unique_products"] == 2


def test_purchase_span_is_first_to_last_purchase_difference(
    transactions: pd.DataFrame,
) -> None:
    features = build_customer_features(transactions)

    assert _customer(features, 1)["purchase_span_days"] == 2
    assert _customer(features, 2)["purchase_span_days"] == 0


def test_country_uses_most_frequent_transaction_country(transactions: pd.DataFrame) -> None:
    customer = _customer(build_customer_features(transactions), 1)

    assert customer["country"] == "United Kingdom"


def test_country_tie_breaks_alphabetically(transactions: pd.DataFrame) -> None:
    customer = _customer(build_customer_features(transactions), 2)

    assert customer["country"] == "France"


def test_multiple_customers_have_one_unique_row_each(transactions: pd.DataFrame) -> None:
    features = build_customer_features(transactions)

    assert len(features) == transactions["customer_id"].nunique()
    assert not features["customer_id"].duplicated().any()
    assert not features["customer_id"].isna().any()


def test_input_dataframe_is_not_mutated(transactions: pd.DataFrame) -> None:
    original = transactions.copy(deep=True)

    build_customer_features(transactions)

    assert_frame_equal(transactions, original)


def test_missing_required_columns_raise_value_error(transactions: pd.DataFrame) -> None:
    incomplete = transactions.drop(columns=["invoice_no", "stock_code"])

    with pytest.raises(ValueError, match=r"invoice_no.*stock_code"):
        build_customer_features(incomplete)


def test_output_uses_exact_feature_schema(transactions: pd.DataFrame) -> None:
    features = build_customer_features(transactions)

    assert tuple(features.columns) == CUSTOMER_FEATURE_COLUMNS


def test_calculated_numeric_features_are_finite(transactions: pd.DataFrame) -> None:
    features = build_customer_features(transactions)
    numeric = features.loc[:, list(NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype=float)

    assert np.isfinite(numeric).all()
    assert (features.loc[:, list(NUMERIC_FEATURE_COLUMNS[:-1])] > 0).all().all()
    assert features["purchase_span_days"].ge(0).all()


def test_accepts_realistic_stage3_dtypes() -> None:
    stage3 = _frame(
        _transaction(
            invoice_no=536365,
            stock_code="85123A",
            quantity=np.int64(6),
            invoice_date=pd.Timestamp("2010-12-01 08:26:00"),
            unit_price=np.float64(2.55),
            customer_id=np.float64(17850),
        )
    )

    features = build_customer_features(stage3)

    assert len(features) == 1
    assert features.loc[0, "customer_id"] == 17850


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("invoice_date", "invalid"),
        ("quantity", np.inf),
        ("unit_price", 0),
        ("customer_id", None),
        ("invoice_no", None),
        ("stock_code", None),
        ("country", None),
    ],
)
def test_rejects_values_outside_stage3_contract(
    column: str,
    value: object,
) -> None:
    invalid = _frame(_transaction(**{column: value}))

    with pytest.raises(ValueError, match="Stage 3 transactions"):
        build_customer_features(invalid)
