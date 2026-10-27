from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cbsi.data.schema import TRANSACTION_COLUMNS
from cbsi.preprocessing.transactions import preprocess_transactions


def _transaction(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "invoice_no": "536365",
        "stock_code": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "quantity": 6,
        "invoice_date": "2010-12-01 08:26:00",
        "unit_price": 2.55,
        "customer_id": 17850,
        "country": "United Kingdom",
    }
    row.update(overrides)
    return row


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)


def test_valid_transaction_survives() -> None:
    result = preprocess_transactions(_frame(_transaction()))

    assert len(result) == 1
    assert result.loc[0, "invoice_no"] == "536365"


def test_cancellation_prefix_with_positive_quantity_is_excluded() -> None:
    result = preprocess_transactions(_frame(_transaction(invoice_no="C536365", quantity=6)))

    assert result.empty


def test_negative_quantity_without_cancellation_prefix_is_excluded() -> None:
    result = preprocess_transactions(_frame(_transaction(invoice_no="536365", quantity=-6)))

    assert result.empty


@pytest.mark.parametrize("quantity", [0, "invalid", np.nan, np.inf, -np.inf])
def test_invalid_quantity_is_excluded(quantity: object) -> None:
    result = preprocess_transactions(_frame(_transaction(quantity=quantity)))

    assert result.empty


@pytest.mark.parametrize("unit_price", [0, -1.0, "invalid", np.nan, np.inf, -np.inf])
def test_invalid_price_is_excluded(unit_price: object) -> None:
    result = preprocess_transactions(_frame(_transaction(unit_price=unit_price)))

    assert result.empty


def test_missing_customer_id_is_excluded() -> None:
    result = preprocess_transactions(_frame(_transaction(customer_id=None)))

    assert result.empty


def test_missing_description_survives() -> None:
    result = preprocess_transactions(_frame(_transaction(description=None)))

    assert len(result) == 1
    assert pd.isna(result.loc[0, "description"])


@pytest.mark.parametrize("invoice_date", [None, "not-a-date"])
def test_invalid_invoice_date_is_excluded(invoice_date: object) -> None:
    result = preprocess_transactions(_frame(_transaction(invoice_date=invoice_date)))

    assert result.empty


def test_exact_duplicate_is_removed() -> None:
    row = _transaction()

    result = preprocess_transactions(_frame(row, row.copy()))

    assert len(result) == 1


def test_distinct_rows_sharing_invoice_number_survive() -> None:
    first = _transaction(stock_code="85123A")
    second = _transaction(stock_code="71053", description="WHITE METAL LANTERN")

    result = preprocess_transactions(_frame(first, second))

    assert len(result) == 2
    assert result["stock_code"].tolist() == ["85123A", "71053"]


def test_distinct_raw_values_are_not_deduplicated_after_numeric_coercion() -> None:
    first = _transaction(quantity="6")
    second = _transaction(quantity=6)

    result = preprocess_transactions(_frame(first, second))

    assert len(result) == 2


def test_input_dataframe_is_not_mutated() -> None:
    transactions = _frame(
        _transaction(quantity="6", invoice_date="2010-12-01 08:26:00"),
        _transaction(invoice_no="C536366", quantity=-1),
    )
    original = transactions.copy(deep=True)

    preprocess_transactions(transactions)

    assert_frame_equal(transactions, original)


def test_missing_required_columns_raise_value_error() -> None:
    transactions = _frame(_transaction()).drop(columns=["customer_id", "country"])

    with pytest.raises(ValueError, match=r"customer_id.*country"):
        preprocess_transactions(transactions)


def test_output_uses_canonical_schema_and_reset_index() -> None:
    transactions = _frame(
        _transaction(invoice_no="C536364"),
        _transaction(invoice_no="536365"),
    )
    transactions.index = [10, 20]
    transactions["unexpected"] = "ignored"

    result = preprocess_transactions(transactions)

    assert tuple(result.columns) == TRANSACTION_COLUMNS
    assert result.index.tolist() == [0]


def test_numeric_and_datetime_columns_are_normalized() -> None:
    transactions = _frame(
        _transaction(quantity="6", unit_price="2.55", invoice_date="2010-12-01 08:26:00")
    )

    result = preprocess_transactions(transactions)

    assert pd.api.types.is_numeric_dtype(result["quantity"])
    assert pd.api.types.is_numeric_dtype(result["unit_price"])
    assert pd.api.types.is_datetime64_any_dtype(result["invoice_date"])
