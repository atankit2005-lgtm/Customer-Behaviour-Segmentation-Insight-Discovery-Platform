from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from cbsi.data.schema import TRANSACTION_COLUMNS

CUSTOMER_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "recency_days",
    "frequency",
    "monetary",
    "total_quantity",
    "active_days",
    "avg_order_value",
    "avg_items_per_invoice",
    "unique_products",
    "purchase_span_days",
    "country",
)

NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = CUSTOMER_FEATURE_COLUMNS[1:-1]


def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eligible Stage 3 transactions into one feature row per customer."""
    _validate_input(transactions)

    working = transactions.loc[:, list(TRANSACTION_COLUMNS)].copy(deep=True)
    working["invoice_date"] = pd.to_datetime(working["invoice_date"], errors="coerce")
    working["quantity"] = pd.to_numeric(working["quantity"], errors="coerce")
    working["unit_price"] = pd.to_numeric(working["unit_price"], errors="coerce")
    _validate_stage3_values(working)

    reference_date = working["invoice_date"].max() + pd.Timedelta(days=1)
    working["line_value"] = working["quantity"] * working["unit_price"]
    working["purchase_date"] = working["invoice_date"].dt.normalize()

    grouped = working.groupby("customer_id", sort=True)
    features = grouped.agg(
        latest_invoice_date=("invoice_date", "max"),
        first_invoice_date=("invoice_date", "min"),
        frequency=("invoice_no", "nunique"),
        monetary=("line_value", "sum"),
        total_quantity=("quantity", "sum"),
        active_days=("purchase_date", "nunique"),
        unique_products=("stock_code", "nunique"),
    )

    features["recency_days"] = (reference_date - features["latest_invoice_date"]).dt.days
    features["avg_order_value"] = features["monetary"] / features["frequency"]
    features["avg_items_per_invoice"] = features["total_quantity"] / features["frequency"]
    features["purchase_span_days"] = (
        features["latest_invoice_date"] - features["first_invoice_date"]
    ).dt.days
    features["country"] = _primary_country(working)

    result = features.reset_index().loc[:, list(CUSTOMER_FEATURE_COLUMNS)]
    _validate_output(result)
    return result


def _primary_country(transactions: pd.DataFrame) -> pd.Series:
    counts = (
        transactions.groupby(["customer_id", "country"], sort=False)
        .size()
        .rename("transaction_count")
        .reset_index()
    )
    ranked = counts.sort_values(
        ["transaction_count", "country"], ascending=[False, True], kind="stable"
    )
    return ranked.drop_duplicates("customer_id").set_index("customer_id")["country"]


def _validate_input(transactions: pd.DataFrame) -> None:
    missing = [column for column in TRANSACTION_COLUMNS if column not in transactions.columns]
    if missing:
        raise ValueError(f"Transaction data is missing required column(s): {missing}")
    if transactions.empty:
        raise ValueError("Transaction data must contain at least one eligible transaction")


def _validate_stage3_values(transactions: pd.DataFrame) -> None:
    if transactions["customer_id"].isna().any():
        raise ValueError("Stage 3 transactions must not contain missing customer IDs")
    if transactions["invoice_date"].isna().any():
        raise ValueError("Stage 3 transactions must contain valid invoice dates")

    for column in ("invoice_no", "stock_code", "country"):
        if transactions[column].isna().any():
            raise ValueError(f"Stage 3 transactions must not contain missing {column} values")

    for column in ("quantity", "unit_price"):
        values = transactions[column]
        if (~np.isfinite(values) | values.le(0)).any():
            raise ValueError(f"Stage 3 transactions must contain positive finite {column} values")


def _validate_output(features: pd.DataFrame) -> None:
    numeric = features.loc[:, list(NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Calculated customer features contain non-finite values")

    positive = features.loc[:, list(NUMERIC_FEATURE_COLUMNS[:-1])]
    if positive.le(0).any().any() or features["purchase_span_days"].lt(0).any():
        raise ValueError("Calculated customer features violate their value constraints")
    if features["customer_id"].isna().any() or features["customer_id"].duplicated().any():
        raise ValueError("Calculated customer features require unique customer IDs")
    if features["country"].isna().any():
        raise ValueError("Calculated customer features require a country for every customer")
