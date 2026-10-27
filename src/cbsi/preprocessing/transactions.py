from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from cbsi.data.schema import TRANSACTION_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = TRANSACTION_COLUMNS
CANCELLATION_PREFIX: Final[str] = "C"


def preprocess_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Prepare canonical transactions for customer-level feature engineering.

    The input is not mutated. Exact duplicates, cancellations, returns, invalid dates,
    non-positive or non-finite quantities and prices, and unidentified customers are excluded.
    Missing descriptions are preserved.
    """
    _validate_columns(transactions)

    cleaned = transactions.loc[:, list(REQUIRED_COLUMNS)].copy(deep=True)
    cleaned = cleaned.drop_duplicates()

    cleaned["invoice_date"] = pd.to_datetime(cleaned["invoice_date"], errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce")

    cancellation_or_return = _cancellation_or_return_mask(cleaned)
    valid_quantity = np.isfinite(cleaned["quantity"]) & cleaned["quantity"].gt(0)
    valid_price = np.isfinite(cleaned["unit_price"]) & cleaned["unit_price"].gt(0)

    eligible = (
        cleaned["invoice_date"].notna()
        & ~cancellation_or_return
        & valid_quantity
        & valid_price
        & cleaned["customer_id"].notna()
    )

    return cleaned.loc[eligible, list(TRANSACTION_COLUMNS)].reset_index(drop=True)


def _cancellation_or_return_mask(transactions: pd.DataFrame) -> pd.Series:
    cancelled_invoice = (
        transactions["invoice_no"].astype("string").str.startswith(CANCELLATION_PREFIX, na=False)
    )
    returned_quantity = transactions["quantity"].lt(0)
    return cancelled_invoice | returned_quantity


def _validate_columns(transactions: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in transactions.columns]
    if missing:
        raise ValueError(f"Transaction data is missing required column(s): {missing}")
