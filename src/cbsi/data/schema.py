from __future__ import annotations

from dataclasses import dataclass
from typing import Final

TRANSACTION_COLUMNS: Final[tuple[str, ...]] = (
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
)


@dataclass(frozen=True)
class TransactionSchema:
    """Canonical schema for transaction-level input data."""

    invoice_no: str = "invoice_no"
    stock_code: str = "stock_code"
    description: str = "description"
    quantity: str = "quantity"
    invoice_date: str = "invoice_date"
    unit_price: str = "unit_price"
    customer_id: str = "customer_id"
    country: str = "country"
