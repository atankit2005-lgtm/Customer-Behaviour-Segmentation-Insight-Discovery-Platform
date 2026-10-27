from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from cbsi.data.schema import TRANSACTION_COLUMNS

EXPECTED_SHEETS: Final[tuple[str, ...]] = ("Year 2009-2010", "Year 2010-2011")

SOURCE_TO_CANONICAL: Final[dict[str, str]] = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load raw Online Retail II transactions into the canonical transaction schema.

    Both workbook sheets are concatenated. Transaction semantics are preserved: no rows
    are filtered, deduplicated or aggregated, and returns/cancellations are retained.
    """
    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Transaction workbook not found: {workbook_path}")

    with pd.ExcelFile(workbook_path) as workbook:
        _validate_sheets(workbook.sheet_names, workbook_path)
        frames = [_read_sheet(workbook, sheet) for sheet in EXPECTED_SHEETS]

    transactions = pd.concat(frames, ignore_index=True)
    transactions["invoice_date"] = pd.to_datetime(transactions["invoice_date"])
    return transactions


def _validate_sheets(available: list[str], workbook_path: Path) -> None:
    missing = [sheet for sheet in EXPECTED_SHEETS if sheet not in available]
    if missing:
        raise ValueError(
            f"{workbook_path} is missing expected sheet(s): {missing}. Found: {available}"
        )


def _read_sheet(workbook: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    missing = [column for column in SOURCE_TO_CANONICAL if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Sheet {sheet!r} is missing expected column(s): {missing}. "
            f"Found: {list(frame.columns)}"
        )

    renamed = frame.rename(columns=SOURCE_TO_CANONICAL)
    return renamed.loc[:, list(TRANSACTION_COLUMNS)]
