from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cbsi.data.ingestion import EXPECTED_SHEETS, SOURCE_TO_CANONICAL, load_transactions
from cbsi.data.schema import TRANSACTION_COLUMNS

SOURCE_COLUMNS = list(SOURCE_TO_CANONICAL)

FIRST_SHEET_ROWS: list[list[object]] = [
    [489434, "85048", "BALL LIGHTS", 12, "2009-12-01 07:45:00", 6.95, 13085, "United Kingdom"],
    [489435, "79323P", "CHERRY LIGHTS", 6, "2009-12-01 09:15:00", 6.75, 13086, "France"],
]

SECOND_SHEET_ROWS: list[list[object]] = [
    [536365, "85123A", "HEART HOLDER", 6, "2010-12-01 08:26:00", 2.55, 17850, "United Kingdom"],
    [536366, "71053", "METAL LANTERN", 3, "2010-12-02 10:03:00", 3.39, 17851, "Germany"],
    [536367, "84406B", "CUPID HEARTS", 8, "2010-12-03 11:30:00", 2.75, 17852, "EIRE"],
]

RETURN_ROWS: list[list[object]] = [
    ["C489449", "22087", "RETURNED BUNTING", -12, "2009-12-01 10:33:00", 2.95, 16321, "Australia"],
    [489450, "22088", "NO CUSTOMER", 4, "2009-12-01 10:40:00", 1.25, None, "United Kingdom"],
]


def _frame(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return path


def _write_both_sheets(tmp_path: Path, first: pd.DataFrame, second: pd.DataFrame) -> Path:
    sheets = dict(zip(EXPECTED_SHEETS, [first, second], strict=True))
    return _write_workbook(tmp_path / "online_retail_II.xlsx", sheets)


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    return _write_both_sheets(tmp_path, _frame(FIRST_SHEET_ROWS), _frame(SECOND_SHEET_ROWS))


def test_load_transactions_concatenates_both_sheets(workbook: Path) -> None:
    transactions = load_transactions(workbook)

    assert len(transactions) == 5
    assert transactions["invoice_no"].tolist() == [489434, 489435, 536365, 536366, 536367]


def test_load_transactions_returns_canonical_columns(workbook: Path) -> None:
    transactions = load_transactions(workbook)

    assert tuple(transactions.columns) == TRANSACTION_COLUMNS


def test_load_transactions_parses_invoice_date(workbook: Path) -> None:
    transactions = load_transactions(workbook)

    assert pd.api.types.is_datetime64_any_dtype(transactions["invoice_date"])
    assert transactions["invoice_date"].iloc[0] == pd.Timestamp("2009-12-01 07:45:00")
    assert transactions["invoice_date"].iloc[-1] == pd.Timestamp("2010-12-03 11:30:00")


def test_load_transactions_preserves_returns_and_missing_customers(tmp_path: Path) -> None:
    path = _write_both_sheets(tmp_path, _frame(RETURN_ROWS), _frame(SECOND_SHEET_ROWS))

    transactions = load_transactions(path)

    assert len(transactions) == 5
    assert (transactions["quantity"] < 0).sum() == 1
    assert transactions["customer_id"].isna().sum() == 1
    assert "C489449" in transactions["invoice_no"].tolist()


def test_load_transactions_rejects_missing_sheet(tmp_path: Path) -> None:
    sheets = {EXPECTED_SHEETS[0]: _frame(FIRST_SHEET_ROWS)}
    path = _write_workbook(tmp_path / "online_retail_II.xlsx", sheets)

    with pytest.raises(ValueError, match="missing expected sheet"):
        load_transactions(path)


def test_load_transactions_rejects_missing_column(tmp_path: Path) -> None:
    incomplete = _frame(SECOND_SHEET_ROWS).drop(columns=["Price"])
    path = _write_both_sheets(tmp_path, _frame(FIRST_SHEET_ROWS), incomplete)

    with pytest.raises(ValueError, match="missing expected column"):
        load_transactions(path)


def test_load_transactions_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "absent.xlsx")
