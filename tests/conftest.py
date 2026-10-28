from __future__ import annotations

import pandas as pd
import pytest

from cbsi.app.pipeline import AnalysisBundle, AnalysisConfig, analyze_transactions
from cbsi.data.schema import TRANSACTION_COLUMNS


@pytest.fixture(scope="module")
def app_transactions() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2021-01-01")
    for index in range(40):
        purchases = 6 + index % 5 if index >= 20 else 1 + index % 3
        days = [index % 10 + purchase * (2 + index % 4) for purchase in range(purchases)]
        if index % 3 == 0:
            days.extend((130 + index % 10, 220 + index % 10, 320 + index % 10))
        for purchase, day in enumerate(days):
            records.append(
                {
                    "invoice_no": f"I-{index}-{purchase}",
                    "stock_code": f"SKU-{index}-{purchase % (2 + index % 5)}",
                    "description": "Synthetic test product",
                    "quantity": (20 if index >= 20 else 1) + index % 7 + purchase,
                    "invoice_date": start + pd.Timedelta(days=day),
                    "unit_price": 2.0 + index / 10,
                    "customer_id": f"U-{index:03d}",
                    "country": "France" if index % 2 else "United Kingdom",
                }
            )
    records.append(
        {
            "invoice_no": "I-NEW",
            "stock_code": "SKU-NEW",
            "description": "Future-only test customer",
            "quantity": 1,
            "invoice_date": pd.Timestamp("2021-12-31"),
            "unit_price": 3.0,
            "customer_id": "FUTURE",
            "country": "France",
        }
    )
    return pd.DataFrame(records, columns=TRANSACTION_COLUMNS)


@pytest.fixture(scope="module")
def app_config() -> AnalysisConfig:
    return AnalysisConfig(
        k_values=(2, 3),
        dbscan_eps_values=(0.5,),
        dbscan_min_samples_values=(3,),
        silhouette_sample_size=None,
    )


@pytest.fixture(scope="module")
def analysis_bundle(app_transactions: pd.DataFrame, app_config: AnalysisConfig) -> AnalysisBundle:
    return analyze_transactions(app_transactions, config=app_config)
