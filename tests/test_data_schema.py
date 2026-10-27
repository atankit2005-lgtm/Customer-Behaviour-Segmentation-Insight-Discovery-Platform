from cbsi.data.schema import TRANSACTION_COLUMNS, TransactionSchema


def test_transaction_columns_match_schema() -> None:
    schema = TransactionSchema()

    assert tuple(vars(schema).values()) == TRANSACTION_COLUMNS
    assert len(TRANSACTION_COLUMNS) == 8
    assert len(set(TRANSACTION_COLUMNS)) == 8
