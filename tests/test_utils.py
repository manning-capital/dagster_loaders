import pandas as pd
import pytest

from dagster_loaders.utils import compare_dataframes


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_returns_added_records():
    old = _df([{"k": "a", "v": 1}])
    new = _df([{"k": "a", "v": 1}, {"k": "b", "v": 2}])

    only_old, added, exact, different = compare_dataframes(old, new, ["k"])

    assert only_old.empty
    assert len(added) == 1
    assert added.iloc[0]["k"] == "b"
    assert len(exact) == 1
    assert different.empty


def test_returns_removed_records():
    old = _df([{"k": "a", "v": 1}, {"k": "b", "v": 2}])
    new = _df([{"k": "a", "v": 1}])

    only_old, added, exact, different = compare_dataframes(old, new, ["k"])

    assert len(only_old) == 1
    assert only_old.iloc[0]["k"] == "b"
    assert added.empty
    assert len(exact) == 1
    assert different.empty


def test_returns_different_records_with_new_values():
    old = _df([{"k": "a", "v": 1}])
    new = _df([{"k": "a", "v": 99}])

    _, added, exact, different = compare_dataframes(old, new, ["k"])

    assert added.empty
    assert exact.empty
    assert len(different) == 1
    assert different.iloc[0]["v"] == 99


def test_treats_nan_equal_to_nan():
    old = _df([{"k": "a", "v": float("nan")}])
    new = _df([{"k": "a", "v": float("nan")}])

    _, added, exact, different = compare_dataframes(old, new, ["k"])

    assert added.empty
    assert different.empty
    assert len(exact) == 1


def test_handles_empty_inputs():
    old = _df([{"k": "a", "v": 1}]).iloc[0:0]
    new = _df([{"k": "a", "v": 1}]).iloc[0:0]

    only_old, added, exact, different = compare_dataframes(old, new, ["k"])

    assert only_old.empty
    assert added.empty
    assert exact.empty
    assert different.empty


def test_raises_when_columns_differ():
    old = _df([{"k": "a", "v": 1}])
    new = _df([{"k": "a", "w": 1}])

    with pytest.raises(ValueError, match="Columns are not the same"):
        compare_dataframes(old, new, ["k"])


def test_raises_on_missing_key_column():
    old = _df([{"k": "a", "v": 1}])
    new = _df([{"k": "a", "v": 1}])

    with pytest.raises(ValueError, match="Key column 'missing' not found"):
        compare_dataframes(old, new, ["missing"])


def test_raises_on_dtype_mismatch():
    old = pd.DataFrame({"k": ["a"], "v": pd.Series([1], dtype="int64")})
    new = pd.DataFrame({"k": ["a"], "v": pd.Series([1], dtype="float64")})

    with pytest.raises(ValueError, match="has the type"):
        compare_dataframes(old, new, ["k"])
