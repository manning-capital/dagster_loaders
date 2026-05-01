from typing import List, Tuple, Optional

import pandas as pd
from dagster import MetadataValue


def compare_dataframes(
    table_1: pd.DataFrame,
    table_2: pd.DataFrame,
    key_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare two DataFrames based on primary keys and return different types of results.

    Args:
        table_1: First DataFrame (old/existing data)
        table_2: Second DataFrame (new data)
        key_columns: List of column names that form the primary key

    Returns:
        Tuple containing:
        - records_in_1_not_2: Records that exist in table_1 but not in table_2
        - records_in_2_not_1: Records that exist in table_2 but not in table_1
        - exact_matches: Records that exist in both tables with identical values
        - different_records: Records that exist in both tables but have different values in non-key columns
    """
    for key in key_columns:
        if key not in table_1.columns:
            raise ValueError(f"Key column '{key}' not found in table_1")
        if key not in table_2.columns:
            raise ValueError(f"Key column '{key}' not found in table_2")

    all_columns = list(set(table_1.columns) | set(table_2.columns))
    comparison_columns = [col for col in all_columns if col not in key_columns]

    symmetric_difference = set(table_1.columns) ^ set(table_2.columns)
    if len(symmetric_difference) != 0:
        raise ValueError(
            "Columns are not the same in both dataframes: {column_differences}".format(
                column_differences=", ".join(symmetric_difference)
            )
        )

    for col in comparison_columns:
        if table_1[col].dtype != table_2[col].dtype:
            raise ValueError(
                f"Column '{col}' has the type {table_1[col].dtype} in table_1 and {table_2[col].dtype} in table_2."
            )

    merged_df = table_1.merge(
        table_2, on=key_columns, how="outer", suffixes=("_1", "_2"), indicator=True
    )

    records_in_1_not_2 = merged_df[merged_df["_merge"] == "left_only"].copy()
    records_in_1_not_2 = records_in_1_not_2.drop(
        columns=[
            col
            for col in records_in_1_not_2.columns
            if col.endswith("_2") or col == "_merge"
        ]
    )
    records_in_1_not_2.columns = [
        col.replace("_1", "") for col in records_in_1_not_2.columns
    ]

    records_in_2_not_1 = merged_df[merged_df["_merge"] == "right_only"].copy()
    records_in_2_not_1 = records_in_2_not_1.drop(
        columns=[
            col
            for col in records_in_2_not_1.columns
            if col.endswith("_1") or col == "_merge"
        ]
    )
    records_in_2_not_1.columns = [
        col.replace("_2", "") for col in records_in_2_not_1.columns
    ]

    common_records = merged_df[merged_df["_merge"] == "both"].copy()

    if not common_records.empty:
        different_mask = pd.Series(False, index=common_records.index)

        for col in comparison_columns:
            col_1 = f"{col}_1"
            col_2 = f"{col}_2"

            if col_1 in common_records.columns and col_2 in common_records.columns:
                series_1 = common_records[col_1]
                series_2 = common_records[col_2]

                col_different = (series_1 != series_2) & ~(
                    pd.isna(series_1) & pd.isna(series_2)
                )
                different_mask = different_mask | col_different

        exact_matches = common_records[~different_mask].copy()
        different_records = common_records[different_mask].copy()

        for col in comparison_columns:
            col_1 = f"{col}_1"
            if col_1 in exact_matches.columns:
                exact_matches[col] = exact_matches[col_1]
        exact_matches = exact_matches.drop(
            columns=[
                col
                for col in exact_matches.columns
                if col.endswith("_1") or col.endswith("_2") or col == "_merge"
            ]
        )

        for col in comparison_columns:
            col_2 = f"{col}_2"
            if col_2 in different_records.columns:
                different_records[col] = different_records[col_2]
        different_records = different_records.drop(
            columns=[
                col
                for col in different_records.columns
                if col.endswith("_1") or col.endswith("_2") or col == "_merge"
            ]
        )
    else:
        exact_matches = pd.DataFrame(
            {
                column_name: pd.Series(dtype=table_1[column_name].dtype)
                for column_name in table_1.columns
            }
        )
        different_records = pd.DataFrame(
            {
                column_name: pd.Series(dtype=table_1[column_name].dtype)
                for column_name in table_1.columns
            }
        )

    return records_in_1_not_2, records_in_2_not_1, exact_matches, different_records


def df_to_md_metadata(
    df: pd.DataFrame,
    *,
    head: Optional[int] = None,
    empty_placeholder: str = "(empty)",
) -> MetadataValue:
    """Render a DataFrame as Markdown for Dagster asset/check metadata.

    Mirrors the existing `preview` convention used by Coindesk/Kraken/Coinbase
    assets: ``MetadataValue.md(df.head().to_markdown(index=False))``.

    - `head=None` renders the whole frame (use for bounded result sets).
    - `head=N` renders only the first N rows (use for previews of large frames).
    - Empty frames render as `empty_placeholder` so the UI shows something readable.
    """
    if df.empty:
        return MetadataValue.md(empty_placeholder)
    rendered = df.head(head) if head is not None else df
    return MetadataValue.md(rendered.to_markdown(index=False))
