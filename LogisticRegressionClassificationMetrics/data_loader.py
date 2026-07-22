"""Data loading and inspection utilities for Titanic classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "titanic.csv"


def load_data(dataset_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Titanic CSV into a pandas DataFrame."""
    path = Path(dataset_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def inspect_data(dataframe: pd.DataFrame, target_column: str = "Survived") -> None:
    """Print a concise data quality summary."""
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    print(f"Shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns")
    print("\nColumns:")
    print(list(dataframe.columns))

    print("\nDtypes:")
    print(dataframe.dtypes)

    print("\nMissing values per column:")
    print(dataframe.isna().sum())

    duplicate_rows = dataframe.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_rows}")

    if target_column in dataframe.columns:
        print(f"\nTarget distribution ({target_column}):")
        print(dataframe[target_column].value_counts(dropna=False).sort_index())

    print("=" * 70)
