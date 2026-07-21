"""Data loading and inspection utilities for car mileage regression."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "3304db2c078848f8ad85537da4d87645_car_(1).csv"


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the car dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    return pd.read_csv(file_path)


def inspect_data(dataframe: pd.DataFrame) -> None:
    """Print quick shape/type/missing/duplicate inspection output."""
    print("=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
    print("\nColumn dtypes:")
    print(dataframe.dtypes.to_string())
    print("\nMissing values per column:")
    print(dataframe.isna().sum().to_string())
    print(f"\nDuplicate rows: {dataframe.duplicated().sum()}")
    print("=" * 60)
