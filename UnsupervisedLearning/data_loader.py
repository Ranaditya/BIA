"""
data_loader.py
===============
Load and inspect the wholesale customer dataset.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data() -> pd.DataFrame:
    """Load the wholesale customer dataset from the data directory."""
    data_files = list(DATA_DIR.glob("*.csv"))
    if not data_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    data_path = data_files[0]
    df = pd.read_csv(data_path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print dataset overview details."""
    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDescriptive statistics:")
    print(df.describe().T)
