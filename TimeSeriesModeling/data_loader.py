"""Load and inspect the Auser aquifer rainfall dataset."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


def find_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Return the single CSV dataset stored in ``data_dir``."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV dataset found in {data_dir}")
    if len(files) > 1:
        raise ValueError(f"Expected one CSV dataset in {data_dir}, found {len(files)}")
    return files[0]


def load_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset without changing its contents."""
    path = filepath or find_dataset()
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    """Print a compact data-quality summary."""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nMissing values (columns with missing data):")
    print(df.isna().sum().loc[lambda values: values.gt(0)].to_string())
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    data = load_data()
    inspect_data(data)
