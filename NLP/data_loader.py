"""Load and inspect the IMDb sentiment dataset."""

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "IMDB.csv"


def load_data(filepath: Path = DATA_FILE) -> pd.DataFrame:
    """Load the IMDb reviews and sentiment labels from CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    dataframe = pd.read_csv(filepath)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    return dataframe.dropna(subset=["review", "sentiment"]).reset_index(drop=True)


def inspect_data(dataframe: pd.DataFrame) -> None:
    """Print the dataset dimensions, types, missing values, and sample rows."""
    print("=" * 60)
    print("IMDB DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
    print("\nColumns and data types:")
    print(dataframe.dtypes.to_string())
    print("\nMissing values:")
    print(dataframe.isna().sum().to_string())
    print("\nSentiment counts:")
    print(dataframe["sentiment"].value_counts().to_string())
    print("\nFirst 3 rows:")
    print(dataframe.head(3).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    inspect_data(load_data())
