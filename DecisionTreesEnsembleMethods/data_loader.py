"""
data_loader.py
==============
Handles loading the Iris dataset from CSV (downloaded file)
or from scikit-learn as a fallback.

Step: 1 - Import Libraries & Load Dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


DATA_DIR = Path(__file__).parent / "data"


def _find_default_csv() -> Path:
    """
    Find the first CSV file in the local data directory.

    Returns:
        Path: Path to the discovered CSV file.

    Raises:
        FileNotFoundError: If no CSV file exists in the data directory.
    """
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in: {DATA_DIR}")
    return csv_files[0]


def load_data(filepath: Path | None = None, source: str = "csv") -> pd.DataFrame:
    """
    Load the Iris dataset.

    Args:
        filepath (Path | None): CSV file path. If None, auto-discovers CSV in data/.
        source (str): "csv" to load with pandas.read_csv, "sklearn" to load from
                      sklearn.datasets.load_iris.

    Returns:
        pd.DataFrame: Iris dataset DataFrame.
    """
    if source == "sklearn":
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df.rename(columns={"target": "Species"}, inplace=True)
        df["Species"] = df["Species"].map(dict(enumerate(iris.target_names)))
        return df

    csv_path = filepath if filepath is not None else _find_default_csv()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    return pd.read_csv(csv_path)


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a quick structure check of the dataset.

    Args:
        df (pd.DataFrame): Iris DataFrame.
    """
    print("=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns & Data Types:")
    print(df.dtypes.to_string())
    print("\nMissing Values Per Column:")
    print(df.isnull().sum().to_string())
    print("\nFirst 5 Rows:")
    print(df.head().to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    iris_df = load_data(source="csv")
    inspect_data(iris_df)
