"""
data_loader.py
==============
Handles loading the Apple Stock dataset from CSV and provides
a quick inspection of its structure and contents.

Step: 1 - Load Dataset
"""

from pathlib import Path

import pandas as pd


# Path to the dataset relative to this file
DATA_FILE = Path(__file__).parent / "data" / "b5b3b342c8694c6bafd4f67814ce5220_apple_stocks.csv"


def load_data(filepath: Path = DATA_FILE) -> pd.DataFrame:
    """
    Load the Apple Stock dataset from a CSV file into a DataFrame.

    Args:
        filepath (Path): Path to the CSV file. Defaults to the bundled dataset.

    Returns:
        pd.DataFrame: Raw Apple Stock data.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a structured inspection of the DataFrame including shape,
    data types, missing values, and a preview of the first few rows.

    Args:
        df (pd.DataFrame): The Apple Stock DataFrame to inspect.
    """
    print("=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)
    print(f"\nShape: {df.shape} (rows, columns)")
    print(f"\nColumn Names and Data Types:\n{df.dtypes.to_string()}")
    print(f"\nMissing Values:\n{df.isnull().sum().to_string()}")
    print(f"\nFirst 5 Rows:\n{df.head().to_string()}")
    print(f"\nLast 5 Rows:\n{df.tail().to_string()}")
    print("=" * 50)
