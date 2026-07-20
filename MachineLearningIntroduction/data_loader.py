"""
data_loader.py
==============
Handles loading the housing dataset from CSV and provides
a quick inspection of its structure and contents.

Step: 1 - Load Dataset
"""

from pathlib import Path

import pandas as pd


# Path to the dataset relative to this file
DATA_FILE = Path(__file__).parent / "data" / "housing_dataset.csv"


def load_data(filepath: Path = DATA_FILE) -> pd.DataFrame:
    """
    Load the housing dataset from a CSV file into a DataFrame.

    Args:
        filepath (Path): Path to the CSV file. Defaults to the bundled dataset.

    Returns:
        pd.DataFrame: Raw housing data.

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
        df (pd.DataFrame): The housing DataFrame to inspect.
    """
    print("=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)

    # Number of rows and columns
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Column names and their data types
    print("\nColumns & Data Types:")
    print(df.dtypes.to_string())

    # Check for missing values in each column
    print("\nMissing Values Per Column:")
    missing = df.isnull().sum()
    print(missing.to_string())

    # First 5 rows for a quick preview
    print("\nFirst 5 Rows:")
    print(df.head().to_string(index=False))

    print("=" * 50)


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
