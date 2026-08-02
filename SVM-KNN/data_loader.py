"""Load and inspect the Iris CSV dataset."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


def find_dataset() -> Path:
    """Return the single CSV file stored in the project's data folder."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {DATA_DIR}")
    if len(csv_files) > 1:
        raise RuntimeError(f"Expected one CSV file in {DATA_DIR}, found {len(csv_files)}")
    return csv_files[0]


def load_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load the Iris data and validate its required columns."""
    path = filepath or find_dataset()
    dataframe = pd.read_csv(path)
    required = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return dataframe


def inspect_data(dataframe: pd.DataFrame) -> None:
    """Print the dataset structure and a short preview."""
    print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
    print("\nData types:")
    print(dataframe.dtypes.to_string())
    print("\nMissing values:")
    print(dataframe.isna().sum().to_string())
    print("\nFirst five rows:")
    print(dataframe.head().to_string(index=False))
