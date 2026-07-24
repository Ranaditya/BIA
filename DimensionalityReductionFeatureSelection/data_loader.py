"""Load and validate the Iris dataset used by this project."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]


def find_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Return the single CSV file stored in the project data directory."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in: {data_dir}")
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise ValueError(f"Expected one CSV dataset in {data_dir}, found: {names}")
    return csv_files[0]


def load_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load the Iris CSV and validate its schema and values."""
    dataset_path = Path(filepath) if filepath is not None else find_dataset()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[EXPECTED_COLUMNS].copy()
    numeric_columns = EXPECTED_COLUMNS[:-1]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="raise")

    if df.empty:
        raise ValueError("The Iris dataset is empty.")
    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(f"Dataset contains missing values:\n{missing[missing > 0]}")
    if (df[numeric_columns] <= 0).any().any():
        raise ValueError("Iris measurements must be positive.")

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a compact structural overview of the dataset."""
    print("=" * 70)
    print("IRIS DATASET INSPECTION")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nSpecies counts:")
    print(df["species"].value_counts().sort_index().to_string())
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    iris_df = load_data()
    inspect_data(iris_df)
