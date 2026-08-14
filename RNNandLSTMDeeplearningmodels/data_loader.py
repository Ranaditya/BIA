"""Load and inspect the bundled stock-price dataset."""

from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).parent / "data" / "Stock_price.csv"


def load_data(filepath: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load the CSV, parse dates, sort chronologically, and remove duplicate dates."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    dataframe = pd.read_csv(filepath, parse_dates=["Date"])
    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return (
        dataframe.sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )


def inspect_data(dataframe: pd.DataFrame) -> None:
    """Print a compact dataset-quality report."""
    print("\nDATASET INSPECTION")
    print("=" * 60)
    print(f"Rows / columns : {dataframe.shape[0]} / {dataframe.shape[1]}")
    print(f"Date range     : {dataframe['Date'].min().date()} to {dataframe['Date'].max().date()}")
    print("Missing values:")
    print(dataframe.isna().sum().to_string())
    print("\nFirst five rows:")
    print(dataframe.head().to_string(index=False))


if __name__ == "__main__":
    data = load_data()
    inspect_data(data)
