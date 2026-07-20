from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).with_name("b5b3b342c8694c6bafd4f67814ce5220_apple_stocks.csv")


def load_apple_stock_data(csv_path: Path = DATA_FILE) -> pd.DataFrame:
	"""Load the Apple stock history CSV into a pandas DataFrame."""
	return pd.read_csv(csv_path)


def summarize_stock_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a compact summary useful for quick validation."""
	summary = df.copy()
	summary["Date"] = pd.to_datetime(summary["Date"], format="%d-%b-%y")
	summary = summary.sort_values("Date")
	return summary


def main() -> None:
	df = load_apple_stock_data()
	summary = summarize_stock_data(df)

	print("Loaded rows:", len(summary))
	print(summary.head().to_string(index=False))


if __name__ == "__main__":
	main()
