"""Utilities for loading and inspecting raw e-commerce transactions."""

from pathlib import Path

import pandas as pd


def load_raw(path="data/data.csv"):
	"""Load raw e-commerce transactions from a CSV file.

	Args:
		path: CSV path. Defaults to ``data/data.csv`` relative to this project.

	Returns:
		A DataFrame with ``InvoiceDate`` parsed as datetime and ``CustomerID``
		stored as a nullable string column.

	Raises:
		AssertionError: If the file is empty or does not contain ``CustomerID``.
	"""
	base_dir = Path(__file__).resolve().parent
	resolved_path = Path(path)
	if not resolved_path.is_absolute():
		resolved_path = base_dir / resolved_path

	df = pd.read_csv(
		resolved_path,
		encoding="latin-1",
		parse_dates=["InvoiceDate"],
		dtype={"CustomerID": "string"},
	)

	assert not df.empty, "loaded an empty dataframe"
	assert "CustomerID" in df.columns, "CustomerID column missing"
	return df


def describe_raw(df):
	"""Print basic quality and coverage statistics for raw transactions.

	Args:
		df: Raw transaction DataFrame returned by :func:`load_raw`.

	Returns:
		None. The shape, date range, cardinalities, missing values, and country
		count are printed to standard output.
	"""
	print(f"Shape: {df.shape}")
	print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
	print(f"Unique customers: {df['CustomerID'].nunique()}")
	print(f"Unique invoices: {df['InvoiceNo'].nunique()}")
	print("Missing values per column:")
	print(df.isna().sum())
	print(f"Unique countries: {df['Country'].nunique()}")
