"""Cleaning helpers for raw e-commerce transaction data."""

import pandas as pd
import numpy as np


def clean_transactions(df,
					   drop_missing_customers=True,
					   remove_cancellations=True,
					   min_unit_price=0.01):
	"""Clean raw transactions and add the line-item total price.

	Missing customers, cancellation invoices, non-positive prices, non-numeric
	stock codes, and duplicate rows are removed. Keeping only numeric stock
	codes excludes service and adjustment codes (for example POST and BANK
	CHARGES), but it also drops product variants with letter suffixes such as
	85123A.

	Args:
		df: Raw transaction DataFrame.
		drop_missing_customers: Whether to remove rows without a CustomerID.
		remove_cancellations: Whether to remove invoices beginning with ``C``.
		min_unit_price: Minimum permitted UnitPrice, inclusive.

	Returns:
		A cleaned copy of ``df`` with a ``TotalPrice`` column. Rows removed
		by each step are recorded in ``df.attrs["cleaning_steps"]``.
	"""
	df = df.copy()
	rows_start = len(df)
	steps = []

	if drop_missing_customers:
		before = len(df)
		df = df.dropna(subset=["CustomerID"])
		print(f"Removed missing customers: {before - len(df)} rows")
		steps.append({"step": "Missing CustomerID", "removed": before - len(df), "remaining": len(df)})

	if remove_cancellations:
		before = len(df)
		cancellation_mask = df["InvoiceNo"].astype("string").str.startswith("C", na=False)
		df = df.loc[~cancellation_mask]
		print(f"Removed cancellations: {before - len(df)} rows")
		steps.append({"step": "Cancellation invoices", "removed": before - len(df), "remaining": len(df)})

	before = len(df)
	df = df.loc[df["UnitPrice"] >= min_unit_price]
	print(f"Removed invalid prices: {before - len(df)} rows")
	steps.append({"step": "Invalid unit price", "removed": before - len(df), "remaining": len(df)})

	before = len(df)
	product_code_mask = df["StockCode"].astype("string").str.fullmatch(r"\d+", na=False)
	df = df.loc[product_code_mask]
	print(f"Removed non-numeric stock codes: {before - len(df)} rows")
	steps.append({"step": "Non-numeric stock codes", "removed": before - len(df), "remaining": len(df)})

	before = len(df)
	df = df.drop_duplicates()
	print(f"Removed duplicates: {before - len(df)} rows")
	steps.append({"step": "Exact duplicates", "removed": before - len(df), "remaining": len(df)})

	df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

	assert len(df) > 0, "cleaning removed everything"
	assert df["UnitPrice"].min() > 0, "non-positive prices remain"
	assert df["CustomerID"].notna().all(), "blank CustomerIDs remain"
	df.attrs["cleaning_steps"] = steps
	print(f"Cleaning: {rows_start} -> {len(df)} rows")
	return df
