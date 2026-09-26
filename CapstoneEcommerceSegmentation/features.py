"""Feature engineering utilities for customer segmentation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_rfm(df, snapshot_offset_days=1):
	"""Build a customer-level Recency, Frequency, Monetary table.

	Args:
		df: Cleaned transaction DataFrame containing CustomerID, InvoiceDate,
			InvoiceNo, and TotalPrice columns.
		snapshot_offset_days: Days added to the latest transaction date for
			the analysis snapshot.

	Returns:
		DataFrame indexed by CustomerID with Recency, Frequency, and Monetary.
	"""
	snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=snapshot_offset_days)
	print(f"Snapshot date: {snapshot}")

	rfm = (
		df.groupby("CustomerID")
		.agg(
			LastPurchaseDate=("InvoiceDate", "max"),
			Frequency=("InvoiceNo", "nunique"),
			Monetary=("TotalPrice", "sum"),
		)
	)
	rfm["Recency"] = (snapshot - rfm.pop("LastPurchaseDate")).dt.days
	rfm = rfm[["Recency", "Frequency", "Monetary"]]

	assert rfm["Recency"].min() >= 0, "negative recency — snapshot date wrong"
	assert rfm["Frequency"].min() >= 1, "customer with zero orders"
	assert rfm["Monetary"].notna().all(), "nulls in Monetary"
	assert rfm.index.is_unique, "duplicate CustomerIDs"
	return rfm


def scale_rfm(rfm, log_transform=True):
	"""Optionally log-transform and standardize RFM features.

	Args:
		rfm: DataFrame containing Recency, Frequency, and Monetary columns.
		log_transform: Apply ``np.log1p`` before standardization when true.

	Returns:
		A tuple containing the scaled DataFrame and fitted StandardScaler.
	"""
	features = rfm[["Recency", "Frequency", "Monetary"]].copy()
	if log_transform:
		features = np.log1p(features)

	scaler = StandardScaler()
	scaled = pd.DataFrame(
		scaler.fit_transform(features),
		index=rfm.index,
		columns=features.columns,
	)
	return scaled, scaler
