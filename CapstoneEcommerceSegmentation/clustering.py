"""Clustering and customer-segment profiling utilities."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(scaled, k_range=range(2, 11), random_state=42):
	"""Evaluate KMeans inertia and silhouette across candidate cluster counts.

	Args:
		scaled: Scaled RFM features, as a DataFrame or array-like object.
		k_range: Candidate values of ``k`` to evaluate.
		random_state: Seed used by KMeans.

	Returns:
		DataFrame with columns ``k``, ``inertia``, and ``silhouette``.
	"""
	values = np.asarray(scaled)
	results = []
	for k in k_range:
		if k < 2 or k >= len(values):
			continue
		model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
		labels = model.fit_predict(values)
		results.append({
			"k": k,
			"inertia": model.inertia_,
			"silhouette": silhouette_score(values, labels),
		})
	return pd.DataFrame(results, columns=["k", "inertia", "silhouette"])


def fit_kmeans(scaled, k=4, random_state=42, n_init=10):
	"""Fit KMeans and return one cluster label for each row."""
	model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
	return model.fit_predict(np.asarray(scaled))


def fit_hierarchical(scaled, n_clusters=4, linkage="ward"):
	"""Fit agglomerative clustering and return cluster labels.

	Agglomerative clustering uses O(n^2) memory. It is suitable for roughly
	4,000 customers, but larger datasets should be sampled first.
	"""
	model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
	return model.fit_predict(np.asarray(scaled))


def fit_dbscan(scaled, eps=0.5, min_samples=5):
	"""Fit DBSCAN and return labels, where ``-1`` identifies noise."""
	model = DBSCAN(eps=eps, min_samples=min_samples)
	return model.fit_predict(np.asarray(scaled))


def _silhouette_or_nan(values, labels):
	"""Return silhouette when labels contain at least two usable clusters."""
	labels = np.asarray(labels)
	cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
	usable = labels != -1
	if cluster_count < 2 or usable.sum() <= cluster_count:
		return np.nan
	return silhouette_score(values[usable], labels[usable])


def compare_algorithms(scaled, n_clusters=4, random_state=42):
	"""Run KMeans, hierarchical clustering, and DBSCAN with silhouettes."""
	values = np.asarray(scaled)
	algorithms = {
		"KMeans": fit_kmeans(values, n_clusters, random_state),
		"Hierarchical": fit_hierarchical(values, n_clusters),
		"DBSCAN": fit_dbscan(values),
	}
	rows = []
	for algorithm, labels in algorithms.items():
		cluster_labels = set(labels)
		n_clusters_found = len(cluster_labels - {-1})
		rows.append({
			"algorithm": algorithm,
			"n_clusters_found": n_clusters_found,
			"silhouette": _silhouette_or_nan(values, labels),
		})
	return pd.DataFrame(rows, columns=["algorithm", "n_clusters_found", "silhouette"])


def _name_segment(profile, overall):
	"""Assign a business label from measured cluster profile values."""
	recency_score = (overall["Recency"].mean() - profile["Recency"]) / overall["Recency"].std()
	frequency_score = (profile["Frequency"] - overall["Frequency"].mean()) / overall["Frequency"].std()
	monetary_score = (profile["Monetary"] - overall["Monetary"].mean()) / overall["Monetary"].std()

	if recency_score > 0.5 and frequency_score > 0.5 and monetary_score > 0.5:
		return "Champions"
	if recency_score > 0.5 and frequency_score < -0.5 and monetary_score < -0.5:
		return "New Customers"
	if recency_score < -0.5 and frequency_score > 0.5:
		return "At Risk"
	if recency_score < -0.5 and frequency_score < -0.5 and monetary_score < -0.5:
		return "Hibernating"
	return "Loyal"


def profile_segments(rfm, labels):
	"""Summarize cluster sizes, RFM means, revenue share, and segment names."""
	required = {"Recency", "Frequency", "Monetary"}
	missing = required - set(rfm.columns)
	if missing:
		raise ValueError(f"rfm is missing required columns: {sorted(missing)}")
	if len(rfm) != len(labels):
		raise ValueError("rfm and labels must contain the same number of rows")

	profiled = rfm[["Recency", "Frequency", "Monetary"]].copy()
	profiled["Cluster"] = np.asarray(labels)
	profile = profiled.groupby("Cluster").agg(
		Size=("Cluster", "size"),
		Recency=("Recency", "mean"),
		Frequency=("Frequency", "mean"),
		Monetary=("Monetary", "mean"),
	)
	profile["RevenuePct"] = (
		profile["Monetary"] * profile["Size"] / rfm["Monetary"].sum() * 100
	)
	overall = rfm[["Recency", "Frequency", "Monetary"]]
	profile["Segment"] = profile.apply(lambda row: _name_segment(row, overall), axis=1)
	return profile.reset_index()
