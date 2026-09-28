"""Clustering and customer-segment profiling utilities."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


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


def _score_or_nan(score, values, labels):
	"""Return ``score`` on non-noise points when at least two clusters exist."""
	labels = np.asarray(labels)
	cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
	usable = labels != -1
	if cluster_count < 2 or usable.sum() <= cluster_count:
		return np.nan
	return score(values[usable], labels[usable])


def compare_algorithms(scaled, n_clusters=4, random_state=42):
	"""Run KMeans, hierarchical clustering, and DBSCAN and score each result.

	Silhouette (higher is better) and Davies-Bouldin (lower is better) are
	calculated on non-noise points only.
	"""
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
		cluster_sizes = pd.Series(labels[labels != -1]).value_counts()
		rows.append({
			"algorithm": algorithm,
			"n_clusters_found": n_clusters_found,
			"noise_points": int((labels == -1).sum()),
			"largest_cluster_pct": float(cluster_sizes.max() / len(labels) * 100) if len(cluster_sizes) else np.nan,
			"silhouette": _score_or_nan(silhouette_score, values, labels),
			"davies_bouldin": _score_or_nan(davies_bouldin_score, values, labels),
		})
	columns = [
		"algorithm", "n_clusters_found", "noise_points",
		"largest_cluster_pct", "silhouette", "davies_bouldin",
	]
	return pd.DataFrame(rows, columns=columns)


def _name_segment(centroid, threshold=0.5):
	"""Assign a business label from a cluster centroid in log-scaled z-space.

	Raw RFM means are dominated by a few very large spenders, so scores are
	taken on ``log1p``-standardized values. The name combines a value tier
	(mean of Frequency and Monetary scores) with an activity status (inverse
	Recency score).
	"""
	value_score = (centroid["Frequency"] + centroid["Monetary"]) / 2
	activity_score = -centroid["Recency"]

	if value_score > threshold:
		tier = "High-Value"
	elif value_score < -threshold:
		tier = "Low-Value"
	else:
		tier = "Mid-Value"

	if activity_score > threshold:
		status = "Active"
	elif activity_score < -threshold:
		status = "Lapsing"
	else:
		status = "Cooling"
	return f"{tier} {status}"


def _unique_names(names):
	"""Append a counter to repeated segment names so each label is distinct."""
	counts = {}
	unique = []
	for name in names:
		counts[name] = counts.get(name, 0) + 1
		unique.append(name if counts[name] == 1 else f"{name} {counts[name]}")
	return unique


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
	logged = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
	z_scores = (logged - logged.mean()) / logged.std(ddof=0)
	z_scores["Cluster"] = np.asarray(labels)
	centroids = z_scores.groupby("Cluster").mean()
	names = [_name_segment(centroids.loc[cluster]) for cluster in profile.index]
	profile["Segment"] = _unique_names(names)
	return profile.reset_index()


def assess_stability(scaled, k, n_seeds=20, n_bootstrap=20, sample_frac=0.8, random_state=42):
	"""Measure KMeans label agreement across seeds and bootstrap subsamples.

	Seed stability refits with ``n_init=1`` so each run depends on a single
	random initialization. Bootstrap stability refits on random subsamples
	drawn without replacement and compares labels on the sampled customers.
	Agreement is the adjusted Rand index (ARI) against the full-data fit.

	Returns:
		Dictionary with ARI lists and summary statistics for both checks.
	"""
	values = np.asarray(scaled)
	reference = fit_kmeans(values, k, random_state)
	rng = np.random.default_rng(random_state)

	seed_ari = [
		adjusted_rand_score(reference, KMeans(n_clusters=k, random_state=seed, n_init=1).fit_predict(values))
		for seed in range(n_seeds)
	]

	bootstrap_ari = []
	sample_size = int(len(values) * sample_frac)
	for _ in range(n_bootstrap):
		index = rng.choice(len(values), size=sample_size, replace=False)
		labels = fit_kmeans(values[index], k, int(rng.integers(1_000_000)))
		bootstrap_ari.append(adjusted_rand_score(reference[index], labels))

	return {
		"k": k,
		"seed_ari": seed_ari,
		"bootstrap_ari": bootstrap_ari,
		"seed_ari_mean": float(np.mean(seed_ari)),
		"seed_ari_min": float(np.min(seed_ari)),
		"bootstrap_ari_mean": float(np.mean(bootstrap_ari)),
		"bootstrap_ari_min": float(np.min(bootstrap_ari)),
	}
