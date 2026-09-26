"""Visualizations for clustering and segment-classification results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig, save_path):
	"""Save a figure relative to the project directory when requested."""
	if save_path is None:
		return
	target = Path(save_path)
	if not target.is_absolute():
		target = Path(__file__).resolve().parent / target
	target.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(target, bbox_inches="tight")


def plot_k_selection(results, save_path=None):
	"""Plot KMeans inertia and silhouette scores for candidate k values."""
	required = {"k", "inertia", "silhouette"}
	missing = required - set(results.columns)
	if missing:
		raise ValueError(f"results is missing required columns: {sorted(missing)}")

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].plot(results["k"], results["inertia"], marker="o", color="#4c72b0", label="KMeans inertia")
	axes[0].set_title("KMeans Elbow Curve")
	axes[0].set_xlabel("Number of clusters (k)")
	axes[0].set_ylabel("Inertia (squared distance, scaled-feature units)")
	axes[0].legend(title="Measure")
	axes[1].plot(results["k"], results["silhouette"], marker="o", color="#dd8452", label="Silhouette score")
	axes[1].set_title("Silhouette Scores")
	axes[1].set_xlabel("Number of clusters (k)")
	axes[1].set_ylabel("Silhouette score (unitless)")
	axes[1].legend(title="Measure")
	fig.tight_layout()
	_save_figure(fig, save_path)
	plt.show()
	plt.close(fig)


def plot_customer_clusters(scaled, labels, save_path=None):
	"""Plot customers by the first two scaled RFM dimensions and cluster label."""
	values = np.asarray(scaled)
	if values.ndim != 2 or values.shape[1] < 2:
		raise ValueError("scaled must contain at least two feature columns")
	if len(values) != len(labels):
		raise ValueError("scaled and labels must contain the same number of rows")

	fig, ax = plt.subplots(figsize=(9, 6))
	for cluster in sorted(set(labels)):
		mask = np.asarray(labels) == cluster
		ax.scatter(values[mask, 0], values[mask, 1], s=18, alpha=0.65, label=f"Cluster {cluster}")
	ax.set_title("Customer Clusters in Scaled RFM Space")
	columns = list(scaled.columns) if isinstance(scaled, pd.DataFrame) else []
	x_name = columns[0] if columns else "Feature 1"
	y_name = columns[1] if len(columns) > 1 else "Feature 2"
	ax.set_xlabel(f"{x_name} (standardized z-score)")
	ax.set_ylabel(f"{y_name} (standardized z-score)")
	ax.legend(title="Cluster")
	fig.tight_layout()
	_save_figure(fig, save_path)
	plt.show()
	plt.close(fig)


def plot_segment_profiles(profile, save_path=None):
	"""Plot mean Recency, Frequency, and Monetary values by segment."""
	if "Segment" not in profile.columns:
		raise ValueError("profile must contain a Segment column")
	metrics = ["Recency", "Frequency", "Monetary"]
	missing = set(metrics) - set(profile.columns)
	if missing:
		raise ValueError(f"profile is missing required columns: {sorted(missing)}")

	plot_data = profile.set_index("Segment")[metrics].copy()
	plot_data["Recency"] = (plot_data["Recency"] - plot_data["Recency"].mean()) / plot_data["Recency"].std(ddof=0)
	plot_data["Frequency"] = (plot_data["Frequency"] - plot_data["Frequency"].mean()) / plot_data["Frequency"].std(ddof=0)
	plot_data["Monetary"] = (plot_data["Monetary"] - plot_data["Monetary"].mean()) / plot_data["Monetary"].std(ddof=0)
	plot_data = plot_data.fillna(0)
	fig, ax = plt.subplots(figsize=(11, 6))
	plot_data.plot(kind="bar", ax=ax)
	ax.set_title("Average RFM Profile by Customer Segment")
	ax.set_xlabel("Segment")
	ax.set_ylabel("Mean RFM value (standardized z-score)")
	plt.xticks(rotation=30, ha="right")
	ax.legend(title="Metric (standardized)")
	fig.tight_layout()
	_save_figure(fig, save_path)
	plt.show()
	plt.close(fig)


def plot_confusion_matrix(confusion, class_labels=None, save_path=None):
	"""Plot a classifier confusion matrix."""
	matrix = np.asarray(confusion)
	if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
		raise ValueError("confusion must be a square matrix")
	if class_labels is None:
		class_labels = [str(index) for index in range(matrix.shape[0])]
	if len(class_labels) != matrix.shape[0]:
		raise ValueError("class_labels must match the confusion matrix size")

	fig, ax = plt.subplots(figsize=(7, 6))
	image = ax.imshow(matrix, cmap="Blues")
	fig.colorbar(image, ax=ax, label="Customer count")
	for row in range(matrix.shape[0]):
		for column in range(matrix.shape[1]):
			ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
	ax.set_xticks(range(len(class_labels)), class_labels, rotation=30, ha="right")
	ax.set_yticks(range(len(class_labels)), class_labels)
	ax.set_xlabel("Predicted segment")
	ax.set_ylabel("Actual segment")
	ax.set_title("Segment Classifier Confusion Matrix")
	fig.tight_layout()
	_save_figure(fig, save_path)
	plt.show()
	plt.close(fig)
