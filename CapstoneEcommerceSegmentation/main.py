"""Run the e-commerce customer segmentation feature pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from clustering import (
	assess_stability,
	compare_algorithms,
	find_optimal_k,
	fit_kmeans,
	profile_segments,
)
from classifier import train_segment_classifier
from data_loader import describe_raw, load_raw
from eda import (
	plot_country_breakdown,
	plot_rfm_correlations,
	plot_rfm_distributions,
	plot_sales_over_time,
	plot_top_products,
)
from features import build_rfm, scale_rfm
from preprocessing import clean_transactions
from evaluation import evaluate_classifier, get_feature_importance
from visualization import (
	plot_confusion_matrix,
	plot_customer_clusters,
	plot_k_selection,
	plot_segment_profiles,
)


def main():
	"""Load, clean, transform, and visualize the transaction data."""
	raw = load_raw("data/data.csv")
	describe_raw(raw)
	clean = clean_transactions(raw)
	rfm = build_rfm(clean)
	scaled, scaler = scale_rfm(rfm)
	print(rfm.describe())

	plot_rfm_distributions(rfm, save_path="outputs/figures/rfm_distributions.png")
	plot_country_breakdown(clean, top_n=10, save_path="outputs/figures/country_breakdown.png")
	plot_sales_over_time(clean, save_path="outputs/figures/sales_over_time.png")
	plot_rfm_correlations(rfm, save_path="outputs/figures/rfm_correlations.png")
	plot_top_products(clean, top_n=15, save_path="outputs/figures/top_products.png")

	optimal_k = find_optimal_k(scaled)
	plot_k_selection(optimal_k, save_path="outputs/figures/k_selection.png")
	print("K selection results:")
	print(optimal_k.to_string(index=False))
	selected_k = int(optimal_k.loc[optimal_k["silhouette"].idxmax(), "k"])
	print(f"Selected k: {selected_k}")

	comparison = compare_algorithms(scaled, n_clusters=selected_k)
	print("Algorithm comparison:")
	print(comparison.to_string(index=False))

	labels = fit_kmeans(scaled, k=selected_k)
	segment_profile = profile_segments(rfm, labels)
	plot_customer_clusters(scaled, labels, save_path="outputs/figures/customer_clusters.png")
	plot_segment_profiles(segment_profile, save_path="outputs/figures/segment_profiles.png")
	print("Segment profile:")
	print(segment_profile.to_string(index=False))

	scaled_centroids = scaled.groupby(labels).mean()
	print("Cluster centroids (scaled):")
	print(scaled_centroids.to_string())

	alternative_k = 4
	alternative_labels = fit_kmeans(scaled, k=alternative_k)
	alternative_profile = profile_segments(rfm, alternative_labels)
	alternative_comparison = compare_algorithms(scaled, n_clusters=alternative_k)
	print(f"Alternative segment profile (k={alternative_k}):")
	print(alternative_profile.to_string(index=False))
	print(f"Algorithm comparison (k={alternative_k}):")
	print(alternative_comparison.to_string(index=False))

	stability = assess_stability(scaled, selected_k)
	print(
		f"Stability ARI: seeds mean {stability['seed_ari_mean']:.3f} "
		f"(min {stability['seed_ari_min']:.3f}); bootstrap mean "
		f"{stability['bootstrap_ari_mean']:.3f} (min {stability['bootstrap_ari_min']:.3f})"
	)

	model, X_test, y_test, metrics = train_segment_classifier(scaled, labels)
	y_pred = model.predict(X_test)
	evaluation = evaluate_classifier(y_test, y_pred)
	print("Segment classifier metrics:")
	print(f"Accuracy: {metrics['accuracy']:.3f}")
	print(f"Precision: {metrics['precision']:.3f}")
	print(f"Recall: {metrics['recall']:.3f}")
	print(f"F1: {metrics['f1']:.3f}")
	print("Classification report:")
	print(pd.DataFrame(evaluation["classification_report"]).transpose().to_string())
	print("Confusion matrix:")
	print(evaluation["confusion_matrix"])
	class_labels = sorted(set(y_test) | set(y_pred))
	plot_confusion_matrix(
		evaluation["confusion_matrix"],
		class_labels=class_labels,
		save_path="outputs/figures/confusion_matrix.png",
	)

	importance = get_feature_importance(model, rfm.columns)
	print("Feature importance:")
	print(importance.to_string(index=False))

	snapshot_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
	sil_score = comparison.loc[
		comparison["algorithm"] == "KMeans", "silhouette"
	].iloc[0]
	top_feature_name = importance.iloc[0]["feature"]
	signature = {
		"sklearn_version": sklearn.__version__,
		"raw_rows": len(raw),
		"clean_rows": len(clean),
		"n_customers": len(rfm),
		"snapshot_date": str(snapshot_date),
		"n_clusters": int(len(set(labels))),
		"cluster_sizes": pd.Series(labels).value_counts().sort_index().tolist(),
		"silhouette": round(float(sil_score), 4),
		"classifier_accuracy": round(float(metrics["accuracy"]), 4),
		"top_feature": str(top_feature_name),
	}
	signature_path = Path(__file__).resolve().parent / "outputs" / "run_signature.json"
	signature_path.parent.mkdir(parents=True, exist_ok=True)
	with signature_path.open("w", encoding="utf-8") as file:
		json.dump(signature, file, indent=2)
	print(json.dumps(signature, indent=2))

	analysis_results = {
		"cleaning_steps": clean.attrs.get("cleaning_steps", []),
		"selected_k": selected_k,
		"algorithm_comparison": comparison.to_dict(orient="records"),
		"segment_profile": segment_profile.to_dict(orient="records"),
		"scaled_centroids": {
			str(cluster): row.to_dict() for cluster, row in scaled_centroids.iterrows()
		},
		"alternative_k": alternative_k,
		"alternative_profile": alternative_profile.to_dict(orient="records"),
		"alternative_comparison": alternative_comparison.to_dict(orient="records"),
		"stability": stability,
		"classifier_metrics": {
			key: float(metrics[key]) for key in ["accuracy", "precision", "recall", "f1"]
		},
		"feature_importance": importance.to_dict(orient="records"),
	}
	analysis_path = Path(__file__).resolve().parent / "outputs" / "analysis_results.json"
	with analysis_path.open("w", encoding="utf-8") as file:
		json.dump(analysis_results, file, indent=2, default=float)

	monthly = clean.groupby(clean["InvoiceDate"].dt.to_period("M")).agg(
		revenue=("TotalPrice", "sum"),
		orders=("InvoiceNo", "nunique"),
	)
	country = clean.groupby("Country").agg(
		transactions=("InvoiceNo", "size"),
		revenue=("TotalPrice", "sum"),
	).sort_values("transactions", ascending=False).head(15)
	spend_counts, spend_edges = np.histogram(rfm["Monetary"], bins=30)
	cluster_points = scaled.copy()
	cluster_points["cluster"] = labels
	cluster_points = cluster_points.sample(n=min(1200, len(cluster_points)), random_state=42)
	confusion_labels = [str(label) for label in class_labels]
	dashboard_data = {
		"signature": signature,
		"monthly": {
			"labels": [str(period) for period in monthly.index],
			"revenue": monthly["revenue"].tolist(),
			"orders": monthly["orders"].tolist(),
		},
		"countries": {
			"labels": country.index.tolist(),
			"transactions": country["transactions"].tolist(),
			"revenue": country["revenue"].tolist(),
		},
		"spending": {
			"labels": [round(float((spend_edges[i] + spend_edges[i + 1]) / 2), 2) for i in range(len(spend_counts))],
			"counts": spend_counts.tolist(),
		},
		"correlations": rfm[["Recency", "Frequency", "Monetary"]].corr().to_dict(),
		"products": {
			"labels": clean.groupby("Description")["TotalPrice"].sum().nlargest(15).index.tolist(),
			"revenue": clean.groupby("Description")["TotalPrice"].sum().nlargest(15).tolist(),
		},
		"cluster_points": [
			{"x": float(row.Recency), "y": float(row.Frequency), "cluster": int(row.cluster)}
			for row in cluster_points.itertuples()
		],
		"segments": segment_profile.to_dict(orient="records"),
		"k_selection": optimal_k.to_dict(orient="records"),
		"confusion": {
			"labels": confusion_labels,
			"matrix": evaluation["confusion_matrix"].tolist(),
		},
	}
	dashboard_data_path = Path(__file__).resolve().parent / "outputs" / "dashboard_data.json"
	dashboard_data_path.parent.mkdir(parents=True, exist_ok=True)
	with dashboard_data_path.open("w", encoding="utf-8") as file:
		json.dump(dashboard_data, file, indent=2)

	return scaled, scaler


if __name__ == "__main__":
	main()
