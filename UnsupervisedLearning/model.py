"""
model.py
========
Run K-Means and hierarchical clustering for the wholesale customer dataset.
"""

from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("QtAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


IMAGE_DIR = Path(__file__).resolve().parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def run_clustering(df: pd.DataFrame, scaled_features: pd.DataFrame, feature_columns: list[str], k: int = 3) -> dict[str, object]:
    """Perform K-Means and hierarchical clustering and summarize the clusters."""
    inertia_values = []
    for n_clusters in range(1, 11):
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(scaled_features)
        inertia_values.append(kmeans_model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia_values, marker="o", linestyle="-", color="royalblue")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, 11))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "elbow_method.png", dpi=200)
    try:
        plt.show()
    except Exception:
        print("Interactive display unavailable; saved plot to images/elbow_method.png")
    plt.close()

    print(f"\n[MODEL] Elbow plot saved to {IMAGE_DIR / 'elbow_method.png'}")
    print(f"[MODEL] Selected K={k} using the elbow point heuristic.")

    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans_model.fit_predict(scaled_features)

    hierarchical_model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    df["Agg_Cluster"] = hierarchical_model.fit_predict(scaled_features)

    kmeans_silhouette = silhouette_score(scaled_features, df["Cluster"])
    hierarchical_silhouette = silhouette_score(scaled_features, df["Agg_Cluster"])

    print(f"\n[MODEL] Silhouette score (K-Means): {kmeans_silhouette:.3f}")
    print(f"[MODEL] Silhouette score (Hierarchical): {hierarchical_silhouette:.3f}")

    print("\n[MODEL] K-Means cluster profile (mean spend by cluster):")
    print(df.groupby("Cluster")[feature_columns].mean().round(2))

    print("\n[MODEL] Hierarchical cluster profile (mean spend by cluster):")
    print(df.groupby("Agg_Cluster")[feature_columns].mean().round(2))

    return {
        "k": k,
        "kmeans_silhouette": kmeans_silhouette,
        "hierarchical_silhouette": hierarchical_silhouette,
        "cluster_summary": df.groupby("Cluster")[feature_columns].mean().round(2),
        "agg_cluster_summary": df.groupby("Agg_Cluster")[feature_columns].mean().round(2),
    }
