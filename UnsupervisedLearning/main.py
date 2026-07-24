"""
main.py
=======
Main entry point for the Wholesale Customer Clustering project.
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on the Python path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import inspect_data, load_data
from eda import run_eda
from preprocessing import prepare_features
from model import run_clustering


def print_header() -> None:
    """Display the project header."""
    print("\n" + "=" * 70)
    print(" " * 12 + "WHOLESALE CUSTOMER CLUSTERING")
    print(" " * 10 + "K-Means and Hierarchical Clustering")
    print("=" * 70)


def main() -> None:
    """Run the full clustering workflow end to end."""
    print_header()

    print("\n[STEP 1] Loading dataset...")
    raw_df = load_data()
    inspect_data(raw_df)

    print("\n[STEP 2] Exploratory data analysis...")
    analysis_df = run_eda(raw_df)

    print("\n[STEP 3] Data preprocessing...")
    feature_frame, scaled_features, feature_columns, _ = prepare_features(analysis_df)

    print("\n[STEP 4] Clustering analysis...")
    results = run_clustering(feature_frame, scaled_features, feature_columns, k=3)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"✓ Dataset shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"✓ Features used: {', '.join(feature_columns)}")
    print(f"✓ Selected K: {results['k']}")
    print(f"✓ K-Means silhouette: {results['kmeans_silhouette']:.3f}")
    print(f"✓ Hierarchical silhouette: {results['hierarchical_silhouette']:.3f}")
    print("✓ Generated outputs:")
    print("  - images/feature_correlation_heatmap.png")
    print("  - images/elbow_method.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
