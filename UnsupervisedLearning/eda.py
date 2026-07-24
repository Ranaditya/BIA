"""
eda.py
======
Exploratory data analysis for the wholesale customer clustering project.
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
import seaborn as sns


IMAGE_DIR = Path(__file__).resolve().parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean analysis copy of the dataset and save a correlation plot."""
    analysis_df = df.copy()

    # Remove duplicate rows if any and report the result
    before_rows = analysis_df.shape[0]
    analysis_df = analysis_df.drop_duplicates().reset_index(drop=True)
    after_rows = analysis_df.shape[0]

    if before_rows != after_rows:
        print(f"Removed {before_rows - after_rows} duplicate rows.")

    # Keep only numeric columns for clustering analysis
    numeric_cols = analysis_df.select_dtypes(include=["number"]).columns.tolist()
    analysis_df = analysis_df[numeric_cols].copy()

    print("\n[EDA] Numeric columns retained for clustering:")
    print(analysis_df.columns.tolist())

    # Correlation heatmap
    plt.figure(figsize=(10, 7))
    corr = analysis_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_correlation_heatmap.png", dpi=200)
    try:
        plt.show()
    except Exception:
        print("Interactive display unavailable; saved plot to images/feature_correlation_heatmap.png")
    plt.close()

    print(f"✓ Saved correlation heatmap to {IMAGE_DIR / 'feature_correlation_heatmap.png'}")
    return analysis_df
