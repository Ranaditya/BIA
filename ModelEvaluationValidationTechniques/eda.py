"""
eda.py
======
Performs Exploratory Data Analysis (EDA) including:
  - Descriptive statistics
  - Data visualizations (distributions, correlations, scatter plots)
  - Outlier detection and handling (box plots, IQR method)

Steps: 2 & 3 - Data Check & EDA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


# ── Visual style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for all numeric columns.

    Covers count, mean, std, min, quartiles, and max.
    Helps understand the distribution and range of each feature.

    Args:
        df (pd.DataFrame): Raw Apple Stock DataFrame.
    """
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe().T.to_string())
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Feature Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> None:
    """
    Plot histograms for all numeric columns to visualize feature distributions.

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    n_cols = len(numeric_cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(15, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "01_distributions.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("✓ Feature distributions plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a correlation heatmap for all numeric features.

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "02_correlation_heatmap.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("✓ Correlation heatmap plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Outlier Detection (Box Plots)
# ─────────────────────────────────────────────────────────────────────────────

def plot_outliers(df: pd.DataFrame) -> None:
    """
    Plot box plots for all numeric columns to detect outliers.

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    n_cols = len(numeric_cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(15, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col])
        ax.set_title(f"Box Plot — {col}")
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "03_outlier_detection.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("✓ Outlier detection plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Missing Data Analysis
# ─────────────────────────────────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> None:
    """
    Report missing values in the dataset.

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.
    """
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values detected.")
    else:
        print(missing[missing > 0].to_string())
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EDA FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """
    Run the complete exploratory data analysis pipeline.

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    descriptive_statistics(df)
    check_missing_values(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_outliers(df)

    print("\n✓ EDA complete. Visualizations saved to 'images/' directory.")
