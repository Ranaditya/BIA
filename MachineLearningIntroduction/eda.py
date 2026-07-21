"""
eda.py
======
Performs Exploratory Data Analysis (EDA) including:
  - Descriptive statistics
  - Data visualizations (histograms, scatter plots)
  - Outlier detection and handling (box plots, IQR method)

Steps: 4 & 5 - EDA & Outlier Detection
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
        df (pd.DataFrame): Raw housing DataFrame.
    """
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe().T.to_string())
    print("=" * 60)

    # Value counts for categorical columns
    print("\nBrick — Value Counts:")
    print(df["Brick"].value_counts().to_string())

    print("\nNeighborhood — Value Counts:")
    print(df["Neighborhood"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame) -> None:
    """
    Plot histograms for all numeric columns to visualize
    the distribution of each feature.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=15, color="steelblue", edgecolor="white")
        axes[i].set_title(f"Distribution of {col}", fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_histograms.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Histogram saved as: images/eda_histograms.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Scatter Plots (Features vs Price)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, target_col: str = "Price") -> None:
    """
    Plot scatter plots of each numeric feature against the target (Price).

    Helps identify linear or non-linear relationships between
    features and the target variable.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.
        target_col (str): Name of the target column.
    """
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].scatter(df[col], df[target_col], alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.4)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f"{col} vs {target_col}", fontsize=11)

    # Hide any unused subplots
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Feature vs {target_col} (Scatter Plots)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_scatter_plots.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Scatter plots saved as: images/eda_scatter_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a heatmap of correlations between all numeric features.

    Helps identify which features are most strongly correlated
    with Price and whether multicollinearity exists.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.
    """
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_correlation_heatmap.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Correlation heatmap saved as: images/eda_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Outlier Detection
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Plot box plots for all numeric columns to visually identify outliers.

    Points beyond the whiskers (1.5 * IQR) are potential outliers.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="red", linewidth=2))
        axes[i].set_title(f"Box Plot — {col}", fontsize=11)
        axes[i].set_ylabel(col)

    # Hide any unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Outlier Detection — Box Plots", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_boxplots.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Box plots saved as: images/eda_boxplots.png")


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    Detect outliers in numeric columns using the IQR (Interquartile Range) method.

    A data point is flagged as an outlier if it falls below
    Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.

    Returns:
        dict: Column name -> list of row indices identified as outliers.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_report = {}

    print("\n" + "=" * 60)
    print("OUTLIER DETECTION — IQR METHOD")
    print("=" * 60)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify rows that fall outside the bounds
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = df.index[outlier_mask].tolist()

        outlier_report[col] = outlier_indices
        print(f"{col:15s} | Bounds: [{lower_bound:.1f}, {upper_bound:.1f}] | Outliers: {len(outlier_indices)} {outlier_indices}")

    print("=" * 60)
    return outlier_report


def remove_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Remove rows containing outliers (IQR method) from specified columns.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.
        columns (list): Columns to check for outliers.
                        Defaults to all numeric columns if None.

    Returns:
        pd.DataFrame: DataFrame with outlier rows removed.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    clean_df = df.copy()

    for col in columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep only rows within bounds
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]

    removed = len(df) - len(clean_df)
    print(f"\nRows removed as outliers : {removed}")
    print(f"Rows remaining           : {len(clean_df)}")
    return clean_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run full EDA pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete EDA pipeline: statistics, visualizations, and outlier handling.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    descriptive_statistics(df)
    plot_histograms(df)
    plot_scatter(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    detect_outliers_iqr(df)

    # Remove outliers only from continuous numeric columns
    clean_df = remove_outliers_iqr(df, columns=["Price", "SqFt"])
    return clean_df


if __name__ == "__main__":
    import os
    # Set working directory so plots save inside the project folder
    os.chdir(r"c:\Users\vaidi\BIA\MachineLearningIntroduction")

    from data_loader import load_data
    raw_df = load_data()
    clean_df = run_eda(raw_df)
    print(f"\nFinal clean dataset shape: {clean_df.shape}")
