"""EDA and visualization utilities for car mileage regression."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
PLOT_DIR = PROJECT_DIR / "images"
TARGET_COLUMN = "mpg"


def descriptive_statistics(dataframe: pd.DataFrame) -> None:
    """Print summary statistics for quick understanding of the dataset."""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(dataframe.describe(include="all").transpose().to_string())
    print("=" * 60)


def print_outlier_summary(dataframe: pd.DataFrame) -> None:
    """Print outlier count per numeric column using the IQR rule."""
    print("Outlier summary (before capping):")
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)).sum()
        print(f"{column}: {outlier_count}")


def create_visualizations(dataframe: pd.DataFrame) -> None:
    """Create required EDA charts, save them, and display them."""
    PLOT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_files: list[Path] = []

    plt.figure(figsize=(8, 5))
    sns.countplot(data=dataframe, x="origin")
    plt.title("Car Count by Origin")
    plt.tight_layout()
    countplot_path = PLOT_DIR / "countplot_origin.png"
    plt.savefig(countplot_path, dpi=150)
    saved_files.append(countplot_path)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=dataframe, x=TARGET_COLUMN, kde=True)
    plt.title("Distribution of MPG")
    plt.tight_layout()
    distplot_path = PLOT_DIR / "distplot_mpg.png"
    plt.savefig(distplot_path, dpi=150)
    saved_files.append(distplot_path)
    plt.show()
    plt.close()

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe[numeric_columns].corr(), annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = PLOT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    saved_files.append(heatmap_path)
    plt.show()
    plt.close()


def run_eda(dataframe: pd.DataFrame) -> None:
    """Run EDA reporting and visualizations."""
    descriptive_statistics(dataframe)
    print_outlier_summary(dataframe)
    create_visualizations(dataframe)
