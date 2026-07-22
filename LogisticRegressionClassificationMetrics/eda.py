"""Exploratory data analysis utilities for Titanic classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def run_eda(dataframe: pd.DataFrame, target_column: str = "Survived") -> None:
    """Generate summary statistics and core EDA visualizations."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nDescriptive statistics (numeric):")
    print(dataframe.describe(include=["number"]))

    print("\nDescriptive statistics (categorical):")
    print(dataframe.describe(include=["object"]))

    _plot_numeric_histograms(dataframe)
    _plot_survival_by_category(dataframe, target_column)


def _plot_numeric_histograms(dataframe: pd.DataFrame) -> None:
    numeric_columns = ["Age", "Fare", "SibSp", "Parch"]
    available_columns = [column for column in numeric_columns if column in dataframe.columns]

    if not available_columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for index, column in enumerate(available_columns):
        sns.histplot(dataframe[column], kde=True, ax=axes[index], color="#2C7FB8")
        axes[index].set_title(f"Distribution of {column}")

    for index in range(len(available_columns), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "eda_numeric_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: images/eda_numeric_histograms.png")


def _plot_survival_by_category(dataframe: pd.DataFrame, target_column: str) -> None:
    if target_column not in dataframe.columns:
        return

    categorical_columns = ["Sex", "Pclass", "Embarked"]
    available_columns = [column for column in categorical_columns if column in dataframe.columns]

    if not available_columns:
        return

    fig, axes = plt.subplots(1, len(available_columns), figsize=(5 * len(available_columns), 4))
    if len(available_columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, available_columns):
        plot_data = dataframe[[column, target_column]].dropna().copy()
        rates = plot_data.groupby(column, as_index=False)[target_column].mean()
        sns.barplot(data=rates, x=column, y=target_column, ax=axis, color="#41AB5D")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Survival Rate")
        axis.set_title(f"Survival Rate by {column}")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "eda_survival_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: images/eda_survival_by_category.png")
