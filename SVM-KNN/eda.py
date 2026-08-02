"""Exploratory data analysis and visualizations for the Iris dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


IMAGE_DIR = Path(__file__).parent / "images"


def summary_statistics(dataframe: pd.DataFrame) -> None:
    """Print numerical and categorical summary statistics."""
    print("\nSummary statistics:")
    print(dataframe.describe(include="all").to_string())
    print("\nClass distribution:")
    print(dataframe["species"].value_counts().to_string())


def run_eda(dataframe: pd.DataFrame) -> None:
    """Inspect distributions and relationships and save plots to images/."""
    IMAGE_DIR.mkdir(exist_ok=True)
    numeric_columns = dataframe.select_dtypes(include="number").columns

    print("\nDuplicate rows:", dataframe.duplicated().sum())
    print("\nFeature correlations:")
    print(dataframe[numeric_columns].corr().round(3).to_string())

    dataframe[numeric_columns].hist(figsize=(10, 7), bins=15, edgecolor="black")
    plt.suptitle("Iris Feature Distributions")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "eda_histograms.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(dataframe[numeric_columns].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Iris Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close()

    melted = dataframe.melt(id_vars="species", value_vars=numeric_columns,
                             var_name="feature", value_name="value")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="species")
    plt.title("Feature Distributions by Species")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "eda_boxplots.png", dpi=150)
    plt.show()
    plt.close()

    pairplot = sns.pairplot(dataframe, hue="species", diag_kind="hist", corner=True)
    pairplot.fig.suptitle("Iris Pairplot", y=1.02)
    pairplot.savefig(IMAGE_DIR / "eda_pairplot.png", dpi=150)
    plt.show()
    plt.close("all")
