"""Exploratory analysis and visualizations for IMDb reviews."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def summarize_data(dataframe: pd.DataFrame) -> None:
    """Print sentiment balance and review-length summary statistics."""
    review_lengths = dataframe["review"].str.len()
    print("\nREVIEW SUMMARY")
    print(dataframe["sentiment"].value_counts().to_string())
    print("\nReview character lengths:")
    print(review_lengths.describe().round(1).to_string())


def plot_sentiment_distribution(dataframe: pd.DataFrame) -> None:
    """Save a bar chart showing the class distribution."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=dataframe, x="sentiment", hue="sentiment", palette="Set2", legend=False)
    plt.title("IMDb Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "sentiment_distribution.png", dpi=150)
    plt.close()


def plot_review_lengths(dataframe: pd.DataFrame) -> None:
    """Save review-length distributions by sentiment."""
    plot_data = dataframe.assign(review_length=dataframe["review"].str.len())
    plt.figure(figsize=(9, 5))
    sns.histplot(data=plot_data, x="review_length", hue="sentiment", bins=50, element="step", stat="density", common_norm=False)
    plt.title("IMDb Review Lengths")
    plt.xlabel("Characters")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "review_lengths.png", dpi=150)
    plt.close()


def run_eda(dataframe: pd.DataFrame) -> None:
    """Run summary statistics and save exploratory plots."""
    summarize_data(dataframe)
    plot_sentiment_distribution(dataframe)
    plot_review_lengths(dataframe)
