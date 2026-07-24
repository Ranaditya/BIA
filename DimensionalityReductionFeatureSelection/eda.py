"""Exploratory analysis and visualizations for the Iris dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


IMAGES_DIR = Path(__file__).resolve().parent / "images"
NUMERIC_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

sns.set_theme(style="whitegrid", context="notebook")


def descriptive_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics and class counts."""
    print("\nDescriptive statistics:")
    print(df[NUMERIC_COLUMNS].describe().T.round(3).to_string())
    print("\nMean measurements by species:")
    print(df.groupby("species")[NUMERIC_COLUMNS].mean().round(3).to_string())


def plot_histograms(df: pd.DataFrame) -> Path:
    """Save feature distributions split by species."""
    output = IMAGES_DIR / "eda_histograms.png"
    melted = df.melt(
        id_vars="species",
        value_vars=NUMERIC_COLUMNS,
        var_name="feature",
        value_name="measurement",
    )
    grid = sns.displot(
        data=melted,
        x="measurement",
        hue="species",
        col="feature",
        col_wrap=2,
        bins=15,
        kde=True,
        height=3.4,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Iris Feature Distributions by Species", y=1.03, fontweight="bold")
    grid.figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(grid.figure)
    return output


def plot_boxplots(df: pd.DataFrame) -> Path:
    """Save species-level box plots for each numeric feature."""
    output = IMAGES_DIR / "eda_boxplots.png"
    melted = df.melt(
        id_vars="species",
        value_vars=NUMERIC_COLUMNS,
        var_name="feature",
        value_name="measurement",
    )
    grid = sns.catplot(
        data=melted,
        x="species",
        y="measurement",
        col="feature",
        col_wrap=2,
        kind="box",
        height=3.4,
        sharey=False,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Species", "Measurement (cm)")
    grid.figure.suptitle("Iris Measurements by Species", y=1.03, fontweight="bold")
    grid.figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(grid.figure)
    return output


def plot_pairwise_relationships(df: pd.DataFrame) -> Path:
    """Save pairwise feature relationships coloured by species."""
    output = IMAGES_DIR / "eda_pairplot.png"
    grid = sns.pairplot(
        df,
        vars=NUMERIC_COLUMNS,
        hue="species",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.75, "s": 35},
    )
    grid.figure.suptitle("Pairwise Iris Feature Relationships", y=1.02, fontweight="bold")
    grid.figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(grid.figure)
    return output


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Save a correlation heatmap for the numeric features."""
    output = IMAGES_DIR / "eda_correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        df[NUMERIC_COLUMNS].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Iris Feature Correlation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output


def run_eda(df: pd.DataFrame) -> list[Path]:
    """Run the complete EDA stage and return generated image paths."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    descriptive_statistics(df)
    outputs = [
        plot_histograms(df),
        plot_boxplots(df),
        plot_pairwise_relationships(df),
        plot_correlation_heatmap(df),
    ]
    print("\nEDA visualizations:")
    for output in outputs:
        print(f"  - {output.relative_to(output.parent.parent)}")
    return outputs


if __name__ == "__main__":
    from data_loader import load_data

    run_eda(load_data())
