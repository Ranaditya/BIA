"""Exploratory visualization for the selected rainfall time series."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


IMAGES_DIR = Path(__file__).parent / "images"


def plot_time_series(series: pd.Series) -> Path:
    """Plot monthly rainfall and its 12-month rolling mean."""
    IMAGES_DIR.mkdir(exist_ok=True)
    output = IMAGES_DIR / "rainfall_time_series.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series, color="steelblue", linewidth=1, label="Monthly rainfall")
    ax.plot(
        series.index,
        series.rolling(12, min_periods=1).mean(),
        color="darkorange",
        linewidth=2,
        label="12-month rolling mean",
    )
    ax.set(title="Monthly Rainfall at Gallicano", xlabel="Date", ylabel="Rainfall (mm)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"EDA plot saved to: {output.relative_to(Path(__file__).parent)}")
    return output


def run_eda(series: pd.Series) -> Path:
    """Print descriptive statistics and create the time-series plot."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(series.describe().to_string())
    return plot_time_series(series)
