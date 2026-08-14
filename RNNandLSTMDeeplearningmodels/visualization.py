"""Plots for training history and chronological stock-price predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, output_dir: str | Path = "images") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_history.png"
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="Training MSE")
    plt.plot(history.history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_predictions(dates, actual, predicted, split_name: str, output_dir: str | Path = "images") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split_name.lower()}_predictions.png"
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual Close", linewidth=1.5)
    plt.plot(dates, predicted, label="Predicted Close", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.title(f"{split_name} Set: Actual vs Predicted Closing Price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_future_forecast(last_date, last_close: float, forecast: np.ndarray, output_dir: str | Path = "images") -> Path:
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "future_forecast.png"
    future_dates = pd.bdate_range(pd.Timestamp(last_date) + pd.offsets.BDay(1), periods=len(forecast))
    dates = [pd.Timestamp(last_date), *future_dates]
    values = [last_close, *forecast]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker="o", label="Recursive forecast")
    plt.axvline(pd.Timestamp(last_date), color="gray", linestyle="--", label="Forecast begins")
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.title("Illustrative Future Closing-Price Forecast")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
