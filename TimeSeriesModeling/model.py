"""Chronological split, ARIMA training, forecasting, and evaluation."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


IMAGES_DIR = Path(__file__).parent / "images"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
ARIMA_ORDER = (2, 0, 2)


def split_data(series: pd.Series, test_size: float = 0.20) -> tuple[pd.Series, pd.Series]:
    """Split in time order; the final 20% is held out for testing."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split_index = int(len(series) * (1 - test_size))
    train, test = series.iloc[:split_index], series.iloc[split_index:]
    print("=" * 60)
    print("CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"Training observations: {len(train)} ({train.index.min():%Y-%m} to {train.index.max():%Y-%m})")
    print(f"Testing observations:  {len(test)} ({test.index.min():%Y-%m} to {test.index.max():%Y-%m})")
    return train, test


def train_model(train: pd.Series, order: tuple[int, int, int] = ARIMA_ORDER) -> ARIMAResults:
    """Fit an ARIMA model to the training series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(
            train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
    print(f"ARIMA{order} trained successfully (AIC: {fitted.aic:.2f})")
    return fitted


def evaluate_model(model: ARIMAResults, test: pd.Series) -> tuple[dict[str, float], pd.Series]:
    """Forecast the test period and calculate MSE, RMSE, and MAE."""
    forecast = model.forecast(steps=len(test))
    predictions = pd.Series(np.asarray(forecast), index=test.index, name="Predicted")
    mse = float(mean_squared_error(test, predictions))
    metrics = {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(test, predictions)),
    }
    print("=" * 60)
    print("MODEL EVALUATION - TEST SET")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:5s}: {value:,.4f}")
    return metrics, predictions


def save_results(test: pd.Series, predictions: pd.Series, metrics: dict[str, float]) -> None:
    """Save forecasts, metrics, and an actual-versus-predicted plot."""
    IMAGES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    pd.DataFrame({"Actual": test, "Predicted": predictions}).to_csv(
        OUTPUTS_DIR / "test_predictions.csv", index_label="Date"
    )
    pd.Series(metrics, name="Value").to_csv(OUTPUTS_DIR / "evaluation_metrics.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test.index, test, marker="o", markersize=3, label="Actual", color="steelblue")
    ax.plot(
        predictions.index,
        predictions,
        marker="o",
        markersize=3,
        label="ARIMA prediction",
        color="crimson",
    )
    ax.set(
        title="Actual vs Predicted Monthly Rainfall - Test Period",
        xlabel="Date",
        ylabel="Rainfall (mm)",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_model(series: pd.Series) -> tuple[ARIMAResults, dict[str, float]]:
    """Run the complete modeling workflow."""
    train, test = split_data(series)
    model = train_model(train)
    metrics, predictions = evaluate_model(model, test)
    save_results(test, predictions, metrics)
    return model, metrics
