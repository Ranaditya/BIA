"""Command-line entry point for the stock-price RNN/LSTM pipeline."""

import argparse
import random
from pathlib import Path

import numpy as np

from data_loader import inspect_data, load_data
from model import (
    actual_prices,
    build_model,
    forecast_future,
    predict_prices,
    regression_metrics,
    train_model,
)
from preprocessing import FEATURE_COLUMNS, preprocess_data
from visualization import plot_future_forecast, plot_predictions, plot_training_history


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RNN or LSTM stock-price regressor.")
    parser.add_argument("--model", choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--future-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name} metrics")
    print("-" * 40)
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R2   : {metrics['R2']:.4f}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"STOCK PRICE PREDICTION - {args.model.upper()}")
    print("=" * 70)
    raw = load_data(ROOT / "data" / "Stock_price.csv")
    inspect_data(raw)
    data = preprocess_data(raw, sequence_length=args.sequence_length)
    print("\nChronological sequence split")
    print(f"Training   : {len(data.X_train)} sequences")
    print(f"Validation : {len(data.X_val)} sequences")
    print(f"Test       : {len(data.X_test)} sequences")
    print(f"Input shape: ({args.sequence_length}, {len(FEATURE_COLUMNS)})")

    model = build_model(
        args.model,
        input_shape=(args.sequence_length, len(FEATURE_COLUMNS)),
        units=args.units,
        learning_rate=args.learning_rate,
    )
    model.summary()
    print("\nHyperparameters")
    print(vars(args))
    history = train_model(
        model, data, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, model_path=ROOT / "models" / f"best_{args.model}.keras",
    )

    val_actual = actual_prices(data.y_val, data)
    val_predicted = predict_prices(model, data.X_val, data)
    test_actual = actual_prices(data.y_test, data)
    test_predicted = predict_prices(model, data.X_test, data)
    print_metrics("Validation", regression_metrics(val_actual, val_predicted))
    print_metrics("Test", regression_metrics(test_actual, test_predicted))

    paths = [
        plot_training_history(history, ROOT / "images"),
        plot_predictions(data.val_dates, val_actual, val_predicted, "Validation", ROOT / "images"),
        plot_predictions(data.test_dates, test_actual, test_predicted, "Test", ROOT / "images"),
    ]
    forecast = forecast_future(model, data, args.future_days)
    paths.append(plot_future_forecast(
        data.clean_data["Date"].iloc[-1], data.clean_data["Close"].iloc[-1], forecast, ROOT / "images"
    ))
    print("\nIllustrative future closing-price forecast:")
    for day, price in enumerate(forecast, start=1):
        print(f"Day {day:2d}: {price:.4f}")
    print("\nSaved plots:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")
    print("\nInterpretation: compare MAE/RMSE with the typical Close price and inspect")
    print("the chronological plot. A validation/test gap suggests overfitting or")
    print("market-regime change. Stock forecasts are uncertain and are not financial advice.")


if __name__ == "__main__":
    main()
