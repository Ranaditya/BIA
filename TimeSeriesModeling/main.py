"""Run the complete Rainfall ARIMA time-series pipeline."""

from data_loader import inspect_data, load_data
from eda import run_eda
from model import run_model
from preprocessing import TARGET_COLUMN, preprocess


def main() -> None:
    print("\n" + "=" * 70)
    print("RAINFALL TIME-SERIES MODELING WITH ARIMA")
    print("=" * 70)

    raw = load_data()
    inspect_data(raw)
    series = preprocess(raw)
    run_eda(series)
    _, metrics = run_model(series)

    print("\nPipeline completed successfully.")
    print(f"Selected series: {TARGET_COLUMN}")
    print("Metrics:", ", ".join(f"{key}={value:,.4f}" for key, value in metrics.items()))
    print("Generated files:")
    print("  images/rainfall_time_series.png")
    print("  images/actual_vs_predicted.png")
    print("  outputs/evaluation_metrics.csv")
    print("  outputs/test_predictions.csv")


if __name__ == "__main__":
    main()
