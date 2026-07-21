"""Multiple Linear Regression pipeline for car mileage prediction.

This script demonstrates the complete workflow step by step:
1. Import libraries
2. Load the dataset
3. Clean and preprocess the data
4. Visualize key relationships
5. Split into train and test sets
6. Train a multiple linear regression model
7. Evaluate the model with regression metrics

Run this file directly from the project folder.
"""

from __future__ import annotations

from data_loader import load_data, inspect_data
from eda import run_eda
from model import run_model
from preprocessing import preprocess


TARGET_COLUMN = "mpg"


def print_step_header(step_number: int, title: str) -> None:
    """Print a clear section header for each pipeline step."""
    print("\n" + "=" * 70)
    print(f"[STEP {step_number}] {title}")
    print("=" * 70)


def main() -> None:
    """Run the complete modeling workflow."""
    print_step_header(1, "Load and Inspect Dataset")
    data = load_data()
    inspect_data(data)

    print_step_header(2, "Exploratory Data Analysis")
    run_eda(data)

    print_step_header(3, "Data Preprocessing")
    data = preprocess(data)

    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found in the dataset")

    print("Dataset shape after preprocessing:", data.shape)

    print_step_header(4, "Model Training and Evaluation")
    _, metrics = run_model(data)

    print_step_header(5, "Pipeline Summary")
    print(f"Final MAE: {metrics['MAE']:.4f}")
    print(f"Final MSE: {metrics['MSE']:.4f}")
    print(f"Final R2 : {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()