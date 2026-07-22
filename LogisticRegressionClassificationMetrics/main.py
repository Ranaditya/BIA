"""End-to-end Titanic classification pipeline using Logistic Regression."""

from __future__ import annotations

from data_loader import inspect_data, load_data
from eda import run_eda
from model import run_model
from preprocessing import preprocess


DATASET_PATH = "data/titanic.csv"
TARGET_COLUMN = "Survived"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def print_step_header(step_number: int, title: str) -> None:
    """Print a clear section header for each pipeline step."""
    print("\n" + "=" * 70)
    print(f"[STEP {step_number}] {title}")
    print("=" * 70)


def main() -> None:
    """Run the complete logistic regression classification workflow."""
    print_step_header(1, "Import Essentials and Load Dataset")
    data = load_data(DATASET_PATH)
    inspect_data(data, target_column=TARGET_COLUMN)

    print_step_header(2, "Exploration and Visualization")
    run_eda(data, target_column=TARGET_COLUMN)

    print_step_header(3, "Preprocessing and Train/Test Split")
    artifacts = preprocess(
        data,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print("Selected features:", artifacts.selected_features)
    print("Train samples:", artifacts.X_train.shape[0])
    print("Test samples :", artifacts.X_test.shape[0])

    print_step_header(4, "Train Logistic Regression Model")
    results = run_model(
        artifacts.X_train,
        artifacts.X_test,
        artifacts.y_train,
        artifacts.y_test,
        random_state=RANDOM_STATE,
    )

    print_step_header(5, "Evaluate Model")
    print(f"Accuracy : {results.metrics['accuracy']:.4f}")
    print(f"Precision: {results.metrics['precision']:.4f}")
    print(f"Recall   : {results.metrics['recall']:.4f}")
    print(f"F1 Score : {results.metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(results.confusion)

    print("\nClassification Report:")
    print(results.report)


if __name__ == "__main__":
    main()
