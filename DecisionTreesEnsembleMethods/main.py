"""
main.py
=======
Main entry point for the Iris Species Prediction project.

Current implementation:
    1. Import libraries
    2. Load dataset
    3. EDA visualizations and IQR outlier handling
    4. Feature/target selection and train-test split
    5. Decision Tree and Random Forest training/evaluation
"""

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data, inspect_data
from eda import run_eda
from preprocessing import prepare_data
from model import run_models


def print_header() -> None:
    """Display project header."""
    print("\n" + "=" * 70)
    print(" " * 16 + "IRIS SPECIES PREDICTION")
    print(" " * 11 + "Decision Tree Classification")
    print("=" * 70)


def main() -> None:
    """Run the current implemented pipeline steps."""
    print_header()

    print("\n[STEP 1] Importing Libraries and Loading Dataset...")
    iris_df = load_data(source="csv")
    print(f"Dataset loaded successfully: {iris_df.shape[0]} rows, {iris_df.shape[1]} columns")
    inspect_data(iris_df)

    print("\n[STEP 2] EDA: Pairplot, Boxplots, and IQR Outlier Handling...")
    clean_df = run_eda(iris_df)
    print(f"EDA complete. Clean dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")

    print("\n[STEP 3] Feature Selection and Train/Test Split...")
    X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data(clean_df)

    print("\n[STEP 4] Train and Evaluate Decision Tree + Random Forest...")
    results = run_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        class_names,
        use_random_forest=True,
    )

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Original rows        : {len(iris_df)}")
    print(f"Rows after IQR clean : {len(clean_df)}")
    print(f"Decision Tree Acc    : {results['decision_tree']['accuracy']:.4f}")
    if "random_forest" in results:
        print(f"Random Forest Acc    : {results['random_forest']['accuracy']:.4f}")
    print("Generated model plot : images/model_decision_tree.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
