"""
main.py
=======
Main entry point for the Apple Stock Price Prediction with Random Forest Classifier.

This script orchestrates the complete machine learning pipeline:
  1. Load dataset
  2. Exploratory Data Analysis (EDA)
  3. Data Preprocessing (feature engineering, scaling, train/test split)
  4. Hyperparameter Tuning (GridSearchCV)
  5. Cross-Validation (K-Fold)
  6. Model Training (Random Forest Classifier)
  7. Model Evaluation (Metrics, ROC Curve, Classification Report)
  8. Visualizations (Confusion Matrix, Feature Importance)

To run: python main.py
"""

import os
import sys
from pathlib import Path

# Ensure we're in the correct working directory
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data, inspect_data
from eda import run_eda
from preprocessing import preprocess
from model import run_model


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def print_header() -> None:
    """Display the project header."""
    print("\n" + "=" * 70)
    print(" " * 10 + "APPLE STOCK PRICE PREDICTION")
    print(" " * 8 + "Random Forest Classifier with GridSearchCV")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Execute the complete machine learning pipeline end-to-end.
    """
    print_header()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: LOAD DATA
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: LOAD DATA")
    print("─" * 70)

    df = load_data()
    inspect_data(df)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("─" * 70)

    run_eda(df)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: DATA PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("─" * 70)

    X_train, X_test, y_train, y_test = preprocess(df)

    # ─────────────────────────────────────────────────────────────────────────
    # STEPS 4-9: MODEL TRAINING AND EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEPS 4-9: MODEL TRAINING AND EVALUATION")
    print("─" * 70)

    results = run_model(X_train, X_test, y_train, y_test)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" " * 15 + "PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print("\nKey Results:")
    print(f"  • Test Accuracy   : {results['metrics']['Accuracy']:.4f}")
    print(f"  • Test Precision  : {results['metrics']['Precision']:.4f}")
    print(f"  • Test Recall     : {results['metrics']['Recall']:.4f}")
    print(f"  • Test F1-Score   : {results['metrics']['F1-Score']:.4f}")
    print(f"  • ROC-AUC Score   : {results['metrics']['ROC-AUC']:.4f}")
    print("\nOutputs Generated:")
    print("  • Visualizations saved to 'images/' directory")
    print("  • Model: RandomForestClassifier with optimized hyperparameters")
    print(f"\nBest Hyperparameters Found:")
    for param, value in results['best_params'].items():
        print(f"  • {param}: {value}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
