"""
main.py
=======
Main entry point for the Housing Price Prediction project.

This script orchestrates the complete machine learning pipeline:
  1. Load dataset
  2. Exploratory Data Analysis (EDA)
  3. Data Preprocessing
  4. Model Training (Linear Regression)
  5. Model Evaluation
  6. Visualizations

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
    print(" " * 15 + "HOUSING PRICE PREDICTION")
    print(" " * 10 + "Simple Linear Regression Model")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Execute the complete machine learning pipeline end-to-end.

    Pipeline flow:
        1. Load the raw housing dataset from CSV
        2. Run exploratory data analysis (EDA)
           - Descriptive statistics
           - Visualizations (histograms, scatter plots, heatmap)
           - Outlier detection using IQR method
        3. Preprocess the data
           - Drop irrelevant columns
           - Encode categorical features
           - Scale numerical features
        4. Train a Linear Regression model
           - 80/20 train/test split
           - Fit model on training data
        5. Evaluate on test data
           - Calculate metrics (MAE, MSE, RMSE, R²)
           - Generate prediction and residual plots
    """

    print_header()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading Dataset...")
    raw_df = load_data()
    print(f"✓ Dataset loaded successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    inspect_data(raw_df)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Exploratory Data Analysis (EDA)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 2] Running Exploratory Data Analysis (EDA)...")
    clean_df = run_eda(raw_df)
    print(f"✓ EDA complete. Final dataset: {clean_df.shape[0]} rows (after outlier removal)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Data Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] Preprocessing Data...")
    X, y, scaler = preprocess(clean_df)
    print(f"✓ Preprocessing complete.")
    print(f"  - Features: {X.shape[1]} columns")
    print(f"  - Samples: {X.shape[0]} rows")
    print(f"  - Target: {y.name} (min: ${y.min():,.0f}, max: ${y.max():,.0f})")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 & 5: Model Training & Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 4 & 5] Training Model & Evaluation...")
    model, metrics = run_model(X, y)
    print(f"✓ Model training and evaluation complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"✓ Dataset            : 128 rows → {clean_df.shape[0]} rows (clean)")
    print(f"✓ Features           : {X.shape[1]} (SqFt, Bedrooms, Bathrooms, Offers, Brick, Neighborhood)")
    print(f"✓ Target             : Price (${y.min():,.0f} – ${y.max():,.0f})")
    print(f"✓ Train/Test Split   : 80/20")
    print(f"✓ Model              : Linear Regression")
    print(f"✓ Best Metric        : R² = {metrics['R2']:.4f} (explains {metrics['R2']*100:.2f}% of variance)")
    print("=" * 70)

    print("\nGenerated plots:")
    print("  - eda_histograms.png")
    print("  - eda_scatter_plots.png")
    print("  - eda_correlation_heatmap.png")
    print("  - eda_boxplots.png")
    print("  - model_predictions.png")
    print("  - model_residuals.png")

    print("\n" + "=" * 70)
    print("Pipeline execution completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
