"""
model.py
========
Handles the complete modelling pipeline including:
  - Train/test split (80/20)
  - Linear Regression training
  - Model evaluation (MAE, MSE, R-squared)
  - Predicted vs Actual visualisation

Steps: 6, 7 & 8 - Split, Train, Evaluate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the feature matrix and target vector into training and test sets.

    An 80/20 split is used so the model trains on the majority of data
    while a held-out set gives an unbiased evaluation.

    Args:
        X (pd.DataFrame): Scaled feature matrix.
        y (pd.Series): Target variable (Price).
        test_size (float): Proportion of data reserved for testing. Default 0.20.
        random_state (int): Seed for reproducibility. Default 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("=" * 50)
    print("TRAIN / TEST SPLIT")
    print("=" * 50)
    print(f"Total samples  : {len(X)}")
    print(f"Training set   : {len(X_train)} samples ({100 - int(test_size * 100)}%)")
    print(f"Test set       : {len(X_test)} samples ({int(test_size * 100)}%)")
    print("=" * 50)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Train the Model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Fit a Linear Regression model on the training data.

    Linear Regression finds the best-fit hyperplane by minimising
    the sum of squared residuals between predicted and actual prices.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.

    Returns:
        LinearRegression: Trained model instance.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n" + "=" * 50)
    print("MODEL — LINEAR REGRESSION")
    print("=" * 50)
    print("Intercept :", round(model.intercept_, 2))
    print("\nCoefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"  {feature:15s}: {coef:>10.2f}")
    print("=" * 50)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Evaluate the Model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on the test set using regression metrics.

    Metrics:
        - MAE  (Mean Absolute Error)   : Average absolute difference between
                                         predicted and actual prices. Interpretable
                                         in the same unit as Price ($).
        - MSE  (Mean Squared Error)    : Penalises large errors more heavily
                                         than MAE.
        - RMSE (Root Mean Squared Error): Square root of MSE — back in $ units.
        - R²   (R-Squared)             : Proportion of variance in Price explained
                                         by the features. Closer to 1.0 is better.

    Args:
        model (LinearRegression): Trained model.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Actual test target values.

    Returns:
        dict: Dictionary of metric names to values.
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    print("\n" + "=" * 50)
    print("MODEL EVALUATION — TEST SET")
    print("=" * 50)
    print(f"  MAE  (Mean Absolute Error)       : ${mae:>10,.2f}")
    print(f"  MSE  (Mean Squared Error)        : {mse:>15,.2f}")
    print(f"  RMSE (Root Mean Squared Error)   : ${rmse:>10,.2f}")
    print(f"  R²   (R-Squared)                 : {r2:>10.4f}")
    print("=" * 50)

    return metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Predicted vs Actual Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Plot predicted vs actual house prices to visually assess model accuracy.

    A perfect model would place all points on the diagonal reference line.
    Scatter around the line indicates prediction error.

    Args:
        y_test (pd.Series): Actual house prices from the test set.
        y_pred (np.ndarray): Predicted house prices from the model.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="steelblue", edgecolors="white", linewidths=0.5)

    # Diagonal reference line — perfect predictions fall on this line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1.5, linestyle="--", label="Perfect Fit")

    plt.xlabel("Actual Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.title("Actual vs Predicted House Prices", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_predictions.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Prediction plot saved as: model_predictions.png")


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Plot residuals (errors) to check for patterns that indicate model bias.

    Residuals should be randomly scattered around zero for a well-fitted model.
    A pattern in residuals suggests the model is missing structure in the data.

    Args:
        y_test (pd.Series): Actual house prices.
        y_pred (np.ndarray): Predicted house prices.
    """
    residuals = y_test.values - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.7, color="steelblue", edgecolors="white", linewidths=0.5)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Price ($)", fontsize=12)
    plt.ylabel("Residuals ($)", fontsize=12)
    plt.title("Residuals vs Predicted Values", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_residuals.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Residuals plot saved as: model_residuals.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run full model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_model(X: pd.DataFrame, y: pd.Series) -> tuple[LinearRegression, dict]:
    """
    Orchestrate the full model pipeline: split → train → evaluate → visualise.

    Args:
        X (pd.DataFrame): Scaled feature matrix.
        y (pd.Series): Target variable (Price).

    Returns:
        tuple:
            - model (LinearRegression): Trained model.
            - metrics (dict): Evaluation metrics.
    """
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    return model, metrics


if __name__ == "__main__":
    import os
    os.chdir(r"c:\Users\vaidi\BIA\MachineLearningIntroduction")

    from data_loader import load_data
    from eda import run_eda
    from preprocessing import preprocess

    raw_df = load_data()
    clean_df = run_eda(raw_df)
    X, y, scaler = preprocess(clean_df)
    run_model(X, y)
