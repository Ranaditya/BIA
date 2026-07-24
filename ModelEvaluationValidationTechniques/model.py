"""
model.py
========
Handles the complete modeling pipeline including:
  - Hyperparameter tuning with GridSearchCV
  - Cross-validation with K-Fold
  - Model training with best hyperparameters
  - Model evaluation (Accuracy, Precision, Recall, F1-Score)
  - ROC Curve and AUC analysis
  - Classification Report

Steps: 5, 6, 7, 8, 9, 10, 11 - Hyperparameter Tuning, CV, Training, Predictions, Metrics, ROC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Hyperparameter Tuning with GridSearchCV
# ─────────────────────────────────────────────────────────────────────────────

def hyperparameter_tuning(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[RandomForestClassifier, dict]:
    """
    Perform hyperparameter tuning using GridSearchCV to find the best
    Random Forest Classifier parameters.

    Grid searches over:
      - n_estimators: [50, 100, 200]
      - max_depth: [5, 10, 20, None]
      - min_samples_split: [2, 5, 10]
      - min_samples_leaf: [1, 2, 4]
      - max_features: ['sqrt', 'log2']

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.

    Returns:
        tuple: (best_model, best_params_dict)
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1
    )

    print("Starting GridSearchCV... (this may take a moment)\n")
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS FOUND:")
    print("=" * 70)
    for param, value in best_params.items():
        print(f"  {param:25s}: {value}")
    print(f"  Best CV Score (F1-Weighted)   : {grid_search.best_score_:.4f}")
    print("=" * 70)

    return best_model, best_params


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Cross-Validation with K-Fold
# ─────────────────────────────────────────────────────────────────────────────

def cross_validation_kfold(
    model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series, k: int = 5
) -> None:
    """
    Perform K-Fold Cross-Validation to assess model stability and performance.

    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.
        k (int): Number of folds. Default 5.
    """
    print("\n" + "=" * 70)
    print(f" " * 20 + "K-FOLD CROSS-VALIDATION (k={k})")
    print("=" * 70)

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Evaluate with different metrics
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="precision_weighted")
    recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="recall_weighted")
    f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="f1_weighted")
    roc_auc_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="roc_auc")

    print(f"\nAccuracy   - Mean: {accuracy_scores.mean():.4f} | Std: {accuracy_scores.std():.4f}")
    print(f"Precision  - Mean: {precision_scores.mean():.4f} | Std: {precision_scores.std():.4f}")
    print(f"Recall     - Mean: {recall_scores.mean():.4f} | Std: {recall_scores.std():.4f}")
    print(f"F1-Score   - Mean: {f1_scores.mean():.4f} | Std: {f1_scores.std():.4f}")
    print(f"ROC-AUC    - Mean: {roc_auc_scores.mean():.4f} | Std: {roc_auc_scores.std():.4f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Train Final Model
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train the final Random Forest Classifier model using best hyperparameters
    on the complete training dataset.

    Args:
        model (RandomForestClassifier): Model with best hyperparameters.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.

    Returns:
        RandomForestClassifier: Trained model.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 70)

    model.fit(X_train, y_train)

    print("✓ Model training complete.")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(X_train.columns)}")
    print("=" * 70)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Make Predictions
# ─────────────────────────────────────────────────────────────────────────────

def make_predictions(
    model: RandomForestClassifier, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make class predictions and probability predictions on test data.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test feature matrix.

    Returns:
        tuple: (class_predictions, probability_predictions)
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "MAKING PREDICTIONS ON TEST SET")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    print(f"✓ Predictions generated for {len(X_test)} test samples.")
    print("=" * 70)

    return y_pred, y_pred_proba


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Calculate Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics for the classification model.

    Metrics include:
      - Accuracy
      - Precision
      - Recall
      - F1-Score
      - ROC-AUC

    Args:
        y_test (pd.Series): True target values.
        y_pred (np.ndarray): Predicted class labels.
        y_pred_proba (np.ndarray): Predicted probabilities for class 1.

    Returns:
        dict: Dictionary of metric names and values.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "MODEL EVALUATION METRICS")
    print("=" * 70)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    precision_increase = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Set Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision (weighted) : {precision:.4f}")
    print(f"Precision (Increase) : {precision_increase:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-Score          : {f1:.4f}")
    print(f"Test Set ROC-AUC  : {roc_auc:.4f}")
    print("=" * 70)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Classification Report
# ─────────────────────────────────────────────────────────────────────────────

def print_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print detailed classification report including per-class metrics.

    Args:
        y_test (pd.Series): True target values.
        y_pred (np.ndarray): Predicted class labels.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(
        y_test, y_pred, target_names=["Price Decrease (0)", "Price Increase (1)"]
    )
    print(report)
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        y_test (pd.Series): True target values.
        y_pred (np.ndarray): Predicted class labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"False Positives      : {fp}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_xticklabels(["Decrease", "Increase"])
    ax.set_yticklabels(["Decrease", "Increase"])

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "04_confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("✓ Confusion matrix plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ROC Curve and AUC
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_test: pd.Series, y_pred_proba: np.ndarray) -> None:
    """
    Plot and save ROC curve showing true positive rate vs false positive rate.

    Args:
        y_test (pd.Series): True target values.
        y_pred_proba (np.ndarray): Predicted probabilities for class 1.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Random Forest Classifier", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "05_roc_curve.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("✓ ROC curve plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model: RandomForestClassifier, feature_names: list) -> None:
    """
    Plot and save feature importance from the Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top 10 Feature Importance — Random Forest", fontsize=14)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "06_feature_importance.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("✓ Feature importance plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODELING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict:
    """
    Run the complete modeling pipeline: hyperparameter tuning, training,
    predictions, and evaluation.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.

    Returns:
        dict: Dictionary containing metrics and model information.
    """
    print("\n" + "=" * 70)
    print(" " * 10 + "RANDOM FOREST CLASSIFIER — COMPLETE PIPELINE")
    print("=" * 70)

    # Step 1: Hyperparameter Tuning
    best_model, best_params = hyperparameter_tuning(X_train, y_train)

    # Step 2: Cross-Validation
    cross_validation_kfold(best_model, X_train, y_train, k=5)

    # Step 3: Train Final Model
    final_model = train_final_model(best_model, X_train, y_train)

    # Step 4: Make Predictions
    y_pred, y_pred_proba = make_predictions(final_model, X_test)

    # Step 5: Calculate Metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print(f"Final Model Test Accuracy: {metrics['Accuracy']:.4f} ({metrics['Accuracy'] * 100:.2f}%)")

    # Step 6: Classification Report
    print_classification_report(y_test, y_pred)

    # Step 7: Visualizations
    print("\n" + "=" * 70)
    print(" " * 15 + "GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_feature_importance(final_model, X_train.columns.tolist())

    # Summary
    print("\n✓ Model training and evaluation complete.")

    return {
        "model": final_model,
        "best_params": best_params,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }
