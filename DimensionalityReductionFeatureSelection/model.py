"""Random Forest training, evaluation, and model visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


IMAGES_DIR = Path(__file__).resolve().parent / "images"
RANDOM_STATE = 42


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fit a reproducible Random Forest classifier."""
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_classifier(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
    experiment_name: str,
    image_stem: str,
) -> dict:
    """Evaluate a classifier and save its confusion matrix."""
    predictions = model.predict(X_test)
    labels = sorted(y_test.unique())
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    print(f"\n{experiment_name}")
    print("-" * len(experiment_name))
    print(f"Accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    output = IMAGES_DIR / f"{image_stem}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"{experiment_name}\nConfusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
        "image": output,
    }


def plot_pca_projection(
    X_train_pca: np.ndarray,
    y_train: pd.Series,
    explained_variance_ratio: np.ndarray,
) -> Path:
    """Plot the two-dimensional PCA representation of training observations."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    output = IMAGES_DIR / "pca_projection.png"
    plot_df = pd.DataFrame(
        {
            "PC1": X_train_pca[:, 0],
            "PC2": X_train_pca[:, 1],
            "species": y_train.to_numpy(),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="species",
        style="species",
        s=70,
        alpha=0.85,
        ax=ax,
    )
    pc1 = explained_variance_ratio[0] * 100
    pc2 = explained_variance_ratio[1] * 100
    ax.set_xlabel(f"Principal Component 1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({pc2:.1f}% variance)")
    ax.set_title("PCA Projection of Iris Training Data", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> Path:
    """Save Random Forest importances for the engineered-feature experiment."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    output = IMAGES_DIR / "engineered_feature_importance.png"
    importance = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    importance.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest Feature Importance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output
