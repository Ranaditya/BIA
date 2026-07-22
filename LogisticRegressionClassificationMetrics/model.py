"""Model training and evaluation for Titanic logistic regression classification."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ModelResults:
    """Container for trained model and computed metrics."""

    model: LogisticRegression
    metrics: dict[str, float]
    confusion: pd.DataFrame
    report: str


def train_model(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test, y_test) -> ModelResults:
    """Evaluate classification performance with core metrics."""
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    confusion = confusion_matrix(y_test, predictions)
    confusion_df = pd.DataFrame(
        confusion,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    )

    report = classification_report(y_test, predictions, zero_division=0)

    return ModelResults(model=model, metrics=metrics, confusion=confusion_df, report=report)


def run_model(X_train, X_test, y_train, y_test, random_state: int = 42) -> ModelResults:
    """Run full model lifecycle: train and evaluate."""
    model = train_model(X_train, y_train, random_state=random_state)
    return evaluate_model(model, X_test, y_test)
