"""Modeling utilities for multiple linear regression on car mileage data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "mpg"


def build_model(dataframe: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Prepare features and target and build the regression pipeline."""
    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]

    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    return model, features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute and print standard regression metrics."""
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {"MAE": mae, "MSE": mse, "R2": r2}

    print("Model Evaluation")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")

    return metrics


def run_model(dataframe: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Run model build, split, train, and evaluation in sequence."""
    model, features, target = build_model(dataframe)
    x_train, x_test, y_train, y_test = split_data(features, target)
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    return model, metrics
