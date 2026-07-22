"""Preprocessing utilities for Titanic logistic regression classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


@dataclass
class PreprocessArtifacts:
    """Container for preprocessed train/test data and fitted transformer."""

    X_train: Any
    X_test: Any
    y_train: pd.Series
    y_test: pd.Series
    transformer: ColumnTransformer
    selected_features: list[str]


def cap_outliers_iqr(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Cap numeric outliers using IQR fences."""
    adjusted = dataframe.copy()

    for column in numeric_columns:
        if column not in adjusted.columns:
            continue

        series = pd.to_numeric(adjusted[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        adjusted[column] = series.clip(lower=lower, upper=upper)

    return adjusted


def preprocess(
    dataframe: pd.DataFrame,
    target_column: str = "Survived",
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessArtifacts:
    """Prepare Titanic data for logistic regression classification."""
    features = feature_columns if feature_columns is not None else DEFAULT_FEATURES

    working = dataframe.copy()
    working.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    working.drop_duplicates(inplace=True)

    if target_column not in working.columns:
        raise KeyError(f"Target column '{target_column}' was not found")

    missing_features = [column for column in features if column not in working.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    # Outlier capping on known numeric columns requested for this pipeline.
    working = cap_outliers_iqr(working, numeric_columns=["Age", "Fare"])

    selected = working[features + [target_column]].dropna(subset=[target_column]).copy()
    X = selected[features]
    y = pd.to_numeric(selected[target_column], errors="coerce")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_features = [column for column in ["Pclass", "Age", "SibSp", "Parch", "Fare"] if column in features]
    categorical_features = [column for column in ["Sex", "Embarked"] if column in features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    X_train = transformer.fit_transform(X_train_df)
    X_test = transformer.transform(X_test_df)

    return PreprocessArtifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        transformer=transformer,
        selected_features=features,
    )
