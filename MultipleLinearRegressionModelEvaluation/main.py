"""Multiple Linear Regression pipeline for car mileage prediction.

This script demonstrates the complete workflow step by step:
1. Import libraries
2. Load the dataset
3. Clean and preprocess the data
4. Visualize key relationships
5. Split into train and test sets
6. Train a multiple linear regression model
7. Evaluate the model with regression metrics

Run this file directly from the project folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "3304db2c078848f8ad85537da4d87645_car_(1).csv"
PLOT_DIR = PROJECT_DIR / "images"
TARGET_COLUMN = "mpg"


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the car dataset from disk."""
    return pd.read_csv(file_path)


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and standardize missing values."""
    cleaned = dataframe.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()
    cleaned.replace("", np.nan, inplace=True)
    cleaned.drop_duplicates(inplace=True)
    return cleaned


def cap_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cap numeric outliers using the IQR rule."""
    adjusted = dataframe.copy()
    numeric_columns = adjusted.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        q1 = adjusted[column].quantile(0.25)
        q3 = adjusted[column].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        adjusted[column] = adjusted[column].clip(lower=lower_bound, upper=upper_bound)

    return adjusted


def create_visualizations(dataframe: pd.DataFrame) -> None:
    """Create basic EDA plots for the dataset."""
    PLOT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=dataframe, x="origin")
    plt.title("Car Count by Origin")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "countplot_origin.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=dataframe, x=TARGET_COLUMN, kde=True)
    plt.title("Distribution of MPG")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "distplot_mpg.png", dpi=150)
    plt.close()

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe[numeric_columns].corr(), annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


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


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print standard regression metrics."""
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")


def main() -> None:
    """Run the complete modeling workflow."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = load_data(DATA_PATH)
    data = clean_data(data)

    print("Outlier summary (before capping):")
    for column in data.select_dtypes(include=[np.number]).columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
        print(f"{column}: {outlier_count}")

    data = cap_outliers(data)

    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found in the dataset")

    print("Dataset shape:", data.shape)
    print("Missing values per column:\n", data.isna().sum())
    print("Duplicate rows:", data.duplicated().sum())

    create_visualizations(data)

    model, features, target = build_model(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model.fit(x_train, y_train)
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()