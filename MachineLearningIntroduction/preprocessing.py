"""
preprocessing.py
================
Handles all data preprocessing steps including:
  - Dropping irrelevant columns
  - Encoding categorical features
  - Scaling numerical features

Step: 3 - Data Preprocessing
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that do not contribute to the model.

    The 'Home' column is just a row identifier and carries
    no predictive value for house prices.

    Args:
        df (pd.DataFrame): Raw housing DataFrame.

    Returns:
        pd.DataFrame: DataFrame without the 'Home' column.
    """
    # Drop the ID column — not a feature
    df = df.drop(columns=["Home"])
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical string columns into numeric values.

    - 'Brick' (Yes/No) -> binary 1/0 using LabelEncoder
    - 'Neighborhood' (East/North/West) -> integer codes using LabelEncoder

    Args:
        df (pd.DataFrame): DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with encoded numeric columns.
    """
    le = LabelEncoder()

    # Encode 'Brick': Yes -> 1, No -> 0
    df["Brick"] = le.fit_transform(df["Brick"])

    # Encode 'Neighborhood': East/North/West -> 0/1/2
    df["Neighborhood"] = le.fit_transform(df["Neighborhood"])

    return df


def scale_features(df: pd.DataFrame, target_col: str = "Price") -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Standardize numerical feature columns using StandardScaler.

    Scaling ensures all features contribute equally to the model
    by bringing them to the same scale (mean=0, std=1).

    Args:
        df (pd.DataFrame): Fully encoded DataFrame.
        target_col (str): Name of the target column to exclude from scaling.

    Returns:
        tuple:
            - X_scaled (pd.DataFrame): Scaled feature matrix.
            - y (pd.Series): Target variable (Price).
            - scaler (StandardScaler): Fitted scaler (for inverse transform later).
    """
    # Separate features from the target variable
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fit and transform the feature matrix
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Run the full preprocessing pipeline in sequence.

    Steps:
        1. Drop irrelevant columns (Home)
        2. Encode categorical features (Brick, Neighborhood)
        3. Scale numerical features

    Args:
        df (pd.DataFrame): Raw housing DataFrame.

    Returns:
        tuple:
            - X_scaled (pd.DataFrame): Preprocessed feature matrix.
            - y (pd.Series): Target variable (Price).
            - scaler (StandardScaler): Fitted scaler instance.
    """
    df = drop_irrelevant_columns(df)
    df = encode_categorical_features(df)
    X_scaled, y, scaler = scale_features(df)
    return X_scaled, y, scaler


if __name__ == "__main__":
    from data_loader import load_data, inspect_data

    raw_df = load_data()

    print("--- Raw Data ---")
    inspect_data(raw_df)

    X, y, scaler = preprocess(raw_df)

    print("\n--- Preprocessed Features (first 5 rows) ---")
    print(X.head().to_string(index=False))

    print("\n--- Target Variable (first 5 values) ---")
    print(y.head().to_string())

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Target vector shape  : {y.shape}")
