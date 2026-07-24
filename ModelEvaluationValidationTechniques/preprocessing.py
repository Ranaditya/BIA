"""
preprocessing.py
================
Handles all data preprocessing steps including:
  - Converting Date to datetime format
  - Handling missing values
  - Feature engineering (daily returns, target variable)
  - Scaling numerical features
  - Train/Test split

Step: 4 - Data Preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Date Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_date_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'Date' column to datetime format if it exists.

    Args:
        df (pd.DataFrame): Raw Apple Stock DataFrame.

    Returns:
        pd.DataFrame: DataFrame with Date as datetime.
    """
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        print("✓ Date column converted to datetime format.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Handle Missing Values
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by forward-filling for time series data
    and removing rows with NaN in critical columns.

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    print("\n" + "=" * 50)
    print("HANDLING MISSING VALUES")
    print("=" * 50)

    # Forward fill for time series data (stock prices)
    df = df.ffill()

    # Remove any remaining NaN values
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)

    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing values.")
    else:
        print("✓ No critical missing values to remove.")

    print("=" * 50)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from the stock data:
      - Daily Returns: percentage change from previous close
      - Price Range: difference between high and low
      - Volume Change: percentage change in volume
      - MA_5: 5-day moving average of close price
      - MA_20: 20-day moving average of close price
      - RSI: Relative Strength Index (14 periods)

    Args:
        df (pd.DataFrame): Apple Stock DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    # Daily Returns: (Close - Close_prev) / Close_prev
    if "Close" in df.columns:
        df["Daily_Returns"] = df["Close"].pct_change()
        print("\nDaily_Returns preview:")
        print(df["Daily_Returns"].head().to_string())

    # Price Range: High - Low
    if "High" in df.columns and "Low" in df.columns:
        df["Price_Range"] = df["High"] - df["Low"]

    # Volume Change: percentage change in volume
    if "Volume" in df.columns:
        df["Volume_Change"] = df["Volume"].pct_change()

    # Moving Averages
    if "Close" in df.columns:
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    if "Close" in df.columns:
        df["RSI"] = compute_rsi(df["Close"], period=14)

    # Remove rows with NaN from engineering
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)

    print(f"✓ Features engineered. Removed {removed_rows} rows (NaN from rolling calculations).")
    print("=" * 50)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        series (pd.Series): Price series.
        period (int): Period for RSI calculation. Default 14.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Define Target Variable
# ─────────────────────────────────────────────────────────────────────────────

def define_target_variable(df: pd.DataFrame, target_column: str = "Close") -> pd.DataFrame:
    """
    Define a binary classification target: Price Increase (1) or Decrease (0).
    
    Based on the Daily_Returns feature:
      - If Daily_Returns > 0: Price increased (class 1)
      - If Daily_Returns <= 0: Price decreased (class 0)

    Args:
        df (pd.DataFrame): Apple Stock DataFrame with Daily_Returns.
        target_column (str): Column to track for target definition.

    Returns:
        pd.DataFrame: DataFrame with target variable added.
    """
    print("\n" + "=" * 50)
    print("DEFINING TARGET VARIABLE")
    print("=" * 50)

    if "Daily_Returns" in df.columns:
        # Create binary target: 1 if price increased, 0 otherwise
        df["Target"] = (df["Daily_Returns"] > 0).astype(int)
        class_distribution = df["Target"].value_counts()
        print(f"Target Distribution:\n{class_distribution.to_string()}")
        print(f"Class Balance: {class_distribution[1] / len(df) * 100:.2f}% increase, {class_distribution[0] / len(df) * 100:.2f}% decrease")
    else:
        print("Warning: Daily_Returns not found. Target not defined.")

    print("=" * 50)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Select and Scale Features
# ─────────────────────────────────────────────────────────────────────────────

def select_and_scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Select relevant features, separate features from target,
    and scale numerical features using StandardScaler.

    Args:
        df (pd.DataFrame): Preprocessed Apple Stock DataFrame.

    Returns:
        tuple: (scaled_features_df, target_series, scaler_object)
    """
    print("\n" + "=" * 50)
    print("FEATURE SELECTION & SCALING")
    print("=" * 50)

    # Select features (exclude non-numeric and non-predictive columns)
    exclude_cols = ["Date", "Target", "Daily_Returns"]  # Don't scale date or target
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"Selected Features: {feature_cols}")

    # Separate features and target
    X = df[feature_cols].copy()
    y = df["Target"].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    if "Daily_Returns" in df.columns:
        daily_returns_scaler = StandardScaler()
        daily_returns_scaled = daily_returns_scaler.fit_transform(df[["Daily_Returns"]])
        print(
            f"Daily_Returns (standardized) max: {daily_returns_scaled[:, 0].max():.4f}"
        )

    print(f"✓ Features scaled using StandardScaler.")
    print("=" * 50)

    return X_scaled, y, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Train/Test Split
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
        y (pd.Series): Target variable (Price Increase/Decrease).
        test_size (float): Proportion of data reserved for testing. Default 0.20.
        random_state (int): Seed for reproducibility. Default 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n" + "=" * 50)
    print("TRAIN / TEST SPLIT")
    print("=" * 50)
    print(f"Total samples        : {len(X)}")
    print(f"Training set size    : {len(X_train)} observations")
    print(f"Training set         : {len(X_train)} samples ({100 - int(test_size * 100)}%)")
    print(f"Test set             : {len(X_test)} samples ({int(test_size * 100)}%)")
    print("=" * 50)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the complete preprocessing pipeline and return train/test sets.

    Args:
        df (pd.DataFrame): Raw Apple Stock DataFrame.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # Step 1: Convert date
    df = convert_date_to_datetime(df)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Engineer features
    df = engineer_features(df)

    # Step 4: Define target variable
    df = define_target_variable(df)

    # Step 5: Select and scale features
    X, y, scaler = select_and_scale_features(df)

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n✓ Preprocessing complete. Data ready for modeling.")
    return X_train, X_test, y_train, y_test
