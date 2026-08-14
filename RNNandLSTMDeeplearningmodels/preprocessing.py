"""Leakage-safe preprocessing for chronological RNN/LSTM sequences."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_COLUMN = "Close"


@dataclass
class SequenceData:
    """All model-ready splits plus fitted scalers and target dates."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    clean_data: pd.DataFrame
    train_end: int
    val_end: int
    sequence_length: int


def clean_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Coerce market columns to numeric and fill gaps without looking ahead first."""
    cleaned = dataframe.copy()
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Forward fill uses only earlier observations; backfill only handles a leading gap.
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].ffill().bfill()
    if cleaned[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Numeric columns still contain missing values after imputation.")
    return cleaned


def _make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, target_dates, target_indices = [], [], [], []
    for target_index in range(sequence_length, len(features)):
        X.append(features[target_index - sequence_length : target_index])
        y.append(targets[target_index])
        target_dates.append(dates[target_index])
        target_indices.append(target_index)
    return np.asarray(X), np.asarray(y), np.asarray(target_dates), np.asarray(target_indices)


def preprocess_data(
    dataframe: pd.DataFrame,
    sequence_length: int = 60,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> SequenceData:
    """Clean, scale, window, and chronologically split the time series.

    Scalers are fitted exclusively on the training period. Windows assigned to
    validation/test may include prior-period context, but their targets never do.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave data for testing.")

    cleaned = clean_missing_values(dataframe)
    row_count = len(cleaned)
    train_end = int(row_count * train_ratio)
    val_end = int(row_count * (train_ratio + validation_ratio))
    if train_end <= sequence_length or val_end >= row_count:
        raise ValueError("Dataset is too small for the requested sequence length and splits.")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(cleaned.loc[: train_end - 1, FEATURE_COLUMNS])
    target_scaler.fit(cleaned.loc[: train_end - 1, [TARGET_COLUMN]])

    scaled_features = feature_scaler.transform(cleaned[FEATURE_COLUMNS]).astype(np.float32)
    scaled_targets = target_scaler.transform(cleaned[[TARGET_COLUMN]]).astype(np.float32)
    dates = cleaned["Date"].to_numpy()
    X, y, target_dates, target_indices = _make_sequences(
        scaled_features, scaled_targets, dates, sequence_length
    )

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    return SequenceData(
        X_train=X[train_mask], y_train=y[train_mask],
        X_val=X[val_mask], y_val=y[val_mask],
        X_test=X[test_mask], y_test=y[test_mask],
        val_dates=target_dates[val_mask], test_dates=target_dates[test_mask],
        feature_scaler=feature_scaler, target_scaler=target_scaler,
        clean_data=cleaned, train_end=train_end, val_end=val_end,
        sequence_length=sequence_length,
    )
