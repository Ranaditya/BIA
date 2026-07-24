"""Feature preparation, splitting, standardization, PCA, and feature engineering."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"
ENGINEERED_FEATURE = "sepal_length_x_petal_length"


@dataclass
class ScaledSplit:
    """Standardized training and testing data plus the fitted scaler."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: list[str]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the four measurements (X) from the species label (y)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def create_split_indices(
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """Create reproducible, stratified row indices shared by both experiments."""
    train_indices, test_indices = train_test_split(
        y.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return pd.Index(train_indices), pd.Index(test_indices)


def standardize_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_indices: pd.Index,
    test_indices: pd.Index,
) -> ScaledSplit:
    """Split data and standardize it without leaking test-set information."""
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return ScaledSplit(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=X.columns.tolist(),
    )


def reduce_with_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on training data and transform both training and testing data."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def add_engineered_feature(X: pd.DataFrame) -> pd.DataFrame:
    """Add the requested sepal_length * petal_length interaction feature."""
    engineered = X.copy()
    engineered[ENGINEERED_FEATURE] = (
        engineered["sepal_length"] * engineered["petal_length"]
    )
    return engineered


if __name__ == "__main__":
    from data_loader import load_data

    features, target = split_features_target(load_data())
    train_idx, test_idx = create_split_indices(target)
    split = standardize_split(features, target, train_idx, test_idx)
    train_pca, test_pca, fitted_pca = reduce_with_pca(split.X_train, split.X_test)
    print(f"Original features: {features.shape}")
    print(f"Training data after PCA: {train_pca.shape}")
    print(f"Testing data after PCA: {test_pca.shape}")
    print(f"Explained variance: {fitted_pca.explained_variance_ratio_.sum():.4f}")
    print(f"Engineered features: {add_engineered_feature(features).shape}")
