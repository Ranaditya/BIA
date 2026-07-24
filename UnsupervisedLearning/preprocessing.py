"""
preprocessing.py
================
Prepare the wholesale customer data for clustering.
"""

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list[str], StandardScaler]:
    """Drop non-numeric identifier-like columns and standardize the clustering features."""
    feature_frame = df.copy()

    # Use the spend-related numeric columns for clustering.
    clustering_columns = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen",
    ]
    feature_frame = feature_frame[clustering_columns].copy()

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(feature_frame),
        columns=feature_frame.columns,
        index=feature_frame.index,
    )

    print("\n[PREPROCESSING] Standardized features:")
    print(scaled_features.head())
    print(f"\nFeature shape after scaling: {scaled_features.shape}")
    return feature_frame, scaled_features, clustering_columns, scaler
