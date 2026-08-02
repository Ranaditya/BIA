"""Prepare Iris features and labels for model training."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_data(dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 189):
    """Split features/labels, encode labels, then standardize without leakage."""
    X = dataframe.drop(columns="species")
    labels = dataframe["species"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler
