"""Preprocessing utilities for car mileage regression."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing sequence."""
    cleaned = clean_data(dataframe)
    adjusted = cap_outliers(cleaned)
    return adjusted
