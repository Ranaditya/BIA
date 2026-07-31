"""Clean the selected rainfall series and prepare monthly observations."""

import pandas as pd


DATE_COLUMN = "Date"
TARGET_COLUMN = "Rainfall_Gallicano"


def preprocess(
    df: pd.DataFrame,
    date_col: str = DATE_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> pd.Series:
    """Return a clean monthly rainfall series for ARIMA.

    Dates are parsed, exact/date duplicates are removed, and non-numeric target
    values are converted to missing values. Daily rainfall is aggregated to
    monthly totals. Missing months inside the observed range are interpolated;
    leading and trailing unavailable periods are excluded.
    """
    required = {date_col, target_col}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    clean = df[[date_col, target_col]].copy()
    original_rows = len(clean)
    exact_duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates()

    clean[date_col] = pd.to_datetime(clean[date_col], dayfirst=True, errors="coerce")
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    invalid_dates = int(clean[date_col].isna().sum())
    clean = clean.dropna(subset=[date_col]).sort_values(date_col)

    date_duplicates = int(clean.duplicated(subset=[date_col]).sum())
    if date_duplicates:
        clean = clean.groupby(date_col, as_index=False)[target_col].mean()

    daily = clean.set_index(date_col)[target_col]
    monthly = daily.resample("MS").sum(min_count=1).dropna()
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index)
    missing_months = int(monthly.isna().sum())
    monthly = monthly.interpolate(method="time").ffill().bfill()
    monthly.name = target_col
    monthly.index.name = date_col

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    print(f"Raw rows: {original_rows}")
    print(f"Exact duplicates removed: {exact_duplicates}")
    print(f"Invalid dates removed: {invalid_dates}")
    print(f"Duplicate dates consolidated: {date_duplicates}")
    print(f"Missing monthly values filled: {missing_months}")
    print(f"Clean monthly observations: {len(monthly)}")
    print(f"Period: {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}")
    return monthly


if __name__ == "__main__":
    from data_loader import load_data

    print(preprocess(load_data()).head())
