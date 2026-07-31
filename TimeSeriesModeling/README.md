# Rainfall Time-Series Modeling with ARIMA

This project loads the Auser aquifer dataset from `data/`, cleans and aggregates
the `Rainfall_Gallicano` daily series into monthly rainfall totals, explores its
pattern and trend, and evaluates an ARIMA model on a chronological test set.

## Project structure

```text
TimeSeriesModeling/
|-- data/
|-- images/                 # generated plots
|-- outputs/                # generated metrics and predictions
|-- data_loader.py          # data loading and inspection
|-- preprocessing.py        # missing values and duplicate cleaning
|-- eda.py                  # time-series and trend visualization
|-- model.py                # split, ARIMA training, forecast, evaluation
|-- main.py                 # complete pipeline entry point
`-- README.md
```

## Method

- Parse `Date` using day-first format.
- Remove exact duplicates and consolidate duplicate dates, if present.
- Aggregate daily `Rainfall_Gallicano` measurements to monthly totals.
- Fill missing months with time interpolation.
- Reserve the last 20% of observations as the test set.
- Fit ARIMA(2, 0, 2) only on the training set.
- Forecast the full testing period and report MSE, RMSE, and MAE.

The split is chronological rather than random because future observations must
not be used to predict the past.

## Run

From this folder:

```powershell
python main.py
```

If using the course virtual environment:

```powershell
..\.venv-1\Scripts\python.exe main.py
```

Generated plots and CSV results are written to `images/` and `outputs/`.
