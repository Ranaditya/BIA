# Stock Price Prediction with RNN and LSTM

This project predicts the next closing price from the preceding 60 trading days of
Open, High, Low, Close, Adjusted Close, and Volume data. Its module layout mirrors
`MachineLearningIntroduction`: loading, preprocessing, modelling, visualisation,
and a single `main.py` entry point.

## Project structure

```text
RNNandLSTMDeeplearningmodels/
|-- data/Stock_price.csv       # bundled historical dataset
|-- data_loader.py             # loading and inspection
|-- preprocessing.py           # missing values, scaling, windows, time splits
|-- model.py                   # RNN/LSTM, training, metrics, forecasting
|-- visualization.py           # loss, predictions, and forecast plots
|-- main.py                    # complete command-line pipeline
|-- images/                    # generated at runtime
`-- models/                    # best checkpoint generated at runtime
```

All repository dependencies are maintained in the single project-level
`BIA/requirements.txt` file.

## Method

- Missing numeric values are forward-filled (then backfilled only for leading gaps).
- Data remains chronological: 70% training, 15% validation, and 15% test.
- Min-max scalers are fitted only on the training period, preventing data leakage.
- Each sample contains 60 days and targets the next day's Close.
- The network contains an explicit input, two LSTM or SimpleRNN layers, dropout,
  a dense hidden layer, and a one-unit regression output.
- Mean squared error is the loss and Adam is the optimizer. Early stopping monitors
  validation loss and restores the best weights.
- MAE, RMSE, and R-squared are reported in original price units.

## Setup and run

TensorFlow does not currently publish standard Python 3.14 wheels. Use Python 3.11,
3.12, or 3.13 for the deep-learning environment.

```powershell
# Run these commands from the BIA repository root.
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd RNNandLSTMDeeplearningmodels
python main.py --model lstm
python main.py --model rnn --epochs 75 --batch-size 32
```

Useful tunable hyperparameters are `--epochs`, `--batch-size`, `--learning-rate`,
`--units`, `--sequence-length`, and `--patience`. Tune against validation metrics;
use the test set only for the final unbiased estimate.

## Evaluation and interpretation

MAE gives the average absolute price error; RMSE penalizes occasional large misses;
R-squared measures variance explained and can be negative on genuinely unseen time
periods. Inspect `images/test_predictions.png` for lag around abrupt changes. Common
challenges include non-stationary markets, distribution shifts, overfitting, and the
fact that historical OHLCV inputs omit news and macroeconomic drivers.

The future forecast is recursive. Because future Open/High/Low/Adjusted Close and
Volume are unknown, it uses the predicted Close as a simple price-field proxy and
holds the latest volume context. It is an educational scenario, not financial advice.
