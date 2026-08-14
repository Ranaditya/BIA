"""Build, train, evaluate, and forecast with LSTM or SimpleRNN models."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import FEATURE_COLUMNS, SequenceData


def _tensorflow() -> Any:
    """Import TensorFlow lazily so data utilities work without the DL runtime."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. From the repository root, run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return tf


def build_model(
    model_type: str,
    input_shape: tuple[int, int],
    units: int = 64,
    dropout: float = 0.20,
    learning_rate: float = 0.001,
):
    """Create a stacked LSTM or SimpleRNN regression network."""
    tf = _tensorflow()
    model_type = model_type.lower()
    layer_class = {"lstm": tf.keras.layers.LSTM, "rnn": tf.keras.layers.SimpleRNN}.get(model_type)
    if layer_class is None:
        raise ValueError("model_type must be 'lstm' or 'rnn'.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="price_sequence"),
            layer_class(units, return_sequences=True),
            tf.keras.layers.Dropout(dropout),
            layer_class(max(units // 2, 8)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="next_close"),
        ],
        name=f"stock_{model_type}",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(
    model,
    data: SequenceData,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 8,
    model_path: str | Path = "models/best_model.keras",
):
    """Train with early stopping and restore the lowest-validation-loss weights."""
    tf = _tensorflow()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(patience // 2, 2), min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]
    return model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )


def predict_prices(model, X: np.ndarray, data: SequenceData) -> np.ndarray:
    """Predict and convert normalized outputs back to stock-price units."""
    scaled = model.predict(X, verbose=0)
    return data.target_scaler.inverse_transform(scaled).ravel()


def actual_prices(y: np.ndarray, data: SequenceData) -> np.ndarray:
    """Convert normalized targets back to stock-price units."""
    return data.target_scaler.inverse_transform(y).ravel()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics in original price units."""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def forecast_future(model, data: SequenceData, days: int = 10) -> np.ndarray:
    """Generate a recursive illustrative forecast for the next trading days.

    Unknown future OHLC fields are approximated from predicted Close and the
    latest known volume. This is scenario extrapolation, not investment advice.
    """
    if days < 1:
        return np.asarray([], dtype=float)

    scaled_rows = data.feature_scaler.transform(data.clean_data[FEATURE_COLUMNS]).astype(np.float32)
    window = scaled_rows[-data.sequence_length :].copy()
    forecasts = []
    close_index = FEATURE_COLUMNS.index("Close")
    price_indices = [FEATURE_COLUMNS.index(name) for name in ["Open", "High", "Low", "Close", "Adj Close"]]

    for _ in range(days):
        scaled_close = float(model.predict(window[np.newaxis, ...], verbose=0)[0, 0])
        close = float(data.target_scaler.inverse_transform([[scaled_close]])[0, 0])
        forecasts.append(close)
        next_row = window[-1].copy()
        # Close uses the same MinMax feature scaling as the target's source column.
        next_row[close_index] = scaled_close
        for index in price_indices:
            if index != close_index:
                # Approximate unknown future price fields by converting predicted Close
                # through each feature's own scaler range.
                feature_min = data.feature_scaler.data_min_[index]
                feature_range = data.feature_scaler.data_range_[index] or 1.0
                next_row[index] = (close - feature_min) / feature_range
        window = np.vstack([window[1:], next_row])
    return np.asarray(forecasts)
