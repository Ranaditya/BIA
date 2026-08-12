"""Define, compile, train, evaluate, and use the CIFAR-10 CNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODELS_DIR = Path(__file__).parent / "models"


def build_model(input_shape: tuple[int, int, int] = (32, 32, 3), num_classes: int = 10) -> keras.Model:
    """Build a compact convolutional neural network for CIFAR-10."""
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cifar10_cnn",
    )


def compile_model(model: keras.Model) -> None:
    """Configure the model for multiclass classification."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
) -> keras.callbacks.History:
    """Train the CNN and return its epoch-by-epoch history."""
    MODELS_DIR.mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            MODELS_DIR / "best_cifar10_cnn.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2,
    )


def evaluate_model(
    model: keras.Model, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Evaluate the trained CNN on unseen test images."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def predict_classes(model: keras.Model, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class IDs and confidence scores for images."""
    probabilities = model.predict(images, verbose=0)
    class_ids = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    return class_ids, confidence


def set_random_seed(seed: int = 42) -> None:
    """Make model initialization and training as reproducible as possible."""
    tf.keras.utils.set_random_seed(seed)
