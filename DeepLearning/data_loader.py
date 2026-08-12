"""Load the CIFAR-10 image-classification dataset."""

from __future__ import annotations

import numpy as np
from tensorflow import keras


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (if needed) and return the CIFAR-10 train/test arrays."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Keras returns labels with shape (samples, 1). A flat vector is easier to use.
    y_train = y_train.squeeze().astype("int64")
    y_test = y_test.squeeze().astype("int64")
    return x_train, y_train, x_test, y_test


def inspect_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print a concise summary of the loaded dataset."""
    print(f"Training images: {x_train.shape}; labels: {y_train.shape}")
    print(f"Test images:     {x_test.shape}; labels: {y_test.shape}")
    print(f"Pixel range:     {x_train.min()} to {x_train.max()}")
    print(f"Classes:         {len(CLASS_NAMES)}")


if __name__ == "__main__":
    inspect_data(*load_data())
