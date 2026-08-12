"""Preprocessing helpers for CIFAR-10."""

from __future__ import annotations

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Convert uint8 pixels in [0, 255] to float32 values in [0, 1]."""
    return images.astype("float32") / 255.0


def preprocess_data(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both training and test image arrays."""
    return normalize_images(x_train), normalize_images(x_test)


if __name__ == "__main__":
    from data_loader import load_data

    train_images, _, test_images, _ = load_data()
    train_images, test_images = preprocess_data(train_images, test_images)
    print(f"Training pixel range: {train_images.min():.1f} to {train_images.max():.1f}")
    print(f"Test pixel range:     {test_images.min():.1f} to {test_images.max():.1f}")
