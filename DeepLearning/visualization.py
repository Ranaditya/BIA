"""Visualizations for CNN training and CIFAR-10 predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


IMAGES_DIR = Path(__file__).parent / "images"


def plot_training_history(history: keras.callbacks.History, show: bool = False) -> Path:
    """Save training/validation loss and accuracy curves."""
    IMAGES_DIR.mkdir(exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.history["loss"], label="Training")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation")
    axes[0].set(title="Model Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history.history["accuracy"], label="Training")
    axes[1].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[1].set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    output = IMAGES_DIR / "training_history.png"
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output


def plot_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    class_names: tuple[str, ...],
    show: bool = False,
) -> Path:
    """Save a grid of test images with actual and predicted class names."""
    IMAGES_DIR.mkdir(exist_ok=True)
    count = len(images)
    columns = min(5, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        actual = class_names[int(true_labels[index])]
        predicted = class_names[int(predicted_labels[index])]
        color = "green" if actual == predicted else "red"
        axis.imshow(images[index])
        axis.set_title(
            f"Actual: {actual}\nPredicted: {predicted} ({confidence[index]:.1%})",
            color=color,
            fontsize=9,
        )

    output = IMAGES_DIR / "test_predictions.png"
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output
