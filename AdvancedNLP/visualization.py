"""Visualization helpers for the LSTM training history."""

from pathlib import Path

import matplotlib.pyplot as plt


IMAGES_DIR = Path(__file__).parent / "images"


def plot_training_history(history, show_plot: bool = False) -> Path:
    """Save training and validation accuracy/loss over epochs."""
    IMAGES_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Model Loss", xlabel="Epoch", ylabel="Binary Cross-Entropy")
    axes[1].legend()
    figure.tight_layout()
    output_file = IMAGES_DIR / "training_history.png"
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)
    return output_file