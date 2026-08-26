"""Define, train, evaluate, and visualize an IMDb sentiment model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def build_model(max_vocab_size: int, sequence_length: int, embedding_dim: int = 64) -> tf.keras.Model:
    """Build a compact binary text classifier using embeddings and flattening."""
    model = Sequential(
        [
            Input(shape=(sequence_length,), dtype="int32"),
            Embedding(input_dim=max_vocab_size, output_dim=embedding_dim),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ],
        name="imdb_embedding_classifier",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 5,
    batch_size: int = 128,
) -> tf.keras.callbacks.History:
    """Train the model while reserving 20% of training data for validation."""
    return model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate test loss/accuracy and report thresholded binary predictions."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities > 0.5).astype("int32")
    print("\nTEST EVALUATION")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Thresholded accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    return {"loss": float(loss), "accuracy": float(accuracy)}


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save training/validation accuracy and loss plots."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(IMAGES_DIR / "training_history.png", dpi=150)
    plt.close(figure)
