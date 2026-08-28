"""Build, train, evaluate, and use an LSTM sentiment classifier."""

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def build_lstm_model(embedding_matrix: np.ndarray, sequence_length: int) -> Sequential:
    """Build a binary LSTM classifier initialized with pre-trained GloVe weights."""
    model = Sequential(
        [
            Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
            ),
            LSTM(64, dropout=0.2, recurrent_dropout=0.0),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    model: Sequential,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    epochs: int = 5,
    batch_size: int = 128,
):
    """Fit the classifier and return Keras history for later visualization."""
    return model.fit(
        train_features,
        train_labels,
        validation_data=(validation_features, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )


def evaluate_model(model: Sequential, test_features: np.ndarray, test_labels: np.ndarray) -> dict[str, float]:
    """Evaluate held-out reviews using accuracy, precision, recall, and F1."""
    probabilities = model.predict(test_features, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(np.int32)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_labels, predictions, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }
    print("\nTest classification report:")
    print(classification_report(test_labels, predictions, target_names=["negative", "positive"], zero_division=0))
    return metrics


def predict_sentiment(model: Sequential, padded_reviews: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sentiment labels and confidence scores for padded custom reviews."""
    probabilities = model.predict(padded_reviews, verbose=0).ravel()
    labels = np.where(probabilities >= 0.5, "positive", "negative")
    return labels, probabilities