"""Run IMDB sentiment classification with N-grams, GloVe, and an LSTM."""

import argparse
import os
from pathlib import Path

import numpy as np
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from data_loader import load_glove_embeddings, load_imdb_data
from eda import create_ngram_features, inspect_data
from model import build_lstm_model, evaluate_model, predict_sentiment, train_model
from preprocessing import clean_reviews, create_embedding_matrix, encode_labels, tokenize_and_pad
from visualization import plot_training_history


def parse_arguments() -> argparse.Namespace:
    """Accept practical training and custom-review options from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional row limit for a faster run.")
    parser.add_argument("--review", action="append", default=[], help="Custom movie review to classify; repeatable.")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the complete IMDB to LSTM sentiment-classification workflow."""
    os.chdir(Path(__file__).parent)
    arguments = parse_arguments()
    np.random.seed(42)

    print("\nIMDB SENTIMENT ANALYSIS WITH GLOVE + LSTM\n" + "=" * 50)
    dataframe = load_imdb_data(max_samples=arguments.max_samples)
    dataframe["review"] = clean_reviews(dataframe["review"])
    inspect_data(dataframe)

    train_reviews, test_reviews, train_labels_text, test_labels_text = train_test_split(
        dataframe["review"], dataframe["sentiment"], test_size=0.2, random_state=42, stratify=dataframe["sentiment"]
    )
    print(f"\nTrain/test split: {len(train_reviews):,} / {len(test_reviews):,}")
    create_ngram_features(train_reviews)
    train_labels, test_labels = encode_labels(train_labels_text), encode_labels(test_labels_text)

    tokenizer, train_features, test_features = tokenize_and_pad(train_reviews, test_reviews)
    glove_embeddings, embedding_dimension = load_glove_embeddings()
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, glove_embeddings, embedding_dimension, 20_000)
    model = build_lstm_model(embedding_matrix, sequence_length=train_features.shape[1])
    model.summary()
    history = train_model(model, train_features, train_labels, test_features, test_labels, arguments.epochs, arguments.batch_size)
    metrics = evaluate_model(model, test_features, test_labels)
    chart_file = plot_training_history(history, arguments.show_plots)

    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize():10}: {value:.4f}")
    print(f"Training chart saved to: {chart_file}")

    for review in arguments.review:
        sequence = tokenizer.texts_to_sequences([review])
        padded_review = pad_sequences(sequence, maxlen=train_features.shape[1], padding="post", truncating="post")
        label, confidence = predict_sentiment(model, padded_review)
        print(f"\nCustom review: {review}\nPrediction: {label[0]} ({confidence[0]:.2%} positive confidence)")


if __name__ == "__main__":
    main()