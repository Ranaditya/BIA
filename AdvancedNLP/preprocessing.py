"""Prepare IMDB review text, numeric labels, and GloVe embedding weights."""

import re

import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


def clean_reviews(reviews: pd.Series) -> pd.Series:
    """Remove HTML markup and normalize whitespace while retaining word content."""
    return reviews.str.replace(r"<br\s*/?>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Convert negative/positive sentiment strings to binary 0/1 labels."""
    label_mapping = {"negative": 0, "positive": 1}
    encoded = labels.map(label_mapping)
    if encoded.isna().any():
        raise ValueError("Labels must contain only 'negative' and 'positive'.")
    return encoded.to_numpy(dtype=np.int32)


def tokenize_and_pad(
    train_reviews: pd.Series,
    test_reviews: pd.Series,
    vocabulary_size: int = 20_000,
    sequence_length: int = 200,
) -> tuple[Tokenizer, np.ndarray, np.ndarray]:
    """Fit a Keras tokenizer on training data and pad train/test sequences."""
    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_reviews)
    train_sequences = tokenizer.texts_to_sequences(train_reviews)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding="post", truncating="post")
    test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding="post", truncating="post")
    return tokenizer, train_padded, test_padded


def create_embedding_matrix(
    word_index: dict[str, int],
    glove_embeddings: dict[str, list[float]],
    embedding_dimension: int,
    vocabulary_size: int,
) -> np.ndarray:
    """Create model embedding weights from tokenizer vocabulary and GloVe vectors."""
    usable_vocabulary = min(vocabulary_size, len(word_index) + 1)
    embedding_matrix = np.zeros((usable_vocabulary, embedding_dimension), dtype=np.float32)
    matched_words = 0
    for word, index in word_index.items():
        if index >= usable_vocabulary:
            continue
        vector = glove_embeddings.get(word)
        if vector is not None:
            embedding_matrix[index] = vector
            matched_words += 1
    print(f"GloVe coverage: {matched_words:,} / {usable_vocabulary - 1:,} tokenizer words")
    return embedding_matrix