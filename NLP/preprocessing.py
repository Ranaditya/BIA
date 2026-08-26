"""Prepare IMDb text and sentiment labels for a Keras classifier."""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DEFAULT_MAX_VOCAB_SIZE = 20_000
DEFAULT_SEQUENCE_LENGTH = 200


def clean_review(review: str) -> str:
    """Remove HTML tags and normalize whitespace in one review."""
    without_tags = re.sub(r"<[^>]+>", " ", str(review))
    return re.sub(r"\s+", " ", without_tags).strip().lower()


def prepare_data(
    dataframe: pd.DataFrame,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    """Split, tokenize, and pad reviews, fitting the tokenizer on training text only."""
    reviews = dataframe["review"].map(clean_review)
    labels = dataframe["sentiment"].str.lower().map({"negative": 0, "positive": 1})
    if labels.isna().any():
        raise ValueError("Sentiment labels must be 'positive' or 'negative'.")

    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        reviews,
        labels.astype("int32"),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_reviews)

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_reviews),
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(test_reviews),
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )

    return (
        X_train,
        X_test,
        train_labels.to_numpy(),
        test_labels.to_numpy(),
        tokenizer,
    )
