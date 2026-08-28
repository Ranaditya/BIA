"""Exploratory summaries for IMDB sentiment reviews, including N-grams."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def inspect_data(dataframe: pd.DataFrame) -> None:
    """Print the dataset size, label balance, and review-length summary."""
    review_lengths = dataframe["review"].str.split().str.len()
    print(f"Dataset: {len(dataframe):,} reviews")
    print("Sentiment distribution:")
    print(dataframe["sentiment"].value_counts().to_string())
    print(f"Average review length: {review_lengths.mean():.1f} words")


def create_ngram_features(reviews: pd.Series, max_features: int = 10_000) -> CountVectorizer:
    """Fit a unigram/bigram count vectorizer to expose local text context."""
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words="english")
    ngram_matrix = vectorizer.fit_transform(reviews)
    print(f"N-gram feature matrix: {ngram_matrix.shape[0]:,} reviews x {ngram_matrix.shape[1]:,} features")
    return vectorizer