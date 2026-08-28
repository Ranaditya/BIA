"""Load the local IMDB review data and pre-trained GloVe word vectors."""

from pathlib import Path
from typing import Iterator

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
IMDB_FILE = DATA_DIR / "IMDB.csv"
GLOVE_FILE = DATA_DIR / "glove.txt"
REQUIRED_COLUMNS = {"review", "sentiment"}


def load_imdb_data(filepath: Path = IMDB_FILE, max_samples: int | None = None) -> pd.DataFrame:
    """Load IMDB reviews and validate the expected text-classification schema."""
    if not filepath.exists():
        raise FileNotFoundError(f"IMDB dataset not found: {filepath}")

    dataframe = pd.read_csv(filepath, nrows=max_samples)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"IMDB dataset is missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.loc[:, ["review", "sentiment"]].dropna().copy()
    dataframe["review"] = dataframe["review"].astype(str)
    dataframe["sentiment"] = dataframe["sentiment"].astype(str).str.lower().str.strip()
    unexpected_labels = set(dataframe["sentiment"].unique()).difference({"positive", "negative"})
    if unexpected_labels:
        raise ValueError(f"Unexpected sentiment labels: {sorted(unexpected_labels)}")
    return dataframe


def load_glove_embeddings(filepath: Path = GLOVE_FILE) -> tuple[dict[str, list[float]], int]:
    """Load a whitespace-delimited GloVe file and return vectors with their dimension."""
    if not filepath.exists():
        raise FileNotFoundError(f"GloVe embedding file not found: {filepath}")

    embeddings: dict[str, list[float]] = {}
    embedding_dimension: int | None = None
    skipped_rows = 0
    with filepath.open(encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.rstrip().split()
            if len(values) < 2:
                skipped_rows += 1
                continue
            word, raw_vector = values[0], values[1:]
            if embedding_dimension is None:
                embedding_dimension = len(raw_vector)
            if len(raw_vector) != embedding_dimension:
                skipped_rows += 1
                continue
            try:
                embeddings[word] = [float(value) for value in raw_vector]
            except ValueError:
                skipped_rows += 1

    if not embeddings or embedding_dimension is None:
        raise ValueError(f"No word vectors found in: {filepath}")
    if skipped_rows:
        print(f"Skipped {skipped_rows} malformed GloVe rows.")
    return embeddings, embedding_dimension


def iter_glove_embeddings(filepath: Path = GLOVE_FILE) -> Iterator[tuple[str, list[float]]]:
    """Yield GloVe entries for lightweight inspection without retaining all vectors."""
    with filepath.open(encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.rstrip().split()
            if len(values) >= 2:
                yield values[0], [float(value) for value in values[1:]]