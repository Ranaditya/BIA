"""Run the complete IMDb sentiment classification pipeline."""

from data_loader import inspect_data, load_data
from eda import run_eda
from model import build_model, evaluate_model, plot_training_history, train_model
from preprocessing import DEFAULT_MAX_VOCAB_SIZE, DEFAULT_SEQUENCE_LENGTH, prepare_data


def main() -> None:
    """Load data, prepare text, train the classifier, and evaluate results."""
    print("\n" + "=" * 70)
    print("IMDb SENTIMENT CLASSIFICATION")
    print("=" * 70)

    print("\n[STEP 1] Loading dataset")
    dataframe = load_data()
    inspect_data(dataframe)

    print("\n[STEP 2] Exploratory data analysis")
    run_eda(dataframe)

    print("\n[STEP 3] Tokenizing and padding reviews")
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(
        dataframe,
        max_vocab_size=DEFAULT_MAX_VOCAB_SIZE,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Vocabulary limit: {DEFAULT_MAX_VOCAB_SIZE}")
    print(f"Sequence length: {DEFAULT_SEQUENCE_LENGTH}")
    print(f"Tokenizer vocabulary learned: {len(tokenizer.word_index):,} words")

    print("\n[STEP 4] Building and compiling model")
    model = build_model(DEFAULT_MAX_VOCAB_SIZE, DEFAULT_SEQUENCE_LENGTH)
    model.summary()

    print("\n[STEP 5] Training model")
    history = train_model(model, X_train, y_train)
    plot_training_history(history)

    print("\n[STEP 6] Evaluating model")
    metrics = evaluate_model(model, X_test, y_test)
    print("\nSaved plots to: images/")
    print(f"Final test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
