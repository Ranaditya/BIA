"""Run the complete CIFAR-10 convolutional neural network pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.resolve()
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import CLASS_NAMES, inspect_data, load_data
from model import (
    build_model,
    compile_model,
    evaluate_model,
    predict_classes,
    set_random_seed,
    train_model,
)
from preprocessing import preprocess_data
from visualization import plot_predictions, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--predictions", type=int, default=10, help="Test images to display")
    parser.add_argument("--show-plots", action="store_true", help="Open plots after saving them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.predictions < 1:
        raise ValueError("epochs, batch size, and predictions must all be positive")

    print("\nCIFAR-10 CNN IMAGE CLASSIFICATION")
    print("=" * 50)
    set_random_seed()

    print("\n[Steps 1-2] Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_data()
    inspect_data(x_train, y_train, x_test, y_test)

    print("\n[Step 3] Normalizing pixel values...")
    x_train, x_test = preprocess_data(x_train, x_test)
    print(f"Normalized range: {x_train.min():.1f} to {x_train.max():.1f}")

    print("\n[Steps 4-5] Building and compiling the CNN...")
    cnn = build_model()
    compile_model(cnn)
    cnn.summary()

    print("\n[Step 6] Training the model...")
    history = train_model(cnn, x_train, y_train, args.epochs, args.batch_size)

    print("\n[Step 7] Evaluating on the test set...")
    metrics = evaluate_model(cnn, x_test, y_test)
    print(f"Test loss:     {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.2%}")

    print("\n[Step 8] Plotting training history...")
    history_path = plot_training_history(history, args.show_plots)

    print("\n[Steps 9-10] Predicting and displaying test images...")
    count = min(args.predictions, len(x_test))
    predicted, confidence = predict_classes(cnn, x_test[:count])
    prediction_path = plot_predictions(
        x_test[:count], y_test[:count], predicted, confidence, CLASS_NAMES, args.show_plots
    )

    print("\nPipeline completed successfully.")
    print(f"Training plot:   {history_path.relative_to(PROJECT_DIR)}")
    print(f"Prediction plot: {prediction_path.relative_to(PROJECT_DIR)}")
    print("Best model:      models/best_cifar10_cnn.keras")


if __name__ == "__main__":
    main()
