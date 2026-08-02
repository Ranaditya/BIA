"""Run the complete Iris SVM and KNN classification workflow."""

from data_loader import inspect_data, load_data
from eda import run_eda, summary_statistics
from model import run_models
from preprocessing import prepare_data


def main() -> None:
    print("=" * 64)
    print("IRIS CLASSIFICATION WITH SVM AND KNN")
    print("=" * 64)

    print("\n[Steps 1-2] Import libraries and load the Iris dataset")
    dataframe = load_data()
    inspect_data(dataframe)

    print("\n[Step 3] Compute summary statistics")
    summary_statistics(dataframe)

    print("\n[Steps 4-5] Run EDA and generate the pairplot")
    run_eda(dataframe)

    print("\n[Steps 6-9] Separate, encode, split, and standardize data")
    X_train, X_test, y_train, y_test, encoder, _ = prepare_data(dataframe)
    print(f"Training samples: {len(X_train)}; testing samples: {len(X_test)}")
    print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

    print("\n[Steps 10-17] Train, predict, and evaluate SVM and KNN")
    _, _, scores = run_models(
        X_train, X_test, y_train, y_test, class_names=encoder.classes_, k=3
    )

    print("\nModel comparison:")
    for model_name, score in scores.items():
        print(f"  {model_name}: {score:.2%}")
    print("\nPlots saved in the images folder.")


if __name__ == "__main__":
    main()
