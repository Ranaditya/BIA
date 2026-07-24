"""Run the complete Iris PCA and feature-engineering comparison pipeline."""

from data_loader import find_dataset, inspect_data, load_data
from eda import run_eda
from model import (
    evaluate_classifier,
    plot_feature_importance,
    plot_pca_projection,
    train_random_forest,
)
from preprocessing import (
    ENGINEERED_FEATURE,
    add_engineered_feature,
    create_split_indices,
    reduce_with_pca,
    split_features_target,
    standardize_split,
)


def main() -> None:
    """Execute all 12 requested steps in a leakage-safe order."""
    print("=" * 78)
    print("IRIS DIMENSIONALITY REDUCTION AND FEATURE ENGINEERING")
    print("=" * 78)

    print("\n[STEP 1] Import necessary libraries")
    print("NumPy, pandas, Matplotlib, seaborn, and scikit-learn are ready.")

    print("\n[STEP 2] Load the Iris dataset with pandas")
    dataset_path = find_dataset()
    df = load_data(dataset_path)
    print(f"Loaded: data/{dataset_path.name}")
    inspect_data(df)

    print("\n[STEP 3] Exploratory data analysis and visualization")
    generated_images = run_eda(df)

    print("\n[STEP 4] Split the dataset into features (X) and target (y)")
    X, y = split_features_target(df)
    print(f"X shape: {X.shape}; y shape: {y.shape}")
    print(f"Features: {', '.join(X.columns)}")
    print(f"Target: {y.name}")

    print("\n[STEP 5] Configure feature standardization")
    print("StandardScaler will be fitted only on training data to prevent data leakage.")

    print("\n[STEP 6] Split into training and testing sets")
    train_indices, test_indices = create_split_indices(y)
    original_split = standardize_split(X, y, train_indices, test_indices)
    print(f"Training samples: {len(train_indices)}")
    print(f"Testing samples: {len(test_indices)}")
    print("Split: 80/20, stratified by species, random_state=42")
    print("Standardization complete using training-set statistics.")

    print("\n[STEP 7] Reduce dimensionality using PCA")
    X_train_pca, X_test_pca, pca = reduce_with_pca(
        original_split.X_train,
        original_split.X_test,
        n_components=2,
    )
    variance_percent = pca.explained_variance_ratio_ * 100
    print(f"Reduced dimensions: {original_split.X_train.shape[1]} -> 2")
    print(
        "Explained variance: "
        f"PC1={variance_percent[0]:.2f}%, "
        f"PC2={variance_percent[1]:.2f}%, "
        f"total={variance_percent.sum():.2f}%"
    )
    generated_images.append(
        plot_pca_projection(
            X_train_pca,
            original_split.y_train,
            pca.explained_variance_ratio_,
        )
    )

    print("\n[STEP 8] Train Random Forest on PCA-reduced data")
    pca_model = train_random_forest(X_train_pca, original_split.y_train)
    print("RandomForestClassifier trained with 200 trees.")

    print("\n[STEP 9] Predict and evaluate after PCA")
    pca_metrics = evaluate_classifier(
        pca_model,
        X_test_pca,
        original_split.y_test,
        experiment_name="Random Forest after PCA",
        image_stem="pca",
    )
    generated_images.append(pca_metrics["image"])

    print("\n[STEP 10] Create sepal_length * petal_length feature")
    X_engineered = add_engineered_feature(X)
    print(f"Created: {ENGINEERED_FEATURE}")
    print(f"Feature count: {X.shape[1]} -> {X_engineered.shape[1]}")

    print("\n[STEP 11] Train Random Forest with the new feature")
    engineered_split = standardize_split(
        X_engineered,
        y,
        train_indices,
        test_indices,
    )
    engineered_model = train_random_forest(
        engineered_split.X_train,
        engineered_split.y_train,
    )
    print("RandomForestClassifier trained on the five standardized features.")

    print("\n[STEP 12] Predict and evaluate with the new feature")
    engineered_metrics = evaluate_classifier(
        engineered_model,
        engineered_split.X_test,
        engineered_split.y_test,
        experiment_name="Random Forest with Engineered Feature",
        image_stem="engineered_feature",
    )
    generated_images.append(engineered_metrics["image"])
    generated_images.append(
        plot_feature_importance(engineered_model, engineered_split.feature_names)
    )

    difference = engineered_metrics["accuracy"] - pca_metrics["accuracy"]
    print("\n" + "=" * 78)
    print("EXPERIMENT COMPARISON")
    print("=" * 78)
    print(f"PCA model accuracy               : {pca_metrics['accuracy']:.4f}")
    print(f"Engineered-feature model accuracy: {engineered_metrics['accuracy']:.4f}")
    print(f"Accuracy difference              : {difference:+.4f}")
    print(f"PCA explained variance (2 PCs)   : {variance_percent.sum():.2f}%")

    print("\n" + "=" * 78)
    print("REQUESTED RESULTS SUMMARY")
    print("=" * 78)
    print(f"Total original features used for modeling       : {X.shape[1]}")
    print(f"Standard deviation of sepal width (cm)          : {df['sepal_width'].std():.4f}")
    print(f"Features after feature engineering              : {X_engineered.shape[1]}")
    print(f"Principal components retained after PCA         : {pca.n_components_}")
    print(
        "Random Forest accuracy after feature engineering: "
        f"{engineered_metrics['accuracy']:.4f} "
        f"({engineered_metrics['accuracy']:.2%})"
    )
    print(f"Maximum petal width (cm)                         : {df['petal_width'].max():.1f}")
    print(
        "Random Forest accuracy after PCA                : "
        f"{pca_metrics['accuracy']:.4f} "
        f"({pca_metrics['accuracy']:.2%})"
    )
    print(f"Training-set samples                            : {len(train_indices)}")
    print(f"Minimum sepal length (cm)                       : {df['sepal_length'].min():.1f}")

    print("\nGenerated images:")
    for image in generated_images:
        print(f"  - {image.relative_to(image.parent.parent)}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
