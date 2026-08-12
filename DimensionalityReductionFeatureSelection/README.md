# Iris Dimensionality Reduction and Feature Selection

## Project overview

This project demonstrates two classification workflows on the Iris dataset:

1. Standardize the four original measurements, reduce them to two principal
   components with PCA, and train a Random Forest classifier.
2. Add the interaction feature `sepal_length * petal_length` and train a
   second Random Forest classifier using all five features.

Both experiments use the same stratified 80/20 split and random seed so their
test-set results can be compared directly. Preprocessing is fitted only on the
training set to prevent data leakage.

## Dataset

The CSV in `data/` contains 150 observations and these columns:

| Column | Role |
|---|---|
| `sepal_length` | Numeric feature (cm) |
| `sepal_width` | Numeric feature (cm) |
| `petal_length` | Numeric feature (cm) |
| `petal_width` | Numeric feature (cm) |
| `species` | Target: setosa, versicolor, or virginica |

## Project structure

```text
DimensionalityReductionFeatureSelection/
|-- data/
|   `-- 147635312288482289acb9a9db4e3ee9_iris_dataset_(1).csv
|-- images/                    # Created when the pipeline runs
|-- data_loader.py             # Dataset discovery, loading, validation, inspection
|-- eda.py                     # Statistics and EDA visualizations
|-- preprocessing.py           # Split, scaling, PCA, and feature engineering
|-- model.py                   # Random Forest training, metrics, model plots
|-- main.py                    # Orchestrates all 12 requested steps
`-- README.md
```

## Installation

From the BIA workspace:

```powershell
.\.venv-1\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the full pipeline

```powershell
cd .\DimensionalityReductionFeatureSelection
python .\main.py
```

The program prints:

- dataset shape, types, missing values, and species counts;
- descriptive statistics and species-level means;
- PCA explained-variance ratios;
- accuracy, precision, recall, F1-score, and confusion matrices;
- a direct accuracy comparison between both experiments.

## Run individual modules

```powershell
python .\data_loader.py
python .\eda.py
python .\preprocessing.py
```

## Generated outputs

The complete pipeline saves the following files under `images/`:

- `eda_histograms.png`
- `eda_boxplots.png`
- `eda_pairplot.png`
- `eda_correlation_heatmap.png`
- `pca_projection.png`
- `pca_confusion_matrix.png`
- `engineered_feature_confusion_matrix.png`
- `engineered_feature_importance.png`

## Implementation notes

- `random_state=42` makes the split and both Random Forest models reproducible.
- The split is stratified, giving every species proportional representation in
  the training and testing sets.
- Although the learning instructions list standardization before splitting,
  the implementation creates the split before fitting `StandardScaler`. This
  is the correct evaluation practice because test-set statistics never affect
  training.
- PCA is also fitted only on standardized training data. The fitted transforms
  are then applied to the test data.
- Random Forests do not require standardization, but the engineered-feature
  workflow retains it for a consistent preprocessing demonstration.
