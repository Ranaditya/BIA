# Iris Species Prediction - Decision Tree and Random Forest Models

## Project Overview

This project implements a modular machine learning workflow to predict Iris species using a Decision Tree classifier, with an optional Random Forest model for comparison. It follows the same step-by-step structure used in the other folders in this repository: data loading, exploratory data analysis, preprocessing, model training, and evaluation.

**Objective:** Build a clear, educational classification pipeline using pandas, seaborn, matplotlib, and scikit-learn.

---

## Dataset

**Source:** Local Iris CSV stored in `DecisionTreesEnsembleMethods/data/`

**Target Variable:** `species`

**Features Used:**
| Column | Type | Description |
|--------|------|-------------|
| sepal_length | float | Sepal length in cm |
| sepal_width | float | Sepal width in cm |
| petal_length | float | Petal length in cm |
| petal_width | float | Petal width in cm |

**Class Labels:**
- setosa
- versicolor
- virginica

**Data Quality:**
- No missing values in the CSV
- IQR-based outlier handling is applied during EDA
- Final clean dataset contains 146 rows after removing 4 outliers from `sepal_width`

---

## Project Structure

```
DecisionTreesEnsembleMethods/
├── data/
│   └── d56c70860c39452dae6bd08a137e2d7d_iris_dataset-1_(2).csv  # Raw Iris data
├── main.py                                                       # Entry point
├── data_loader.py                                                # Step 1: load and inspect data
├── eda.py                                                        # Step 2: pairplot, boxplots, outlier handling
├── preprocessing.py                                              # Step 3: feature/target selection and split
├── model.py                                                      # Step 4: train and evaluate models
├── images/                                                       # Generated plots
│   ├── eda_pairplot.png
│   ├── eda_boxplots_by_species.png
│   └── model_decision_tree.png
└── README.md                                                     # Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.13
- Virtual environment: `.venv-1`

### Required Libraries
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

These libraries are available in the workspace environment used for the project.

### Activate Virtual Environment
```powershell
# Run from the repository root
.\.venv-1\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Or run Python directly
c:\Users\vaidi\BIA\.venv-1\Scripts\python.exe
```

---

## How to Run

### Run the Complete Pipeline
```bash
cd DecisionTreesEnsembleMethods
python main.py
```

This executes the workflow in order:
1. Import libraries and load the dataset
2. Run EDA and IQR outlier handling
3. Select features and target, then split the data
4. Train and evaluate Decision Tree and Random Forest models
5. Visualize the trained Decision Tree structure

### Run Individual Modules
```bash
# Load and inspect data
python data_loader.py

# Run EDA only
python eda.py

# Run preprocessing only
python preprocessing.py

# Train and evaluate models only
python model.py
```

---

## Machine Learning Pipeline

### Step 1: Import Libraries and Load Dataset
- Import pandas, scikit-learn, matplotlib, and seaborn
- Load the Iris CSV using `pandas.read_csv()`
- Fallback support is available through `sklearn.datasets.load_iris`
- Inspect shape, data types, missing values, and the first few rows

**File:** `data_loader.py`

### Step 2: Exploratory Data Analysis (EDA)
**Visualizations:**
- Pairplot to visualize relationships between all features simultaneously
- Boxplots grouped by species to compare feature distributions

**Outlier Detection (IQR Method):**
- Compute Q1, Q3, and IQR for each numeric feature
- Remove rows outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
- Apply the filter feature-by-feature for cleaner modeling data

**File:** `eda.py`

### Step 3: Feature Selection and Train/Test Split
- Separate the independent variables from the target variable
- Use the four numeric features as predictors
- Convert the unique class labels into a sorted list for visualization
- Split the cleaned dataset into training and testing sets using `train_test_split`
- Use stratified splitting to keep class balance stable across splits

**Files:** `preprocessing.py`

### Step 4: Model Training and Evaluation
**Decision Tree Model:**
- Train a `DecisionTreeClassifier` on the training data
- Visualize the final tree with `plot_tree`
- Evaluate with `accuracy_score` and `classification_report`

**Random Forest Model:**
- Train a `RandomForestClassifier` as a comparison model
- Evaluate with the same metrics as the Decision Tree

**Model Outputs:**
- Accuracy score
- Precision, recall, and F1-score per class
- Decision tree visualization saved to `images/model_decision_tree.png`

**File:** `model.py`

---

## Key Findings

### Data Insights
- The Iris dataset contains 4 numeric predictor features and 3 species classes.
- `sepal_width` contained the only values removed by IQR filtering in this workflow.
- Pairplots and boxplots show that the species are separable, especially with petal measurements.

### Model Performance
- Both the Decision Tree and Random Forest models achieved strong test performance in the current run.
- The Decision Tree is easier to interpret because the tree structure can be directly visualized.
- The Random Forest is used as a comparison model and is typically more robust because it combines multiple trees.

---

## Expected Outputs

When you run `python main.py`, you should see:
- Console inspection of the dataset
- Pairplot and boxplot visualizations
- Outlier removal summary
- Train/test split summary
- Decision Tree and Random Forest evaluation metrics
- Decision tree plot saved to `images/model_decision_tree.png`

---

## Files & Functions

### `data_loader.py`
- `load_data(filepath, source)` - Load Iris dataset from CSV or sklearn fallback
- `inspect_data(df)` - Print dataset shape, types, missing values, and preview rows

### `eda.py`
- `plot_pairplot(df)` - Visualize pairwise feature relationships
- `plot_boxplots_by_species(df)` - Compare feature distributions across species
- `remove_outliers_iqr(df)` - Remove outliers using the IQR method
- `run_eda(df)` - Execute the full EDA workflow

### `preprocessing.py`
- `select_features_target(df)` - Split predictors and target
- `get_class_names(y)` - Convert unique class labels to a sorted list
- `split_data(X, y)` - Create train/test splits
- `prepare_data(df)` - Run the preprocessing workflow end-to-end

### `model.py`
- `train_decision_tree(X_train, y_train)` - Train a Decision Tree classifier
- `visualize_decision_tree(model, feature_names, class_names)` - Plot the tree
- `train_random_forest(X_train, y_train)` - Train a Random Forest classifier
- `evaluate_model(model_name, model, X_test, y_test, class_names)` - Print evaluation metrics
- `run_models(X_train, X_test, y_train, y_test, feature_names, class_names)` - Execute the model workflow

### `main.py`
- Orchestrates the full pipeline end-to-end

---

## Summary

This folder is structured as a complete, modular Iris classification project. It follows the same documentation and workflow style as the other machine learning folders in this repository, while focusing on Decision Tree and Random Forest classification.
