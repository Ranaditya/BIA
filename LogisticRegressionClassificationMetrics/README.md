# Titanic Survival Classification with Logistic Regression

This project implements a modular machine learning workflow for binary classification on the Titanic dataset.

## Workflow Overview

1. Import essentials (`numpy`, `pandas`, `scikit-learn`, plotting libraries)
2. Load Titanic dataset from `data/titanic.csv`
3. Preprocess data
   - Handle missing values (`Age` median, `Embarked` most frequent)
   - Remove duplicates
   - Cap numeric outliers with IQR (`Age`, `Fare`)
   - Encode categorical columns (`Sex`, `Embarked`) with one-hot encoding
4. Explore and visualize data
   - Descriptive statistics
   - Numeric histograms
   - Survival rate by category charts
5. Train logistic regression model
6. Evaluate with classification metrics
   - Accuracy, Precision, Recall, F1
   - Confusion matrix and classification report

## Project Structure

- `data_loader.py`: loading and inspection helpers
- `eda.py`: EDA summaries and plot generation
- `preprocessing.py`: cleaning, outlier handling, splitting, encoding
- `model.py`: model training and evaluation
- `main.py`: orchestration script
- `images/`: generated EDA visualizations

## Feature Set Used

- Included: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- Excluded: `PassengerId`, `Name`, `Ticket`, `Cabin`

## Run

From this folder, run:

```bash
python main.py
```

## Expected Outputs

- Console metrics: accuracy, precision, recall, f1, confusion matrix, classification report
- Plots:
  - `images/eda_numeric_histograms.png`
  - `images/eda_survival_by_category.png`
