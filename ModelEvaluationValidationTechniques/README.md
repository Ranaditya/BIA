# Apple Stock Price Prediction — Random Forest Classifier

## Project Overview

This project builds a **Random Forest Classifier** to predict whether Apple stock prices will increase or decrease based on historical stock data. The project demonstrates a complete machine learning workflow with hyperparameter tuning, cross-validation, and comprehensive evaluation metrics.

## Methodology

### 1. **Load Data**
   - Load Apple stock data from CSV using Pandas
   - File: `data_loader.py`

### 2. **Exploratory Data Analysis (EDA)**
   - Compute descriptive statistics
   - Visualize feature distributions
   - Analyze correlations between features
   - Detect outliers using box plots
   - File: `eda.py`

### 3. **Data Preprocessing**
   - Convert 'Date' column to datetime format
   - Handle missing values (forward fill for time series)
   - **Feature Engineering:**
     - Daily Returns: percentage change from previous close
     - Price Range: High - Low
     - Volume Change: percentage change in volume
     - Moving Averages (5-day, 20-day)
     - Relative Strength Index (RSI, 14-period)
   - Define target variable: Binary classification (0 = Price Decrease, 1 = Price Increase)
   - Scale numerical features using StandardScaler
   - Split data (80/20 train/test)
   - File: `preprocessing.py`

### 4. **Hyperparameter Tuning**
   - Use **GridSearchCV** to optimize Random Forest hyperparameters
   - Search space includes:
     - `n_estimators`: [50, 100, 200]
     - `max_depth`: [5, 10, 20, None]
     - `min_samples_split`: [2, 5, 10]
     - `min_samples_leaf`: [1, 2, 4]
     - `max_features`: ['sqrt', 'log2']
   - Cross-validation: 5-fold with F1-weighted scoring

### 5. **Cross-Validation**
   - Implement **K-Fold Cross-Validation** (k=5) on training data
   - Evaluate multiple metrics: Accuracy, Precision, Recall, F1-Score

### 6. **Model Training**
   - Train final Random Forest Classifier using best hyperparameters
   - Training set: 80% of preprocessed data

### 7. **Predictions**
   - Generate class predictions and probability predictions on test set

### 8. **Model Evaluation**
   - **Classification Metrics:**
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC Score
   - **Visualizations:**
     - Confusion Matrix
     - ROC Curve with AUC
     - Feature Importance (Top 10)
   - **Classification Report** with per-class metrics

## Project Structure

```
ModelEvaluationValidationTechniques/
├── main.py                         # Entry point (orchestrates pipeline)
├── data_loader.py                  # Data loading and inspection
├── eda.py                          # Exploratory Data Analysis
├── preprocessing.py                # Data preprocessing and feature engineering
├── model.py                        # Model training and evaluation
├── README.md                       # This file
├── data/
│   └── b5b3b342c8694c6bafd4f67814ce5220_apple_stocks.csv  # Apple stock data
└── images/
    ├── 01_distributions.png        # Feature distributions
    ├── 02_correlation_heatmap.png  # Feature correlations
    ├── 03_outlier_detection.png    # Box plots for outlier detection
    ├── 04_confusion_matrix.png     # Confusion matrix heatmap
    ├── 05_roc_curve.png            # ROC curve with AUC
    └── 06_feature_importance.png   # Top 10 feature importance
```

## How to Run

### Prerequisites
Ensure the following Python packages are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execute the Pipeline
From the project directory, run:
```bash
python main.py
```

### Output
The script will:
1. Load and inspect the Apple stock dataset
2. Perform EDA and generate visualizations
3. Preprocess data and engineer features
4. Perform hyperparameter tuning with GridSearchCV
5. Execute cross-validation (K-Fold)
6. Train the final model
7. Evaluate on test set
8. Generate classification metrics and visualizations
9. Save all plots to the `images/` directory

## Key Results Summary

The model's performance is displayed at the end of execution, showing:
- **Test Accuracy**: Overall correctness of predictions
- **Test Precision**: Proportion of positive predictions that are correct
- **Test Recall**: Proportion of actual positives correctly identified
- **Test F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under the ROC curve (model discrimination ability)

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates the complete ML pipeline |
| `data_loader.py` | Loads and inspects raw Apple stock data |
| `eda.py` | Generates statistical summaries and visualizations |
| `preprocessing.py` | Handles data cleaning, feature engineering, and train/test split |
| `model.py` | Implements hyperparameter tuning, training, and evaluation |

## Notes

- The target variable is binary: 1 (Price Increase) / 0 (Price Decrease)
- Stratified train/test split ensures balanced class representation
- All numerical features are scaled using StandardScaler
- Random state is set to 42 for reproducibility
- GridSearchCV uses 5-fold cross-validation with F1-weighted scoring
- Feature importance is derived from the trained Random Forest model

## Author & Date

Created: 2026-07-23

---

For questions or modifications, refer to individual module docstrings for detailed function documentation.
