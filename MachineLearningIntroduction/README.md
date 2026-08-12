# Housing Price Prediction — Simple Linear Regression Model

## Project Overview

This project implements a **Simple Linear Regression model** to predict housing prices based on property features. It serves as an educational introduction to the machine learning workflow, covering data exploration, preprocessing, model training, and evaluation.

**Objective:** Learn how to build an end-to-end machine learning pipeline using scikit-learn, pandas, and matplotlib.

---

## Dataset

**Source:** Custom housing dataset (`286623f0ccc34b76bd64f1d9899ba1cd_housing_dataset.csv`)

**Size:** 128 house records

**Features:**
| Column | Type | Description |
|--------|------|-------------|
| Home | int | House ID (dropped during preprocessing) |
| **Price** | int | **Target variable** — house price in USD |
| SqFt | int | Square footage of the house |
| Bedrooms | int | Number of bedrooms |
| Bathrooms | int | Number of bathrooms |
| Offers | int | Number of offers received |
| Brick | str | Whether the house is brick (Yes/No) |
| Neighborhood | str | Location: East, North, or West |

**Data Quality:**
- No missing values
- 3 outliers removed during EDA (Price and SqFt bounds)
- Final clean dataset: 125 samples

---

## Project Structure

```
MachineLearningIntroduction/
├── 286623f0ccc34b76bd64f1d9899ba1cd_housing_dataset.csv    # Raw data
├── main.py                                                  # Entry point — orchestrates full pipeline
├── data_loader.py                                           # Step 1-2: Load & inspect data
├── eda.py                                                   # Step 3-4: EDA & outlier detection
├── preprocessing.py                                         # Step 5: Feature engineering & scaling
├── model.py                                                 # Step 6-8: Split, train, evaluate
├── README.md                                                # Documentation (this file)
│
└── Generated Outputs/
    ├── eda_histograms.png                                   # Feature distributions
    ├── eda_scatter_plots.png                                # Feature vs Price relationships
    ├── eda_correlation_heatmap.png                          # Feature correlations
    ├── eda_boxplots.png                                     # Outlier detection
    ├── model_predictions.png                                # Predicted vs Actual prices
    └── model_residuals.png                                  # Residual analysis
```

---

## Installation & Setup

### Prerequisites
- Python 3.13
- Virtual environment (`.venv-1`)

### Required Libraries
```
pandas==3.0.3
numpy==2.5.1
scikit-learn==1.9.0
scipy==1.18.0
matplotlib==3.11.1
seaborn==0.13.2
statsmodels==0.14.6
```

All libraries are listed in `requirements.txt` and pre-installed in the venv.

### Activate Virtual Environment
```powershell
# From the repository root (PowerShell)
.\.venv-1\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Or use Python directly
c:\Users\vaidi\BIA\.venv-1\Scripts\python.exe
```

---

## How to Run

### Run the Complete Pipeline
```bash
cd MachineLearningIntroduction
python main.py
```

This executes all 6 steps sequentially:
1. ✓ Load dataset
2. ✓ Exploratory Data Analysis (EDA)
3. ✓ Data Preprocessing
4. ✓ Train/Test Split
5. ✓ Linear Regression Training
6. ✓ Model Evaluation & Visualization

### Run Individual Modules
```bash
# Load and inspect data
python data_loader.py

# Run EDA only
python eda.py

# Run preprocessing only
python preprocessing.py

# Train model only
python model.py
```

---

## Machine Learning Pipeline

### Step 1: Data Loading
- Load CSV into pandas DataFrame
- Inspect shape, dtypes, and missing values
- Preview first few rows

**File:** `data_loader.py`

### Step 2: Exploratory Data Analysis (EDA)
**Descriptive Statistics:**
- Count, mean, std, quartiles, min/max for each feature
- Value counts for categorical columns

**Visualizations:**
- Histograms — show feature distributions
- Scatter plots — reveal feature-target relationships
- Correlation heatmap — identify multicollinearity
- Box plots — detect outliers visually

**Outlier Detection (IQR Method):**
- Calculate Q1, Q3, IQR for each numeric column
- Flag points outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Remove 3 outlier rows from Price and SqFt

**File:** `eda.py`

### Step 3: Data Preprocessing
**Feature Engineering:**
1. **Drop irrelevant columns:** Remove `Home` (just an ID, no predictive value)
2. **Encode categorical features:**
   - `Brick`: Yes → 1, No → 0 (binary encoding)
   - `Neighborhood`: East → 0, North → 1, West → 2 (ordinal encoding)
3. **Standardize numerical features:**
   - Apply StandardScaler: (x - mean) / std
   - Ensures all features have mean=0, std=1
   - Improves model convergence and interpretability

**Output:** Feature matrix X (125 × 6) and target y (125,)

**File:** `preprocessing.py`

### Step 4: Train/Test Split
- **Ratio:** 80% training (100 samples), 20% testing (25 samples)
- **Random seed:** 42 (for reproducibility)
- **Purpose:** Evaluate model on unseen data

**File:** `model.py`

### Step 5: Linear Regression Training
**Model:** sklearn's `LinearRegression`

**Equation:** `Price = Intercept + Σ(Coefficient × Feature)`

**Trained Coefficients:**
```
Intercept       : $129,081.32
─────────────────────────────
SqFt            : $10,581.82     ← Largest driver of price
Neighborhood    :  $8,522.32
Brick           :  $8,475.40
Bathrooms       :  $4,596.43
Bedrooms        :  $4,490.05
Offers          : -$10,694.51     ← Negative: more offers → lower price
```

**Interpretation:**
- For every 100 sqft increase → **~$1,058 more in price**
- Brick homes command a **~$8,475 premium**
- Each extra bathroom adds **~$4,596**
- High number of offers suggests distressed sale → **lower price**

**File:** `model.py`

### Step 6: Model Evaluation
**Metrics on Test Set (25 unseen samples):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | $9,165 | On average, predictions are off by ~$9K |
| **MSE** | 133,347,746 | Squared error metric (harder to interpret) |
| **RMSE** | $11,548 | Dollar-based error (penalises large errors) |
| **R²** | 0.7928 | Model explains **79.28%** of price variance |

**Visualizations:**
- **Predicted vs Actual:** Scatter plot with diagonal reference line. Points close to the line = accurate predictions.
- **Residuals Plot:** Errors should be randomly scattered around zero. Patterns suggest the model is missing structure.

**File:** `model.py`

---

## Key Findings

### Feature Importance (by coefficient magnitude)
1. **SqFt** (+$10,582) — Largest positive influence
2. **Offers** (-$10,695) — Strong negative correlation (counterintuitive)
3. **Neighborhood** (+$8,522) — West > North > East
4. **Brick** (+$8,475) — Significant premium
5. **Bathrooms** (+$4,596) — Moderate positive influence
6. **Bedrooms** (+$4,490) — Smaller positive influence

### Model Performance
- **R² = 0.7928** indicates the model captures ~79% of price variance
- **RMSE = $11,548** means typical error is within ~$11.5K for a $130K average house (8.9% error)
- Model is **reasonably good** for a simple linear model
- **Improvement opportunities:** Non-linear relationships, interaction terms, additional features

### Data Quality Issues
- **Bedrooms** shows limited variance (mostly 3 bedrooms) → IQR-based outlier detection flagged many rows
- **Offers** seems counterintuitive (more offers → lower price) — may indicate distressed sales
- Small dataset (128 → 125 samples) — limited generalisability

---

## How to Interpret the Results

### Example Prediction
Suppose you have a house with:
- SqFt: 2,000
- Bedrooms: 3
- Bathrooms: 2
- Offers: 2
- Brick: Yes (encoded as 1)
- Neighborhood: West (encoded as 2)

After standardization, the model predicts the price. The prediction vs actual plot shows how close the model gets.

### Residual Analysis
- **Good residuals:** Randomly scattered around zero (no pattern)
- **Bad residuals:** Systematic pattern suggests the model is biased
- **Your model:** Appears reasonable with minor deviations

---

## Files & Functions

### `data_loader.py`
- `load_data(filepath)` — Load CSV into DataFrame
- `inspect_data(df)` — Print shape, dtypes, missing values, preview

### `eda.py`
- `descriptive_statistics(df)` — Print summary stats
- `plot_histograms(df)` — Visualise feature distributions
- `plot_scatter(df)` — Visualise feature-target relationships
- `plot_correlation_heatmap(df)` — Visualise feature correlations
- `plot_boxplots(df)` — Visualise outliers
- `detect_outliers_iqr(df)` — Flag outliers by IQR method
- `remove_outliers_iqr(df)` — Remove outlier rows
- `run_eda(df)` — Execute full EDA pipeline

### `preprocessing.py`
- `drop_irrelevant_columns(df)` — Remove ID columns
- `encode_categorical_features(df)` — Convert strings to numbers
- `scale_features(df)` — Standardise numerical features
- `preprocess(df)` — Execute full preprocessing pipeline

### `model.py`
- `split_data(X, y)` — 80/20 train/test split
- `train_model(X_train, y_train)` — Fit LinearRegression
- `evaluate_model(model, X_test, y_test)` — Calculate metrics
- `plot_predictions(y_test, y_pred)` — Visualise predictions
- `plot_residuals(y_test, y_pred)` — Visualise residuals
- `run_model(X, y)` — Execute full model pipeline

### `main.py`
- `main()` — Orchestrate the complete pipeline

---

## Next Steps & Improvements

### Model Enhancements
1. **Feature Engineering:**
   - Create interaction terms (SqFt × Bathrooms)
   - Add polynomial features (SqFt²)
   - Generate new features from existing ones

2. **Non-Linear Models:**
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Neural Networks

3. **Hyperparameter Tuning:**
   - Grid search / Random search
   - Cross-validation (k-fold)
   - Regularization (Ridge, Lasso)

4. **Address Data Issues:**
   - Investigate the negative Offers coefficient
   - Collect more data for better generalisation
   - Balance feature distributions

### Validation & Testing
- **Cross-validation:** Test model stability across different data splits
- **Train/test curves:** Diagnose overfitting vs underfitting
- **Residual diagnostics:** Check for heteroscedasticity

### Production Deployment
- Save trained model using `joblib` or `pickle`
- Create API endpoint for price predictions
- Monitor model performance on new data
- Retrain periodically with fresh data

---

## References & Learning Resources

### Linear Regression
- [Scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Understanding Linear Regression](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)

### Feature Scaling
- [StandardScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Why Feature Scaling Matters](https://towardsdatascience.com/feature-scaling-and-normalization-in-python-4c5b91e6c123)

### Evaluation Metrics
- [Regression Metrics Explained](https://towardsdatascience.com/regression-metrics-which-metric-should-you-use-d08f40bac0c0)
- [R-Squared Interpretation](https://www.investopedia.com/terms/r/r-squared.asp)

### Pandas & Scikit-learn
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## License

Educational project — free to use and modify for learning purposes.

---

## Contact & Questions

This project is part of the **BIA Machine Learning Introduction** series.

For questions or suggestions, refer to the inline documentation in each Python file.

---

**Last Updated:** 2026-07-20  
**Python Version:** 3.14.0  
**Scikit-learn Version:** 1.9.0
